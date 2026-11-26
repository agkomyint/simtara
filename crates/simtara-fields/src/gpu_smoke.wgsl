// The gas solver on the graphics card: the same steps as Fluid::smoke_substep on the CPU.
// Every kernel only gathers (reads around, writes its own element), sums are taken in a fixed
// tree, and the few emitter loops run on one thread, so a run repeats exactly on one GPU.

struct P {
    nx: u32, ny: u32, nz: u32, nc: u32,
    nu: u32, nv: u32, nw: u32, ng: u32,
    sel: u32, n_src: u32, flags: u32, nwg: u32,
    s: f32, sdt: f32, h: f32, g: f32,
    buoy: f32, weight: f32, eps: f32, cool: f32,
    fade: f32, nu0: f32, cs: f32, expn: f32,
    omega: f32, inflow: f32, lid: f32, rho0: f32,
    t0: f32, nu_cap: f32, fluid_cells: f32, kfac: f32,
    dif: array<vec4<f32>, 2>,
    mol: array<vec4<f32>, 2>,
};

@group(0) @binding(0) var<uniform> P_: P;
@group(0) @binding(1) var<storage, read_write> vel: array<f32>;
@group(0) @binding(2) var<storage, read_write> vel_old: array<f32>;
@group(0) @binding(3) var<storage, read_write> vel_tmp: array<f32>;
@group(0) @binding(4) var<storage, read_write> scal: array<f32>;
@group(0) @binding(5) var<storage, read_write> scal_old: array<f32>;
@group(0) @binding(6) var<storage, read_write> scal_tmp: array<f32>;
@group(0) @binding(7) var<storage, read_write> solid: array<u32>;
// per cell: temperature before the sources (0), eddy viscosity (1), density (2), divergence
// target (3), gas flow then pressure right-hand side (4), pressure (5)
@group(0) @binding(8) var<storage, read_write> cells: array<f32>;
// per cell: vorticity + its size (0..nc), confinement force (nc..2nc)
@group(0) @binding(9) var<storage, read_write> vecs: array<vec4<f32>>;
// emitters: 12 numbers per cell
@group(0) @binding(10) var<storage, read_write> src: array<f32>;
// partial sums (nwg), then 48 results
@group(0) @binding(11) var<storage, read_write> red: array<f32>;
// largest residual, speed and eddy viscosity (float bits)
@group(0) @binding(12) var<storage, read_write> stats: array<atomic<u32>>;

const R_GAS: f32 = 8.314462;
const P0: f32 = 101325.0;
const M_AIR: f32 = 0.028965;
const ALPHA_AIR: f32 = 2.2e-5;
const OPEN_TOP: u32 = 1u;
const OPEN_SIDE: u32 = 2u;
const INFLOW: u32 = 4u;
const NO_SLIP: u32 = 8u;

fn gid(id: vec3<u32>, n: vec3<u32>) -> u32 {
    return id.x + id.y * n.x * 64u;
}

fn ci(i: u32, j: u32, k: u32) -> u32 { return i + P_.nx * (j + P_.ny * k); }
fn ui(i: u32, j: u32, k: u32) -> u32 { return i + (P_.nx + 1u) * (j + P_.ny * k); }
fn vi(i: u32, j: u32, k: u32) -> u32 { return P_.nu + i + P_.nx * (j + (P_.ny + 1u) * k); }
fn wi(i: u32, j: u32, k: u32) -> u32 { return P_.nu + P_.nv + i + P_.nx * (j + P_.ny * k); }
fn is_solid(i: u32, j: u32, k: u32) -> bool { return solid[ci(i, j, k)] != 0u; }
fn has(f: u32) -> bool { return (P_.flags & f) != 0u; }

fn dims_of(axis: u32) -> vec3<u32> {
    if (axis == 0u) { return vec3<u32>(P_.nx + 1u, P_.ny, P_.nz); }
    if (axis == 1u) { return vec3<u32>(P_.nx, P_.ny + 1u, P_.nz); }
    return vec3<u32>(P_.nx, P_.ny, P_.nz + 1u);
}
fn base_of(axis: u32) -> u32 {
    if (axis == 0u) { return 0u; }
    if (axis == 1u) { return P_.nu; }
    return P_.nu + P_.nv;
}
fn off_of(axis: u32) -> vec3<f32> {
    if (axis == 0u) { return vec3<f32>(0.0, 0.5, 0.5); }
    if (axis == 1u) { return vec3<f32>(0.5, 0.0, 0.5); }
    return vec3<f32>(0.5, 0.5, 0.0);
}
// which axis face f is on, and its i, j, k
fn face(f: u32) -> vec4<u32> {
    var axis = 0u;
    var e = f;
    if (f >= P_.nu + P_.nv) { axis = 2u; e = f - P_.nu - P_.nv; } else if (f >= P_.nu) { axis = 1u; e = f - P_.nu; }
    let d = dims_of(axis);
    return vec4<u32>(e % d.x, (e / d.x) % d.y, e / (d.x * d.y), axis);
}

// the same clamped corner as grid::corner on the CPU
fn corner(x: f32, n: u32) -> vec2<f32> {
    let xc = clamp(x, 0.0, f32(n - 1u));
    var i = floor(xc);
    i = min(i, f32(max(n, 2u) - 2u));
    return vec2<f32>(i, xc - i);
}

struct Lerp { i: vec3<u32>, i1: vec3<u32>, t: vec3<f32> };
fn lerp_at(d: vec3<u32>, p: vec3<f32>) -> Lerp {
    let a = corner(p.x, d.x);
    let b = corner(p.y, d.y);
    let c = corner(p.z, d.z);
    var o: Lerp;
    o.i = vec3<u32>(u32(a.x), u32(b.x), u32(c.x));
    o.i1 = min(o.i + vec3<u32>(1u), d - vec3<u32>(1u));
    o.t = vec3<f32>(a.y, b.y, c.y);
    return o;
}
fn li(d: vec3<u32>, i: u32, j: u32, k: u32) -> u32 { return i + d.x * (j + d.y * k); }

fn mix8(v: array<f32, 8>, t: vec3<f32>) -> f32 {
    let x00 = v[0] * (1.0 - t.x) + v[1] * t.x;
    let x10 = v[2] * (1.0 - t.x) + v[3] * t.x;
    let x01 = v[4] * (1.0 - t.x) + v[5] * t.x;
    let x11 = v[6] * (1.0 - t.x) + v[7] * t.x;
    let y0 = x00 * (1.0 - t.y) + x10 * t.y;
    let y1 = x01 * (1.0 - t.y) + x11 * t.y;
    return y0 * (1.0 - t.z) + y1 * t.z;
}
fn range8(v: array<f32, 8>) -> vec2<f32> {
    let lo = min(min(min(v[0], v[1]), min(v[2], v[3])), min(min(v[4], v[5]), min(v[6], v[7])));
    let hi = max(max(max(v[0], v[1]), max(v[2], v[3])), max(max(v[4], v[5]), max(v[6], v[7])));
    return vec2<f32>(lo, hi);
}

fn eight_vo(base: u32, d: vec3<u32>, l: Lerp) -> array<f32, 8> {
    return array<f32, 8>(
        vel_old[base + li(d, l.i.x, l.i.y, l.i.z)], vel_old[base + li(d, l.i1.x, l.i.y, l.i.z)],
        vel_old[base + li(d, l.i.x, l.i1.y, l.i.z)], vel_old[base + li(d, l.i1.x, l.i1.y, l.i.z)],
        vel_old[base + li(d, l.i.x, l.i.y, l.i1.z)], vel_old[base + li(d, l.i1.x, l.i.y, l.i1.z)],
        vel_old[base + li(d, l.i.x, l.i1.y, l.i1.z)], vel_old[base + li(d, l.i1.x, l.i1.y, l.i1.z)]);
}
fn eight_vt(base: u32, d: vec3<u32>, l: Lerp) -> array<f32, 8> {
    return array<f32, 8>(
        vel_tmp[base + li(d, l.i.x, l.i.y, l.i.z)], vel_tmp[base + li(d, l.i1.x, l.i.y, l.i.z)],
        vel_tmp[base + li(d, l.i.x, l.i1.y, l.i.z)], vel_tmp[base + li(d, l.i1.x, l.i1.y, l.i.z)],
        vel_tmp[base + li(d, l.i.x, l.i.y, l.i1.z)], vel_tmp[base + li(d, l.i1.x, l.i.y, l.i1.z)],
        vel_tmp[base + li(d, l.i.x, l.i1.y, l.i1.z)], vel_tmp[base + li(d, l.i1.x, l.i1.y, l.i1.z)]);
}
fn eight_so(base: u32, d: vec3<u32>, l: Lerp) -> array<f32, 8> {
    return array<f32, 8>(
        scal_old[base + li(d, l.i.x, l.i.y, l.i.z)], scal_old[base + li(d, l.i1.x, l.i.y, l.i.z)],
        scal_old[base + li(d, l.i.x, l.i1.y, l.i.z)], scal_old[base + li(d, l.i1.x, l.i1.y, l.i.z)],
        scal_old[base + li(d, l.i.x, l.i.y, l.i1.z)], scal_old[base + li(d, l.i1.x, l.i.y, l.i1.z)],
        scal_old[base + li(d, l.i.x, l.i1.y, l.i1.z)], scal_old[base + li(d, l.i1.x, l.i1.y, l.i1.z)]);
}
fn eight_st(base: u32, d: vec3<u32>, l: Lerp) -> array<f32, 8> {
    return array<f32, 8>(
        scal_tmp[base + li(d, l.i.x, l.i.y, l.i.z)], scal_tmp[base + li(d, l.i1.x, l.i.y, l.i.z)],
        scal_tmp[base + li(d, l.i.x, l.i1.y, l.i.z)], scal_tmp[base + li(d, l.i1.x, l.i1.y, l.i.z)],
        scal_tmp[base + li(d, l.i.x, l.i.y, l.i1.z)], scal_tmp[base + li(d, l.i1.x, l.i.y, l.i1.z)],
        scal_tmp[base + li(d, l.i.x, l.i1.y, l.i1.z)], scal_tmp[base + li(d, l.i1.x, l.i1.y, l.i1.z)]);
}

// velocity (of the old field) at a grid position
fn vel_at(p: vec3<f32>) -> vec3<f32> {
    var out = vec3<f32>(0.0);
    for (var a = 0u; a < 3u; a = a + 1u) {
        let d = dims_of(a);
        let l = lerp_at(d, p - off_of(a));
        out[a] = mix8(eight_vo(base_of(a), d, l), l.t);
    }
    return out;
}
fn back(p: vec3<f32>, s: f32) -> vec3<f32> {
    let v = vel_at(p);
    let mid = p - 0.5 * s * v;
    let v2 = vel_at(mid);
    return p - s * v2;
}

@compute @workgroup_size(64) fn save_vel(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let f = gid(id, n);
    if (f >= P_.nu + P_.nv + P_.nw) { return; }
    vel_old[f] = vel[f];
}
@compute @workgroup_size(64) fn save_scal(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let e = gid(id, n);
    if (e >= P_.nc * (2u + P_.ng)) { return; }
    scal_old[e] = scal[e];
}

// ---- MacCormack advection of the face velocities ----
@compute @workgroup_size(64) fn adv_vel_a(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let f = gid(id, n);
    if (f >= P_.nu + P_.nv + P_.nw) { return; }
    let q = face(f);
    let off = off_of(q.w);
    let p = vec3<f32>(f32(q.x), f32(q.y), f32(q.z)) + off;
    let d = dims_of(q.w);
    let l = lerp_at(d, back(p, P_.s) - off);
    vel_tmp[f] = mix8(eight_vo(base_of(q.w), d, l), l.t);
}
@compute @workgroup_size(64) fn adv_vel_b(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let f = gid(id, n);
    if (f >= P_.nu + P_.nv + P_.nw) { return; }
    let q = face(f);
    let off = off_of(q.w);
    let p = vec3<f32>(f32(q.x), f32(q.y), f32(q.z)) + off;
    let d = dims_of(q.w);
    let base = base_of(q.w);
    let lb = lerp_at(d, back(p, P_.s) - off);
    let lf = lerp_at(d, back(p, -P_.s) - off);
    let a = vel_tmp[f];
    let b = mix8(eight_vt(base, d, lf), lf.t);
    let v = a + 0.5 * (vel_old[f] - b);
    let r = range8(eight_vo(base, d, lb));
    vel[f] = select(v, a, v < r.x || v > r.y);
}

// ---- MacCormack advection of smoke, heat and the gases ----
fn cell_pos(c: u32) -> vec3<f32> {
    return vec3<f32>(f32(c % P_.nx), f32((c / P_.nx) % P_.ny), f32(c / (P_.nx * P_.ny))) + vec3<f32>(0.5);
}
@compute @workgroup_size(64) fn adv_scal_a(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let e = gid(id, n);
    if (e >= P_.nc * (2u + P_.ng)) { return; }
    let c = e % P_.nc;
    let base = e - c;
    let d = vec3<u32>(P_.nx, P_.ny, P_.nz);
    let l = lerp_at(d, back(cell_pos(c), P_.s) - vec3<f32>(0.5));
    scal_tmp[e] = mix8(eight_so(base, d, l), l.t);
}
@compute @workgroup_size(64) fn adv_scal_b(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let e = gid(id, n);
    if (e >= P_.nc * (2u + P_.ng)) { return; }
    let c = e % P_.nc;
    let base = e - c;
    let d = vec3<u32>(P_.nx, P_.ny, P_.nz);
    let p = cell_pos(c);
    let lb = lerp_at(d, back(p, P_.s) - vec3<f32>(0.5));
    let lf = lerp_at(d, back(p, -P_.s) - vec3<f32>(0.5));
    let a = scal_tmp[e];
    let b = mix8(eight_st(base, d, lf), lf.t);
    let v = a + 0.5 * (scal_old[e] - b);
    let r = range8(eight_so(base, d, lb));
    scal[e] = select(v, a, v < r.x || v > r.y);
}

// ---- sums in a fixed order (two passes) ----
var<workgroup> sh: array<f32, 256>;

fn div_old(c: u32) -> f32 {
    let i = c % P_.nx;
    let j = (c / P_.nx) % P_.ny;
    let k = c / (P_.nx * P_.ny);
    return vel_old[ui(i + 1u, j, k)] - vel_old[ui(i, j, k)] + vel_old[vi(i, j + 1u, k)] - vel_old[vi(i, j, k)] + vel_old[wi(i, j, k + 1u)] - vel_old[wi(i, j, k)];
}

fn sum_value(c: u32) -> f32 {
    let sel = P_.sel;
    if (sel < 16u) {
        // a gas before it is carried: its amount, grown where the flow spreads, less what leaves
        let x = scal_old[(2u + sel) * P_.nc + c];
        if (x == 0.0) { return 0.0; }
        var v = x * (1.0 + div_old(c) * P_.s);
        let i = c % P_.nx;
        let j = (c / P_.nx) % P_.ny;
        let k = c / (P_.nx * P_.ny);
        if (has(OPEN_TOP) && k == P_.nz - 1u) {
            let w = vel_old[wi(i, j, P_.nz)];
            if (w > 0.0) { v = v - x * min(w * P_.s, 1.0); }
        }
        if (has(OPEN_SIDE) && i == P_.nx - 1u) {
            let u = vel_old[ui(P_.nx, j, k)];
            if (u > 0.0) { v = v - x * min(u * P_.s, 1.0); }
        }
        return v;
    }
    if (sel < 32u) {
        return scal[(2u + sel - 16u) * P_.nc + c];
    }
    if (solid[c] != 0u) { return 0.0; }
    return cells[3u * P_.nc + c];
}

@compute @workgroup_size(256) fn sum_a(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let w = wg.x + wg.y * n.x;
    let c = w * 256u + l;
    var v = 0.0;
    if (c < P_.nc) { v = sum_value(c); }
    sh[l] = v;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (l < st) { sh[l] = sh[l] + sh[l + st]; }
        workgroupBarrier();
    }
    if (l == 0u) { red[w] = sh[0]; }
}
@compute @workgroup_size(256) fn sum_b(@builtin(local_invocation_index) l: u32) {
    var v = 0.0;
    for (var i = l; i < P_.nwg; i = i + 256u) { v = v + red[i]; }
    sh[l] = v;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (l < st) { sh[l] = sh[l] + sh[l + st]; }
        workgroupBarrier();
    }
    if (l == 0u) { red[P_.nwg + P_.sel] = sh[0]; }
}

// each gas back to the amount it must have (advection does not keep it exactly)
@compute @workgroup_size(64) fn rescale(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let c = gid(id, n);
    if (c >= P_.nc) { return; }
    for (var g = 0u; g < P_.ng; g = g + 1u) {
        let want = max(red[P_.nwg + g], 0.0);
        let have = red[P_.nwg + 16u + g];
        if (have > 1e-12 && want > 0.0) {
            let e = (2u + g) * P_.nc + c;
            scal[e] = min(scal[e] * (want / have), 1.0);
        }
    }
}

// ---- emitters: one thread walks them in order, exactly as the CPU does ----
@compute @workgroup_size(64) fn clear_flow(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let c = gid(id, n);
    if (c >= P_.nc) { return; }
    cells[4u * P_.nc + c] = 0.0;
    cells[c] = scal[P_.nc + c];
}
@compute @workgroup_size(1) fn src_gas() {
    for (var r = 0u; r < P_.n_src; r = r + 1u) {
        let c = bitcast<u32>(src[r * 12u]);
        let gas = src[r * 12u + 3u];
        let amount = src[r * 12u + 4u];
        let flow = src[r * 12u + 5u];
        if (gas >= 0.0) {
            let g = u32(gas);
            if (flow > 0.0) {
                let q = flow * P_.sdt;
                let f = q / (1.0 + q);
                for (var o = 0u; o < P_.ng; o = o + 1u) {
                    let e = (2u + o) * P_.nc + c;
                    var add = 0.0;
                    if (o == g) { add = amount * f; }
                    scal[e] = scal[e] * (1.0 - f) + add;
                }
            } else {
                let eg = (2u + g) * P_.nc + c;
                if (scal[eg] < amount) {
                    scal[eg] = amount;
                    var others = 0.0;
                    for (var o = 0u; o < P_.ng; o = o + 1u) {
                        if (o != g) { others = others + scal[(2u + o) * P_.nc + c]; }
                    }
                    if (others + amount > 1.0 && others > 0.0) {
                        let k = (1.0 - amount) / others;
                        for (var o = 0u; o < P_.ng; o = o + 1u) {
                            if (o != g) { scal[(2u + o) * P_.nc + c] = scal[(2u + o) * P_.nc + c] * k; }
                        }
                    }
                }
            }
        }
        cells[4u * P_.nc + c] = cells[4u * P_.nc + c] + flow;
    }
}

@compute @workgroup_size(1) fn src_heat() {
    for (var r = 0u; r < P_.n_src; r = r + 1u) {
        let c = bitcast<u32>(src[r * 12u]);
        scal[c] = scal[c] + src[r * 12u + 1u] * P_.sdt;
        scal[P_.nc + c] = max(scal[P_.nc + c], src[r * 12u + 2u]);
    }
}
@compute @workgroup_size(1) fn src_vel() {
    for (var r = 0u; r < P_.n_src; r = r + 1u) {
        if (src[r * 12u + 9u] < 0.5) { continue; }
        let c = bitcast<u32>(src[r * 12u]);
        let i = c % P_.nx;
        let j = (c / P_.nx) % P_.ny;
        let k = c / (P_.nx * P_.ny);
        let v = vec3<f32>(src[r * 12u + 6u], src[r * 12u + 7u], src[r * 12u + 8u]);
        vel[ui(i, j, k)] = v.x;
        vel[ui(i + 1u, j, k)] = v.x;
        vel[vi(i, j, k)] = v.y;
        vel[vi(i, j + 1u, k)] = v.y;
        vel[wi(i, j, k)] = v.z;
        vel[wi(i, j, k + 1u)] = v.z;
    }
}

// ---- turbulence: Smagorinsky eddy viscosity ----
fn centre(i: u32, j: u32, k: u32) -> vec3<f32> {
    return vec3<f32>(0.5 * (vel[ui(i, j, k)] + vel[ui(i + 1u, j, k)]), 0.5 * (vel[vi(i, j, k)] + vel[vi(i, j + 1u, k)]), 0.5 * (vel[wi(i, j, k)] + vel[wi(i, j, k + 1u)]));
}
fn nb_cell(i: u32, j: u32, k: u32, axis: u32, up: bool) -> vec3<u32> {
    var q = vec3<u32>(i, j, k);
    let n = vec3<u32>(P_.nx, P_.ny, P_.nz);
    if (up) { q[axis] = min(q[axis] + 1u, n[axis] - 1u); } else { q[axis] = max(q[axis], 1u) - 1u; }
    return q;
}
@compute @workgroup_size(64) fn nut(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let c = gid(id, n);
    if (c >= P_.nc) { return; }
    if (solid[c] != 0u || P_.cs <= 0.0) { cells[P_.nc + c] = 0.0; return; }
    let i = c % P_.nx;
    let j = (c / P_.nx) % P_.ny;
    let k = c / (P_.nx * P_.ny);
    let h = P_.h;
    let dudx = (vel[ui(i + 1u, j, k)] - vel[ui(i, j, k)]) / h;
    let dvdy = (vel[vi(i, j + 1u, k)] - vel[vi(i, j, k)]) / h;
    let dwdz = (vel[wi(i, j, k + 1u)] - vel[wi(i, j, k)]) / h;
    var gr: array<vec3<f32>, 3>;
    for (var a = 0u; a < 3u; a = a + 1u) {
        let lo = nb_cell(i, j, k, a, false);
        let hi = nb_cell(i, j, k, a, true);
        let span = f32(hi[a] - lo[a]);
        if (span == 0.0) { gr[a] = vec3<f32>(0.0); } else { gr[a] = (centre(hi.x, hi.y, hi.z) - centre(lo.x, lo.y, lo.z)) / (span * h); }
    }
    // gr[axis][component]
    let sxy = 0.5 * (gr[1].x + gr[0].y);
    let sxz = 0.5 * (gr[2].x + gr[0].z);
    let syz = 0.5 * (gr[2].y + gr[1].z);
    let s2 = 2.0 * (dudx * dudx + dvdy * dvdy + dwdz * dwdz) + 4.0 * (sxy * sxy + sxz * sxz + syz * syz);
    let v = (P_.cs * h) * (P_.cs * h) * sqrt(s2);
    atomicMax(&stats[2], bitcast<u32>(v));
    cells[P_.nc + c] = min(v, P_.nu_cap);
}

// ---- diffusion of one quantity (sel = which) ----
fn k_of(field: u32) -> f32 {
    if (field == 0u) { return 0.0; }
    if (field == 1u) { return ALPHA_AIR; }
    let g = field - 2u;
    return P_.dif[g / 4u][g % 4u];
}
@compute @workgroup_size(64) fn diffuse_scal(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let c = gid(id, n);
    if (c >= P_.nc) { return; }
    let base = P_.sel * P_.nc;
    let me = scal[base + c];
    if (solid[c] != 0u) { scal_tmp[base + c] = me; return; }
    let k0 = k_of(P_.sel);
    let kc = k0 + 2.0 * cells[P_.nc + c];
    let i = c % P_.nx;
    let j = (c / P_.nx) % P_.ny;
    let k = c / (P_.nx * P_.ny);
    var flux = 0.0;
    for (var a = 0u; a < 3u; a = a + 1u) {
        for (var s = 0u; s < 2u; s = s + 1u) {
            let up = s == 1u;
            var q = vec3<u32>(i, j, k);
            let n3 = vec3<u32>(P_.nx, P_.ny, P_.nz);
            if (up) { if (q[a] + 1u >= n3[a]) { continue; } q[a] = q[a] + 1u; } else { if (q[a] == 0u) { continue; } q[a] = q[a] - 1u; }
            let o = ci(q.x, q.y, q.z);
            if (solid[o] != 0u) { continue; }
            let kf = 0.5 * (kc + k0 + 2.0 * cells[P_.nc + o]);
            flux = flux + kf * (scal[base + o] - me);
        }
    }
    scal_tmp[base + c] = me + P_.kfac * flux;
}
@compute @workgroup_size(64) fn copy_scal(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let c = gid(id, n);
    if (c >= P_.nc) { return; }
    scal[P_.sel * P_.nc + c] = scal_tmp[P_.sel * P_.nc + c];
}

// ---- cooling, thinning, nothing in solids; then the density ----
@compute @workgroup_size(64) fn cool(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let c = gid(id, n);
    if (c >= P_.nc) { return; }
    let s = solid[c] != 0u;
    if (s) {
        scal[c] = 0.0;
        scal[P_.nc + c] = 0.0;
    } else {
        scal[P_.nc + c] = scal[P_.nc + c] * P_.cool;
        scal[c] = max(scal[c] * P_.fade, 0.0);
    }
    var rest = 1.0;
    var m = 0.0;
    for (var g = 0u; g < P_.ng; g = g + 1u) {
        let e = (2u + g) * P_.nc + c;
        var x = clamp(scal[e], 0.0, 1.0);
        if (s) { x = 0.0; }
        scal[e] = x;
        rest = rest - x;
        m = m + x * P_.mol[g / 4u][g % 4u];
    }
    m = m + max(rest, 0.0) * M_AIR;
    cells[2u * P_.nc + c] = P0 * m / (R_GAS * max(P_.t0 + scal[P_.nc + c], 50.0));
}

fn rho_face(axis: u32, i: u32, j: u32, k: u32) -> f32 {
    // face (i, j, k) of an axis: between cell (.. - 1) and cell (..), as gas::faces
    let n3 = vec3<u32>(P_.nx, P_.ny, P_.nz);
    var q = vec3<u32>(i, j, k);
    let has_lo = q[axis] > 0u;
    let has_hi = q[axis] < n3[axis];
    var lo = q;
    if (has_lo) { lo[axis] = q[axis] - 1u; }
    let rl = cells[2u * P_.nc + ci(lo.x, lo.y, lo.z)];
    if (has_lo && has_hi) { return 0.5 * (rl + cells[2u * P_.nc + ci(q.x, q.y, q.z)]); }
    if (has_lo) { return rl; }
    if (has_hi) { return cells[2u * P_.nc + ci(q.x, q.y, q.z)]; }
    return P_.rho0;
}

@compute @workgroup_size(64) fn buoy(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let f = gid(id, n);
    if (f < P_.nu + P_.nv || f >= P_.nu + P_.nv + P_.nw) { return; }
    let q = face(f);
    if (q.z == 0u || q.z == P_.nz) { return; }
    let rf = rho_face(2u, q.x, q.y, q.z);
    let s = 0.5 * (scal[ci(q.x, q.y, q.z - 1u)] + scal[ci(q.x, q.y, q.z)]);
    vel[f] = vel[f] + P_.sdt * (P_.buoy * P_.g * (P_.rho0 - rf) / rf - P_.weight * s);
}

// ---- vorticity confinement ----
@compute @workgroup_size(64) fn vort(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let c = gid(id, n);
    if (c >= P_.nc) { return; }
    let i = c % P_.nx;
    let j = (c / P_.nx) % P_.ny;
    let k = c / (P_.nx * P_.ny);
    var dv: array<vec3<f32>, 3>;
    for (var a = 0u; a < 3u; a = a + 1u) {
        let hi = nb_cell(i, j, k, a, true);
        let lo = nb_cell(i, j, k, a, false);
        dv[a] = (centre(hi.x, hi.y, hi.z) - centre(lo.x, lo.y, lo.z)) / (2.0 * P_.h);
    }
    // dv[axis][component]
    let w = vec3<f32>(dv[1].z - dv[2].y, dv[2].x - dv[0].z, dv[0].y - dv[1].x);
    vecs[c] = vec4<f32>(w, length(w));
}
@compute @workgroup_size(64) fn vforce(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let c = gid(id, n);
    if (c >= P_.nc) { return; }
    var out = vec4<f32>(0.0);
    if (solid[c] == 0u) {
        let i = c % P_.nx;
        let j = (c / P_.nx) % P_.ny;
        let k = c / (P_.nx * P_.ny);
        var g = vec3<f32>(0.0);
        for (var a = 0u; a < 3u; a = a + 1u) {
            let hi = nb_cell(i, j, k, a, true);
            let lo = nb_cell(i, j, k, a, false);
            g[a] = vecs[ci(hi.x, hi.y, hi.z)].w - vecs[ci(lo.x, lo.y, lo.z)].w;
        }
        let l = length(g);
        if (l >= 1e-9) {
            let nn = g / l;
            let w = vecs[c].xyz;
            out = vec4<f32>(P_.eps * P_.h * cross(nn, w), 0.0);
        }
    }
    vecs[P_.nc + c] = out;
}
@compute @workgroup_size(64) fn vapply(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let f = gid(id, n);
    if (f >= P_.nu + P_.nv + P_.nw) { return; }
    let q = face(f);
    let a = q.w;
    let p = vec3<u32>(q.x, q.y, q.z);
    let n3 = vec3<u32>(P_.nx, P_.ny, P_.nz);
    if (p[a] == 0u || p[a] >= n3[a]) { return; }
    var lo = p;
    lo[a] = p[a] - 1u;
    let fa = vecs[P_.nc + ci(lo.x, lo.y, lo.z)][a];
    let fb = vecs[P_.nc + ci(p.x, p.y, p.z)][a];
    vel[f] = vel[f] + P_.sdt * 0.5 * (fa + fb);
}

// ---- viscosity (air's own plus the eddies'), walls slip or stick, the lid may slide ----
@compute @workgroup_size(64) fn diffuse_vel(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let f = gid(id, n);
    if (f >= P_.nu + P_.nv + P_.nw) { return; }
    let q = face(f);
    let a = q.w;
    let p = vec3<u32>(q.x, q.y, q.z);
    let d = dims_of(a);
    let me = vel[f];
    vel_tmp[f] = me;
    if (p[a] == 0u || p[a] == d[a] - 1u) { return; }
    var lo = p;
    lo[a] = p[a] - 1u;
    let cl = ci(lo.x, lo.y, lo.z);
    let ch = ci(p.x, p.y, p.z);
    if (solid[cl] != 0u || solid[ch] != 0u) { return; }
    let nu = P_.nu0 + 0.5 * (cells[P_.nc + cl] + cells[P_.nc + ch]);
    let base = base_of(a);
    let n3 = vec3<u32>(P_.nx, P_.ny, P_.nz);
    var lap = 0.0;
    for (var b = 0u; b < 3u; b = b + 1u) {
        for (var s = 0u; s < 2u; s = s + 1u) {
            let up = s == 1u;
            var inside = p[b] > 0u;
            if (up) { inside = p[b] + 1u < d[b]; }
            var v = me;
            if (inside) {
                var o = p;
                if (up) { o[b] = p[b] + 1u; } else { o[b] = p[b] - 1u; }
                v = vel[base + li(d, o.x, o.y, o.z)];
            } else if (b != a) {
                let open = (b == 2u && up && has(OPEN_TOP)) || (b == 0u && up && has(OPEN_SIDE)) || n3[b] == 1u;
                var wall = 0.0;
                if (a == 0u && b == 2u && up) { wall = P_.lid; }
                if (has(NO_SLIP) && !open) { v = 2.0 * wall - me; }
            }
            lap = lap + v - me;
        }
    }
    vel_tmp[f] = me + nu * P_.kfac * lap;
}
@compute @workgroup_size(64) fn copy_vel(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let f = gid(id, n);
    if (f >= P_.nu + P_.nv + P_.nw) { return; }
    vel[f] = vel_tmp[f];
}

// ---- walls, solids and the inflow (grid::enforce) ----
@compute @workgroup_size(64) fn enforce(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let f = gid(id, n);
    if (f >= P_.nu + P_.nv + P_.nw) { return; }
    let q = face(f);
    let i = q.x;
    let j = q.y;
    let k = q.z;
    if (q.w == 0u) {
        if (i == 0u) {
            if (is_solid(0u, j, k)) { vel[f] = 0.0; } else if (has(INFLOW)) { vel[f] = P_.inflow; } else { vel[f] = 0.0; }
        } else if (i == P_.nx) {
            if (!has(OPEN_SIDE) || is_solid(i - 1u, j, k)) { vel[f] = 0.0; }
        } else if (is_solid(i - 1u, j, k) || is_solid(i, j, k)) {
            vel[f] = 0.0;
        }
    } else if (q.w == 1u) {
        if (j == 0u || j == P_.ny || is_solid(i, j - 1u, k) || is_solid(i, j, k)) { vel[f] = 0.0; }
    } else {
        if (k == 0u) {
            vel[f] = 0.0;
        } else if (k == P_.nz) {
            if (!has(OPEN_TOP) || is_solid(i, j, k - 1u)) { vel[f] = 0.0; }
        } else if (is_solid(i, j, k - 1u) || is_solid(i, j, k)) {
            vel[f] = 0.0;
        }
    }
}

// ---- the divergence each cell should have: heated air grows, leaks push gas out ----
@compute @workgroup_size(64) fn div_target(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let c = gid(id, n);
    if (c >= P_.nc) { return; }
    var t = 0.0;
    if (solid[c] == 0u) {
        let ln_t = log(max(P_.t0 + scal[P_.nc + c], 50.0) / max(P_.t0 + cells[c], 50.0));
        t = (P_.expn * ln_t / P_.sdt + cells[4u * P_.nc + c]) * P_.h;
    }
    cells[3u * P_.nc + c] = t;
}
@compute @workgroup_size(64) fn sub_mean(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let c = gid(id, n);
    if (c >= P_.nc || solid[c] != 0u) { return; }
    cells[3u * P_.nc + c] = cells[3u * P_.nc + c] - red[P_.nwg + 32u] / P_.fluid_cells;
}

fn div_now(i: u32, j: u32, k: u32) -> f32 {
    return vel[ui(i + 1u, j, k)] - vel[ui(i, j, k)] + vel[vi(i, j + 1u, k)] - vel[vi(i, j, k)] + vel[wi(i, j, k + 1u)] - vel[wi(i, j, k)];
}
@compute @workgroup_size(64) fn prep(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let c = gid(id, n);
    if (c >= P_.nc) { return; }
    let i = c % P_.nx;
    let j = (c / P_.nx) % P_.ny;
    let k = c / (P_.nx * P_.ny);
    cells[4u * P_.nc + c] = -(div_now(i, j, k) - cells[3u * P_.nc + c]);
    if (solid[c] != 0u) { cells[5u * P_.nc + c] = 0.0; }
}

// ---- pressure: red-black over-relaxation with the 1/rho face weights ----
@compute @workgroup_size(64) fn sor(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let c = gid(id, n);
    if (c >= P_.nc) { return; }
    let i = c % P_.nx;
    let j = (c / P_.nx) % P_.ny;
    let k = c / (P_.nx * P_.ny);
    if (((i + j + k) & 1u) != (P_.sel & 1u) || solid[c] != 0u) { return; }
    let n3 = vec3<u32>(P_.nx, P_.ny, P_.nz);
    let p = vec3<u32>(i, j, k);
    var diag = 0.0;
    var sum = 0.0;
    for (var a = 0u; a < 3u; a = a + 1u) {
        for (var s = 0u; s < 2u; s = s + 1u) {
            let up = s == 1u;
            var fp = p;
            if (up) { fp[a] = p[a] + 1u; }
            let b = P_.rho0 / rho_face(a, fp.x, fp.y, fp.z);
            var inside = p[a] > 0u;
            if (up) { inside = p[a] + 1u < n3[a]; }
            if (inside) {
                var o = p;
                if (up) { o[a] = p[a] + 1u; } else { o[a] = p[a] - 1u; }
                let oc = ci(o.x, o.y, o.z);
                if (solid[oc] == 0u) {
                    diag = diag + b;
                    sum = sum + b * cells[5u * P_.nc + oc];
                }
            } else if (up && ((a == 0u && has(OPEN_SIDE)) || (a == 2u && has(OPEN_TOP)))) {
                diag = diag + b;
            }
        }
    }
    if (diag <= 0.0) { cells[5u * P_.nc + c] = 0.0; return; }
    let old = cells[5u * P_.nc + c];
    let fresh = (sum + cells[4u * P_.nc + c]) / diag;
    cells[5u * P_.nc + c] = old + P_.omega * (fresh - old);
}

@compute @workgroup_size(64) fn gradient(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let f = gid(id, n);
    if (f >= P_.nu + P_.nv + P_.nw) { return; }
    let q = face(f);
    let a = q.w;
    let p = vec3<u32>(q.x, q.y, q.z);
    let n3 = vec3<u32>(P_.nx, P_.ny, P_.nz);
    if (p[a] == 0u) { return; }
    let b = P_.rho0 / rho_face(a, p.x, p.y, p.z);
    var lo = p;
    lo[a] = p[a] - 1u;
    let cl = ci(lo.x, lo.y, lo.z);
    if (p[a] == n3[a]) {
        let open = (a == 0u && has(OPEN_SIDE)) || (a == 2u && has(OPEN_TOP));
        if (open && solid[cl] == 0u) {
            vel[f] = vel[f] - b * (0.0 - cells[5u * P_.nc + cl]);
        }
        return;
    }
    let ch = ci(p.x, p.y, p.z);
    if (solid[cl] != 0u || solid[ch] != 0u) { return; }
    vel[f] = vel[f] - b * (cells[5u * P_.nc + ch] - cells[5u * P_.nc + cl]);
}

@compute @workgroup_size(64) fn measure(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
    let e = gid(id, n);
    if (e < P_.nu + P_.nv + P_.nw) {
        atomicMax(&stats[1], bitcast<u32>(abs(vel[e])));
    }
    if (e < P_.nc && solid[e] == 0u) {
        let i = e % P_.nx;
        let j = (e / P_.nx) % P_.ny;
        let k = e / (P_.nx * P_.ny);
        atomicMax(&stats[0], bitcast<u32>(abs(div_now(i, j, k) - cells[3u * P_.nc + e])));
    }
}
