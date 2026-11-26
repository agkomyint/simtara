// SimTara liquid solver on the GPU: the same steps as the CPU path in liquid.rs / lib.rs / grid.rs
// (APIC transfers, volume correction, ghost-fluid free surface, red-black SOR pressure), one
// thread per particle, face or cell. Sums that many threads add to use fixed-point integers, so the
// result does not depend on the order threads run in: runs repeat exactly on the same GPU.

struct Params {
    nx: u32, ny: u32, nz: u32, np: u32,
    dt: f32, h: f32, gravity: f32, visc: f32,
    nu: u32, nv: u32, nw: u32, nc: u32,
    omega: f32, colour: u32, pad0: u32, pad1: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> ppos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pvel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> paff: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> vel: array<f32>;
@group(0) @binding(5) var<storage, read_write> acc: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> known: array<u32>;
@group(0) @binding(7) var<storage, read_write> known2: array<u32>;
@group(0) @binding(8) var<storage, read_write> flags: array<u32>;
@group(0) @binding(9) var<storage, read_write> solid: array<u32>;
@group(0) @binding(10) var<storage, read_write> rho: array<atomic<i32>>;
@group(0) @binding(11) var<storage, read_write> pres: array<f32>;
@group(0) @binding(12) var<storage, read_write> cellf: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read_write> bin_count: array<atomic<u32>>;
@group(0) @binding(14) var<storage, read_write> bins: array<u32>;
@group(0) @binding(15) var<storage, read_write> ppos2: array<vec4<f32>>;
@group(0) @binding(16) var<storage, read_write> vel2: array<f32>;
// what goes back to the CPU each tick, tightly packed: per particle position, velocity, matrix
@group(0) @binding(17) var<storage, read_write> packed: array<f32>;

const FLUID: u32 = 0u;
const SOLID: u32 = 1u;
const AIR: u32 = 2u;
const S: f32 = 65536.0;
const K: u32 = 48u;
const ROW: u32 = 4194240u; // 65535 * 64

fn gid(id: vec3<u32>) -> u32 {
    return id.x + id.y * ROW;
}

fn nf() -> u32 {
    return P.nu + P.nv + P.nw;
}

fn dims_of(axis: u32) -> vec3<u32> {
    if (axis == 0u) { return vec3<u32>(P.nx + 1u, P.ny, P.nz); }
    if (axis == 1u) { return vec3<u32>(P.nx, P.ny + 1u, P.nz); }
    return vec3<u32>(P.nx, P.ny, P.nz + 1u);
}

fn base_of(axis: u32) -> u32 {
    if (axis == 0u) { return 0u; }
    if (axis == 1u) { return P.nu; }
    return P.nu + P.nv;
}

fn off_of(axis: u32) -> vec3<f32> {
    if (axis == 0u) { return vec3<f32>(0.0, 0.5, 0.5); }
    if (axis == 1u) { return vec3<f32>(0.5, 0.0, 0.5); }
    return vec3<f32>(0.5, 0.5, 0.0);
}

fn fidx(axis: u32, i: u32, j: u32, k: u32) -> u32 {
    let d = dims_of(axis);
    return base_of(axis) + i + d.x * (j + d.y * k);
}

fn cidx(i: u32, j: u32, k: u32) -> u32 {
    return i + P.nx * (j + P.ny * k);
}

// face index -> (axis, i, j, k)
fn face(f: u32) -> vec4<u32> {
    var axis = 0u;
    var r = f;
    if (f >= P.nu + P.nv) { axis = 2u; r = f - P.nu - P.nv; } else if (f >= P.nu) { axis = 1u; r = f - P.nu; }
    let d = dims_of(axis);
    return vec4<u32>(axis, r % d.x, (r / d.x) % d.y, r / (d.x * d.y));
}

fn cell_of(c: u32) -> vec3<u32> {
    return vec3<u32>(c % P.nx, (c / P.nx) % P.ny, c / (P.nx * P.ny));
}

fn is_solid_cell(i: u32, j: u32, k: u32) -> bool {
    return solid[cidx(i, j, k)] == SOLID;
}

fn fl(i: u32, j: u32, k: u32) -> u32 {
    return flags[cidx(i, j, k)];
}

fn is_solid_at(q: vec3<f32>) -> bool {
    let m = max(q, vec3<f32>(0.0));
    let i = u32(m.x);
    let j = u32(m.y);
    let k = u32(m.z);
    if (i >= P.nx || j >= P.ny || k >= P.nz) { return true; }
    return is_solid_cell(i, j, k);
}

// trilinear sample of one velocity component at a grid-space position (as grid::sample)
fn corner1(x: f32, n: u32) -> vec2<f32> {
    let xc = clamp(x, 0.0, f32(n - 1u));
    let i = min(floor(xc), f32(max(n, 2u) - 2u));
    return vec2<f32>(i, xc - i);
}

fn sample_axis(axis: u32, p: vec3<f32>) -> f32 {
    let d = dims_of(axis);
    let q = p - off_of(axis);
    let cx = corner1(q.x, d.x);
    let cy = corner1(q.y, d.y);
    let cz = corner1(q.z, d.z);
    let i = u32(cx.x);
    let j = u32(cy.x);
    let k = u32(cz.x);
    let i1 = min(i + 1u, d.x - 1u);
    let j1 = min(j + 1u, d.y - 1u);
    let k1 = min(k + 1u, d.z - 1u);
    let tx = cx.y;
    let ty = cy.y;
    let tz = cz.y;
    let x00 = vel[fidx(axis, i, j, k)] * (1.0 - tx) + vel[fidx(axis, i1, j, k)] * tx;
    let x10 = vel[fidx(axis, i, j1, k)] * (1.0 - tx) + vel[fidx(axis, i1, j1, k)] * tx;
    let x01 = vel[fidx(axis, i, j, k1)] * (1.0 - tx) + vel[fidx(axis, i1, j, k1)] * tx;
    let x11 = vel[fidx(axis, i, j1, k1)] * (1.0 - tx) + vel[fidx(axis, i1, j1, k1)] * tx;
    let y0 = x00 * (1.0 - ty) + x10 * ty;
    let y1 = x01 * (1.0 - ty) + x11 * ty;
    return y0 * (1.0 - tz) + y1 * tz;
}

fn vel_at(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(sample_axis(0u, p), sample_axis(1u, p), sample_axis(2u, p));
}

fn w1(t: f32, o: u32) -> f32 {
    return select(1.0 - t, t, o == 1u);
}

// ---------------------------------------------------------------- clear
@compute @workgroup_size(64) fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = gid(id);
    if (i < nf()) {
        atomicStore(&acc[2u * i], 0);
        atomicStore(&acc[2u * i + 1u], 0);
    }
    if (i == 0u) {
        atomicStore(&acc[2u * nf()], 0);
    }
    if (i < P.nc) {
        atomicStore(&rho[i], 0);
        flags[i] = select(AIR, SOLID, solid[i] == SOLID);
    }
}

// ---------------------------------------------------------------- particles -> grid (APIC)
@compute @workgroup_size(64) fn p2g(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = gid(id);
    if (p >= P.np) { return; }
    let x = ppos[p].xyz;
    let v = pvel[p].xyz;
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        let d = dims_of(axis);
        let q = x - off_of(axis);
        let b = floor(q);
        let t = q - b;
        let c = paff[3u * p + axis].xyz;
        for (var corner = 0u; corner < 8u; corner = corner + 1u) {
            let o = vec3<u32>(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u);
            let ii = vec3<i32>(b) + vec3<i32>(o);
            if (any(ii < vec3<i32>(0)) || any(ii >= vec3<i32>(d))) { continue; }
            let w = w1(t.x, o.x) * w1(t.y, o.y) * w1(t.z, o.z);
            let dx = vec3<f32>(ii) - q;
            let val = v[axis] + dot(c, dx);
            let f = fidx(axis, u32(ii.x), u32(ii.y), u32(ii.z));
            atomicAdd(&acc[2u * f], i32(round(w * val * S)));
            atomicAdd(&acc[2u * f + 1u], i32(round(w * S)));
        }
    }
    // density at cell centres (nodes outside reflect back in, as liquid::density)
    let q = x - vec3<f32>(0.5);
    let b = floor(q);
    let t = q - b;
    for (var corner = 0u; corner < 8u; corner = corner + 1u) {
        let o = vec3<u32>(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u);
        let ii = clamp(vec3<i32>(b) + vec3<i32>(o), vec3<i32>(0), vec3<i32>(i32(P.nx) - 1, i32(P.ny) - 1, i32(P.nz) - 1));
        let w = w1(t.x, o.x) * w1(t.y, o.y) * w1(t.z, o.z);
        atomicAdd(&rho[cidx(u32(ii.x), u32(ii.y), u32(ii.z))], i32(round(w * S)));
    }
    let ci = min(vec3<u32>(x), vec3<u32>(P.nx - 1u, P.ny - 1u, P.nz - 1u));
    let cc = cidx(ci.x, ci.y, ci.z);
    if (solid[cc] != SOLID) {
        flags[cc] = FLUID;
    }
}

@compute @workgroup_size(64) fn normalize(@builtin(global_invocation_id) id: vec3<u32>) {
    let f = gid(id);
    if (f >= nf()) { return; }
    let s = atomicLoad(&acc[2u * f]);
    let w = atomicLoad(&acc[2u * f + 1u]);
    if (w > 0) {
        vel[f] = f32(s) / f32(w);
        known[f] = 1u;
    } else {
        vel[f] = 0.0;
        known[f] = 0u;
    }
}

// ---------------------------------------------------------------- extrapolation (one layer)
@compute @workgroup_size(64) fn extrapolate(@builtin(global_invocation_id) id: vec3<u32>) {
    let f = gid(id);
    if (f >= nf()) { return; }
    if (known[f] != 0u) {
        vel2[f] = vel[f];
        known2[f] = 1u;
        return;
    }
    let a = face(f);
    let d = dims_of(a.x);
    var s = 0.0;
    var n = 0u;
    if (a.y > 0u) { let o = fidx(a.x, a.y - 1u, a.z, a.w); if (known[o] != 0u) { s += vel[o]; n += 1u; } }
    if (a.y + 1u < d.x) { let o = fidx(a.x, a.y + 1u, a.z, a.w); if (known[o] != 0u) { s += vel[o]; n += 1u; } }
    if (a.z > 0u) { let o = fidx(a.x, a.y, a.z - 1u, a.w); if (known[o] != 0u) { s += vel[o]; n += 1u; } }
    if (a.z + 1u < d.y) { let o = fidx(a.x, a.y, a.z + 1u, a.w); if (known[o] != 0u) { s += vel[o]; n += 1u; } }
    if (a.w > 0u) { let o = fidx(a.x, a.y, a.z, a.w - 1u); if (known[o] != 0u) { s += vel[o]; n += 1u; } }
    if (a.w + 1u < d.z) { let o = fidx(a.x, a.y, a.z, a.w + 1u); if (known[o] != 0u) { s += vel[o]; n += 1u; } }
    if (n > 0u) {
        vel2[f] = s / f32(n);
        known2[f] = 1u;
    } else {
        vel2[f] = vel[f];
        known2[f] = 0u;
    }
}

@compute @workgroup_size(64) fn copy_back(@builtin(global_invocation_id) id: vec3<u32>) {
    let f = gid(id);
    if (f >= nf()) { return; }
    vel[f] = vel2[f];
    known[f] = known2[f];
}

// ---------------------------------------------------------------- walls (closed box) and gravity
fn wall_face(a: vec4<u32>) -> bool {
    if (a.x == 0u) {
        if (a.y == 0u || a.y == P.nx) { return true; }
        return is_solid_cell(a.y - 1u, a.z, a.w) || is_solid_cell(a.y, a.z, a.w);
    }
    if (a.x == 1u) {
        if (a.z == 0u || a.z == P.ny) { return true; }
        return is_solid_cell(a.y, a.z - 1u, a.w) || is_solid_cell(a.y, a.z, a.w);
    }
    if (a.w == 0u || a.w == P.nz) { return true; }
    return is_solid_cell(a.y, a.z, a.w - 1u) || is_solid_cell(a.y, a.z, a.w);
}

@compute @workgroup_size(64) fn enforce(@builtin(global_invocation_id) id: vec3<u32>) {
    let f = gid(id);
    if (f >= nf()) { return; }
    if (wall_face(face(f))) { vel[f] = 0.0; }
}

@compute @workgroup_size(64) fn gravity(@builtin(global_invocation_id) id: vec3<u32>) {
    let f = gid(id);
    if (f >= nf()) { return; }
    let a = face(f);
    if (a.x == 2u) { vel[f] = vel[f] - P.gravity * P.dt; }
    if (wall_face(a)) { vel[f] = 0.0; }
}

fn touches_liquid(a: vec4<u32>) -> bool {
    if (a.x == 0u) { return (a.y > 0u && fl(a.y - 1u, a.z, a.w) == FLUID) || (a.y < P.nx && fl(a.y, a.z, a.w) == FLUID); }
    if (a.x == 1u) { return (a.z > 0u && fl(a.y, a.z - 1u, a.w) == FLUID) || (a.z < P.ny && fl(a.y, a.z, a.w) == FLUID); }
    return (a.w > 0u && fl(a.y, a.z, a.w - 1u) == FLUID) || (a.w < P.nz && fl(a.y, a.z, a.w) == FLUID);
}

// ---------------------------------------------------------------- viscosity (one explicit step into vel2)
@compute @workgroup_size(64) fn viscous(@builtin(global_invocation_id) id: vec3<u32>) {
    let f = gid(id);
    if (f >= nf()) { return; }
    let a = face(f);
    let me = vel[f];
    if (!touches_liquid(a)) { vel2[f] = me; return; }
    let d = dims_of(a.x);
    // past the edge: along this velocity it slides, across it the wall holds still (no-slip)
    var lap = -6.0 * me;
    if (a.y > 0u) { lap += vel[fidx(a.x, a.y - 1u, a.z, a.w)]; } else { lap += select(0.0, me, a.x == 0u); }
    if (a.y + 1u < d.x) { lap += vel[fidx(a.x, a.y + 1u, a.z, a.w)]; } else { lap += select(0.0, me, a.x == 0u); }
    if (a.z > 0u) { lap += vel[fidx(a.x, a.y, a.z - 1u, a.w)]; } else { lap += select(0.0, me, a.x == 1u); }
    if (a.z + 1u < d.y) { lap += vel[fidx(a.x, a.y, a.z + 1u, a.w)]; } else { lap += select(0.0, me, a.x == 1u); }
    if (a.w > 0u) { lap += vel[fidx(a.x, a.y, a.z, a.w - 1u)]; } else { lap += select(0.0, me, a.x == 2u); }
    if (a.w + 1u < d.z) { lap += vel[fidx(a.x, a.y, a.z, a.w + 1u)]; } else { lap += select(0.0, me, a.x == 2u); }
    vel2[f] = me + P.visc * lap;
}

@compute @workgroup_size(64) fn viscous_back(@builtin(global_invocation_id) id: vec3<u32>) {
    let f = gid(id);
    if (f >= nf()) { return; }
    vel[f] = select(vel2[f], 0.0, wall_face(face(f)));
}

// ---------------------------------------------------------------- pressure: build, solve, apply
fn fill(c: u32) -> f32 {
    return f32(atomicLoad(&rho[c])) / S / 8.0;
}

fn theta(c: u32, n: u32) -> f32 {
    let a = min(fill(c), 1.0);
    let b = min(fill(n), 1.0);
    if (a > 0.5 && a - b > 1e-4) { return clamp((a - 0.5) / (a - b), 0.2, 1.0); }
    return 1.0;
}

fn inside(ok: bool, n: u32) -> bool {
    return !ok || flags[n] == FLUID || solid[n] == SOLID;
}

@compute @workgroup_size(64) fn cells(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = gid(id);
    if (c >= P.nc) { return; }
    if (flags[c] != FLUID) { cellf[c] = vec4<f32>(0.0); pres[c] = 0.0; return; }
    let g = cell_of(c);
    let i = g.x;
    let j = g.y;
    let k = g.z;
    var dv = vel[fidx(0u, i + 1u, j, k)] - vel[fidx(0u, i, j, k)] + vel[fidx(1u, i, j + 1u, k)] - vel[fidx(1u, i, j, k)] + vel[fidx(2u, i, j, k + 1u)] - vel[fidx(2u, i, j, k)];
    // back towards the rest density (liquid::volume_target)
    let rate = P.h / P.dt;
    let interior = inside(i > 0u, c - 1u) && inside(i + 1u < P.nx, c + 1u) && inside(j > 0u, c - P.nx) && inside(j + 1u < P.ny, c + P.nx) && inside(k > 0u, c - P.nx * P.ny) && inside(k + 1u < P.nz, c + P.nx * P.ny);
    let excess = (fill(c) * 8.0 - 8.0) / 8.0;
    var tgt = 0.0;
    if (excess > 0.0) { tgt = 0.5 * min(excess, 1.0) * rate; } else if (interior) { tgt = 0.25 * max(excess, -0.5) * rate; }
    dv = dv - tgt;
    var diag = 0.0;
    var mask = 0u;
    var nbs = array<u32, 6>(c - 1u, c + 1u, c - P.nx, c + P.nx, c - P.nx * P.ny, c + P.nx * P.ny);
    var ok = array<bool, 6>(i > 0u, i + 1u < P.nx, j > 0u, j + 1u < P.ny, k > 0u, k + 1u < P.nz);
    for (var s = 0u; s < 6u; s = s + 1u) {
        if (!ok[s]) { continue; }
        let n = nbs[s];
        let f = flags[n];
        if (f == SOLID) { continue; }
        if (f == AIR) { diag += 1.0 / theta(c, n); } else { diag += 1.0; mask = mask | (1u << s); }
    }
    if (diag == 0.0) { pres[c] = 0.0; }
    cellf[c] = vec4<f32>(dv, diag, bitcast<f32>(mask), 0.0);
}

@compute @workgroup_size(64) fn sor(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = gid(id);
    if (c >= P.nc) { return; }
    let g = cell_of(c);
    if ((g.x + g.y + g.z) % 2u != P.colour) { return; }
    if (flags[c] != FLUID) { return; }
    let cf = cellf[c];
    if (cf.y <= 0.0) { return; }
    let m = bitcast<u32>(cf.z);
    var sum = 0.0;
    if ((m & 1u) != 0u) { sum += pres[c - 1u]; }
    if ((m & 2u) != 0u) { sum += pres[c + 1u]; }
    if ((m & 4u) != 0u) { sum += pres[c - P.nx]; }
    if ((m & 8u) != 0u) { sum += pres[c + P.nx]; }
    if ((m & 16u) != 0u) { sum += pres[c - P.nx * P.ny]; }
    if ((m & 32u) != 0u) { sum += pres[c + P.nx * P.ny]; }
    let gs = (sum - cf.x) / cf.y;
    pres[c] = (1.0 - P.omega) * pres[c] + P.omega * gs;
}

// pressure jump across a face; next to the surface the air side holds a ghost value
fn jump(a: u32, b: u32) -> vec2<f32> {
    let fa = flags[a];
    let fb = flags[b];
    if (fa == SOLID || fb == SOLID) { return vec2<f32>(0.0, 0.0); }
    if (fa == FLUID && fb == AIR) { return vec2<f32>(-pres[a] / theta(a, b), 1.0); }
    if (fa == AIR && fb == FLUID) { return vec2<f32>(pres[b] / theta(b, a), 1.0); }
    return vec2<f32>(pres[b] - pres[a], 1.0);
}

@compute @workgroup_size(64) fn gradient(@builtin(global_invocation_id) id: vec3<u32>) {
    let f = gid(id);
    if (f >= nf()) { return; }
    let a = face(f);
    var lo = 0u;
    var hi = 0u;
    if (a.x == 0u) {
        if (a.y == 0u || a.y == P.nx) { return; }
        lo = cidx(a.y - 1u, a.z, a.w); hi = cidx(a.y, a.z, a.w);
    } else if (a.x == 1u) {
        if (a.z == 0u || a.z == P.ny) { return; }
        lo = cidx(a.y, a.z - 1u, a.w); hi = cidx(a.y, a.z, a.w);
    } else {
        if (a.w == 0u || a.w == P.nz) { return; }
        lo = cidx(a.y, a.z, a.w - 1u); hi = cidx(a.y, a.z, a.w);
    }
    if (flags[lo] != FLUID && flags[hi] != FLUID) { return; }
    let j = jump(lo, hi);
    if (j.y > 0.0) { vel[f] = vel[f] - j.x; }
}

@compute @workgroup_size(64) fn mark_touch(@builtin(global_invocation_id) id: vec3<u32>) {
    let f = gid(id);
    if (f >= nf()) { return; }
    known[f] = select(0u, 1u, touches_liquid(face(f)));
}

// ---------------------------------------------------------------- grid -> particles (APIC), move
@compute @workgroup_size(64) fn g2p(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = gid(id);
    if (p >= P.np) { return; }
    let x = ppos[p].xyz;
    var v = vec3<f32>(0.0);
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        let d = dims_of(axis);
        let q = x - off_of(axis);
        let b = floor(q);
        let t = q - b;
        var val = 0.0;
        var g = vec3<f32>(0.0);
        for (var corner = 0u; corner < 8u; corner = corner + 1u) {
            let o = vec3<u32>(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u);
            let ii = clamp(vec3<i32>(b) + vec3<i32>(o), vec3<i32>(0), vec3<i32>(d) - vec3<i32>(1));
            let fv = vel[fidx(axis, u32(ii.x), u32(ii.y), u32(ii.z))];
            let wx = w1(t.x, o.x);
            let wy = w1(t.y, o.y);
            let wz = w1(t.z, o.z);
            let sx = select(-1.0, 1.0, o.x == 1u);
            let sy = select(-1.0, 1.0, o.y == 1u);
            let sz = select(-1.0, 1.0, o.z == 1u);
            val += wx * wy * wz * fv;
            g += vec3<f32>(sx * wy * wz, wx * sy * wz, wx * wy * sz) * fv;
        }
        v[axis] = val;
        paff[3u * p + axis] = vec4<f32>(g, 0.0);
    }
    pvel[p] = vec4<f32>(v, 0.0);
}

@compute @workgroup_size(64) fn advect(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = gid(id);
    if (p >= P.np) { return; }
    let k = P.dt / P.h;
    let x = ppos[p].xyz;
    var vp = pvel[p].xyz;
    let v0 = vel_at(x);
    let vm = vel_at(x + 0.5 * k * v0);
    var pos = x;
    let lim = vec3<f32>(f32(P.nx), f32(P.ny), f32(P.nz));
    for (var a = 0u; a < 3u; a = a + 1u) {
        var q = pos;
        q[a] = clamp(pos[a] + k * vm[a], 0.01, lim[a] - 0.01);
        if (is_solid_at(q)) {
            vp[a] = 0.0;
        } else {
            if (q[a] <= 0.01 || q[a] >= lim[a] - 0.01) { vp[a] = 0.0; }
            pos = q;
        }
    }
    ppos[p] = vec4<f32>(pos, 0.0);
    pvel[p] = vec4<f32>(vp, 0.0);
}

// ---------------------------------------------------------------- keep particles apart
@compute @workgroup_size(64) fn clear_bins(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = gid(id);
    if (c >= P.nc) { return; }
    atomicStore(&bin_count[c], 0u);
}

fn pcell(x: vec3<f32>) -> u32 {
    let ci = min(vec3<u32>(max(x, vec3<f32>(0.0))), vec3<u32>(P.nx - 1u, P.ny - 1u, P.nz - 1u));
    return cidx(ci.x, ci.y, ci.z);
}

@compute @workgroup_size(64) fn bin(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = gid(id);
    if (p >= P.np) { return; }
    let c = pcell(ppos[p].xyz);
    let slot = atomicAdd(&bin_count[c], 1u);
    if (slot < K) { bins[c * K + slot] = p; }
}

@compute @workgroup_size(64) fn separate(@builtin(global_invocation_id) id: vec3<u32>) {
    let me = gid(id);
    if (me >= P.np) { return; }
    let x = ppos[me].xyz;
    let ci = min(vec3<u32>(max(x, vec3<f32>(0.0))), vec3<u32>(P.nx - 1u, P.ny - 1u, P.nz - 1u));
    let mn = 0.45;
    var push = vec3<i32>(0);
    for (var k = max(ci.z, 1u) - 1u; k < min(ci.z + 2u, P.nz); k = k + 1u) {
        for (var j = max(ci.y, 1u) - 1u; j < min(ci.y + 2u, P.ny); j = j + 1u) {
            for (var i = max(ci.x, 1u) - 1u; i < min(ci.x + 2u, P.nx); i = i + 1u) {
                let c = cidx(i, j, k);
                let n = min(atomicLoad(&bin_count[c]), K);
                for (var s = 0u; s < n; s = s + 1u) {
                    let o = bins[c * K + s];
                    if (o == me) { continue; }
                    let dv = x - ppos[o].xyz;
                    let dd = dot(dv, dv);
                    if (dd >= mn * mn) { continue; }
                    let dist = sqrt(dd);
                    var dir = vec3<f32>(0.0, 0.0, select(-1.0, 1.0, me < o));
                    if (dist > 1e-6) { dir = dv / dist; }
                    // fixed point: the total does not depend on the order neighbours are found in
                    push += vec3<i32>(round(0.5 * (mn - dist) * dir * S));
                }
            }
        }
    }
    let pf = clamp(vec3<f32>(push) / S, vec3<f32>(-0.25), vec3<f32>(0.25));
    var pos = x;
    let lim = vec3<f32>(f32(P.nx), f32(P.ny), f32(P.nz)) - vec3<f32>(0.01);
    for (var a = 0u; a < 3u; a = a + 1u) {
        var q = pos;
        q[a] = clamp(q[a] + pf[a], 0.01, lim[a]);
        if (!is_solid_at(q)) { pos = q; }
    }
    ppos2[me] = vec4<f32>(pos, 0.0);
}

@compute @workgroup_size(64) fn copy_pos(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = gid(id);
    if (p >= P.np) { return; }
    ppos[p] = ppos2[p];
}

// ---------------------------------------------------------------- what the CPU needs back
// largest divergence left in any liquid cell (non-negative floats order like their bits)
@compute @workgroup_size(64) fn measure(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = gid(id);
    if (c >= P.nc || flags[c] != FLUID) { return; }
    let g = cell_of(c);
    let dv = vel[fidx(0u, g.x + 1u, g.y, g.z)] - vel[fidx(0u, g.x, g.y, g.z)] + vel[fidx(1u, g.x, g.y + 1u, g.z)] - vel[fidx(1u, g.x, g.y, g.z)] + vel[fidx(2u, g.x, g.y, g.z + 1u)] - vel[fidx(2u, g.x, g.y, g.z)];
    atomicMax(&acc[2u * nf()], bitcast<i32>(abs(dv)));
}

@compute @workgroup_size(64) fn pack(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = gid(id);
    if (p >= P.np) { return; }
    let o = 15u * p;
    let x = ppos[p];
    let v = pvel[p];
    packed[o] = x.x; packed[o + 1u] = x.y; packed[o + 2u] = x.z;
    packed[o + 3u] = v.x; packed[o + 4u] = v.y; packed[o + 5u] = v.z;
    for (var a = 0u; a < 3u; a = a + 1u) {
        let c = paff[3u * p + a];
        packed[o + 6u + 3u * a] = c.x; packed[o + 7u + 3u * a] = c.y; packed[o + 8u + 3u * a] = c.z;
    }
}