// Particle pair forces. Mirrors `accelerations_cpu` in simtara-particles exactly.
// One thread per particle; neighbours come from a CPU-built uniform grid.

struct Params {
    n: u32,
    gw: u32,
    gh: u32,
    kinds: u32,
    wrap: u32,
    width: f32,
    height: f32,
    rmax: f32,
    beta: f32,
    gscale: f32,   // gravity * FORCE_SCALE * rmax
    coulomb: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pos: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> info: array<vec4<f32>>;      // kind, charge, mass, _
@group(0) @binding(3) var<storage, read> cells: array<u32>;
@group(0) @binding(4) var<storage, read> cell_start: array<u32>;
@group(0) @binding(5) var<storage, read> order: array<u32>;
@group(0) @binding(6) var<storage, read> attraction: array<f32>;
@group(0) @binding(7) var<storage, read_write> acc: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.n) { return; }

    let pi = pos[i];
    let ki = min(u32(info[i].x), P.kinds - 1u);
    let qi = info[i].y;
    let ci = cells[i];
    let cx = i32(ci % P.gw);
    let cy = i32(ci / P.gw);
    let w = P.width;
    let h = P.height;
    let r2max = P.rmax * P.rmax;

    var seen: array<u32, 9>;
    var ns = 0u;
    var a = vec2<f32>(0.0, 0.0);

    for (var oy = -1; oy <= 1; oy++) {
        for (var ox = -1; ox <= 1; ox++) {
            var nx = cx + ox;
            var ny = cy + oy;
            if (P.wrap != 0u) {
                nx = (nx + i32(P.gw)) % i32(P.gw);
                ny = (ny + i32(P.gh)) % i32(P.gh);
            } else if (nx < 0 || ny < 0 || nx >= i32(P.gw) || ny >= i32(P.gh)) {
                continue;
            }
            let cid = u32(ny) * P.gw + u32(nx);
            var dup = false;
            for (var s = 0u; s < ns; s++) {
                if (seen[s] == cid) { dup = true; }
            }
            if (dup) { continue; }
            seen[ns] = cid;
            ns++;

            for (var k = cell_start[cid]; k < cell_start[cid + 1u]; k++) {
                let j = order[k];
                if (j == i) { continue; }
                var d = pos[j] - pi;
                if (P.wrap != 0u) {
                    if (d.x > w * 0.5) { d.x -= w; } else if (d.x < -w * 0.5) { d.x += w; }
                    if (d.y > h * 0.5) { d.y -= h; } else if (d.y < -h * 0.5) { d.y += h; }
                }
                let r2 = dot(d, d);
                if (r2 >= r2max || r2 < 1e-12) { continue; }
                let r = sqrt(r2);
                let t = r / P.rmax;
                let kj = min(u32(info[j].x), P.kinds - 1u);
                let at = attraction[ki * P.kinds + kj] - P.coulomb * qi * info[j].y;
                var f: f32;
                if (t < P.beta) {
                    f = t / P.beta - 1.0;
                } else {
                    f = at * (1.0 - abs(1.0 + P.beta - 2.0 * t) / (1.0 - P.beta));
                }
                a += d * (f / r);
            }
        }
    }
    let g = P.gscale / max(info[i].z, 1e-6);
    acc[i] = a * g;
}
