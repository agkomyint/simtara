//! Pictures of a universe without a window: run a script, draw chosen ticks on the GPU, save PNGs.
//!
//! `simtara-studio --picture FILE.luau --ticks 30,60,120 --out frame --size 960x600`
//! writes `frame-30.png`, `frame-60.png`, ... (add `--obj` for the liquid's surface mesh too,`n//! `frame-30.obj`, ... for Blender and other tools). It uses the same camera, light and drawing as the
//! 3D view, so it shows exactly what Studio shows.

use crate::view3d::{self, Item, Mesh, Volume};
use crate::view3d_gpu::Gpu3d;
use eframe::egui;
use eframe::egui_wgpu::wgpu;
use simtara_things::Things;
use std::collections::HashMap;

fn device() -> Result<(wgpu::Device, wgpu::Queue), String> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).map_err(|e| format!("no graphics card: {e}"))?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).map_err(|e| e.to_string())
}

/// What the 3D view would show: things that have x, y (and z), water drops, the smoke volume.
fn contents(sim: &simtara_core::Simulation) -> (Vec<Item>, Option<Volume>) {
    let mut items = Vec::new();
    if let Some(t) = sim.world().get::<Things>() {
        let num = |th: &simtara_things::Thing, k: &str| th.props.get(k).and_then(|v| v.as_f64());
        for th in t.things.values() {
            if let (Some(x), Some(y)) = (num(th, "x"), num(th, "y")) {
                let shape = th.props.get("shape").and_then(|v| v.as_str()).unwrap_or("").to_string();
                items.push(Item { id: th.id, pos: [x as f32, y as f32, num(th, "z").unwrap_or(0.0) as f32], color: egui::Color32::from_rgb(200, 200, 210), size: 1.0, heading: 0.0, shape });
            }
        }
    }
    let mut volume = None;
    let light = sim.world().get::<Things>().map(|t| t.light.clone()).unwrap_or_default();
    let sun = {
        let (y, p) = ((light.yaw as f32).to_radians(), (light.pitch as f32).to_radians());
        [p.cos() * y.cos(), p.cos() * y.sin(), p.sin()]
    };
    if let Some(f) = sim.world().get::<simtara_fields::Fluid>().filter(|f| f.ready()) {
        let mut copy = simtara_fields::Fluid::default();
        copy.clone_from(f);
        copy.want_render = true;
        copy.sun = sun;
        copy.make_render();
        let r = copy.render.unwrap_or_default();
        for (i, p) in r.spray.iter().enumerate() {
            items.push(Item { id: u64::MAX - i as u64, pos: *p, color: crate::build::water_colour(1.0), size: 1.0, heading: 0.0, shape: "spray".into() });
        }
        let d = f.dims();
        let water = f.kind == simtara_fields::Kind::Water;
        volume = Some(Volume { dims: [d.nx as u32, d.ny as u32, d.nz as u32], size: f.extent(), data: r.rgba.clone(), water_radius: if water { f.cell * 0.55 } else { 0.0 }, caustics: r.caustics.clone(), liquid: (water && !r.liquid.is_empty()).then(|| r.liquid.clone()), generation: r.generation, blend: 1.0, objects: r.objects.clone() });
    }    (items, volume)
}

pub fn run(script_path: &str, ticks: &[u64], out: &str, size: [u32; 2], obj: bool) -> Result<Vec<String>, String> {
    let script = std::fs::read_to_string(script_path).map_err(|e| format!("{script_path}: {e}"))?;
    let mut sim = simtara_store::build_simulation(&simtara_store::new_build_config(1, &script));
    let (dev, queue) = device()?;
    let mut gpu = Gpu3d::new(dev, queue);
    let mut models: HashMap<String, Mesh> = HashMap::new();
    for n in ["cube", "pyramid"] {
        models.insert(n.into(), view3d::builtin(n).unwrap());
    }
    let mut written = Vec::new();
    let mut sorted = ticks.to_vec();
    sorted.sort();
    for t in sorted {
        let now = sim.world().clock.tick;
        if t > now {
            sim.advance(t - now);
        }
        if let (Some(err), _) = simtara_script::status(sim.world()) {
            return Err(err);
        }
        let (items, volume) = contents(&sim);
        let (cam, light) = sim.world().get::<Things>().map(|t| (t.camera.clone(), t.light.clone())).unwrap_or_default();
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(size[0] as f32, size[1] as f32));
        let Some(sc) = view3d::scene(rect, &items, &cam, &light, volume.as_ref()) else { return Err("nothing to draw".into()) };
        gpu.render(size, &sc.uniforms, &items, &models, volume.as_ref());
        let rgba = gpu.read_back(size);
        if std::env::var_os("SIMTARA_TIMING").is_some() {
            // the same picture again: the fluid's data is already on the card, as in a window frame
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                gpu.render(size, &sc.uniforms, &items, &models, volume.as_ref());
                let _ = gpu.read_back([1, 1]);
            }
            eprintln!("frame at {}x{}: {:.1} ms", size[0], size[1], t0.elapsed().as_secs_f64() * 100.0);
        }
        let path = format!("{out}-{t}.png");
        let file = std::fs::File::create(&path).map_err(|e| format!("{path}: {e}"))?;
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), size[0], size[1]);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().and_then(|mut w| w.write_image_data(&rgba)).map_err(|e| e.to_string())?;
        written.push(path);
        if obj {
            if let Some(f) = sim.world().get::<simtara_fields::Fluid>().filter(|f| f.ready() && !f.particles.is_empty()) {
                let mesh = f.surface_mesh(2);
                let path = format!("{out}-{t}.obj");
                std::fs::write(&path, mesh.to_obj("water")).map_err(|e| format!("{path}: {e}"))?;
                written.push(format!("{path}  ({} triangles, {:.4} m³ inside)", mesh.triangles.len(), mesh.volume()));
            }
        }
    }
    Ok(written)
}
