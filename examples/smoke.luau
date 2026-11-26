-- Smoke and wind around a building, in 3D, in real units (metres, seconds).
--   A wind tunnel 8 m long, 4 m wide and 4 m high. Wind blows in at x = 0 and out at the far end.
--   A chimney on the roof puts out hot smoke; a smoke pot on the ground upwind shows the wind.
--   The air is solved as a real fluid: it flows around the building, swirls behind it (the wake),
--   and hot smoke rises because hot air is lighter (buoyancy).
-- Drag to turn, scroll to zoom. Try: wind to 0 (the chimney smoke rises straight up), more heat,
-- or swirl to 0 (smoother, less lively smoke).
function setup()
  param{name = "wind", value = 1.5, min = 0, max = 4, about = "wind speed, m/s", owner = "nature"}
  param{name = "heat", value = 250, min = 0, max = 500, about = "chimney temperature above the air, °C", owner = "people"}
  param{name = "chimney", value = 4, min = 0, max = 10, about = "smoke from the chimney", owner = "people"}
  param{name = "ground_smoke", value = 2, min = 0, max = 10, about = "smoke pot upwind, to see the wind", owner = "people"}
  param{name = "swirl", value = 0.3, min = 0, max = 1.5, about = "small swirls (vorticity confinement)", owner = "nature"}

  fluid{size = {80, 40, 40}, cell = 0.1, dt = 1 / 30, wind = "wind", vorticity = "swirl", cooling = 0.2, fade = 0.03}
  solid{box = {2.6, 1.4, 0, 3.6, 2.6, 1.8}}                 -- the building
  solid{box = {3.3, 1.9, 1.8, 3.5, 2.1, 2.3}}               -- its chimney
  emitter{at = {3.4, 2.0, 2.45}, radius = 0.12, smoke = "chimney", heat = "heat", velocity = {0, 0, 1}}
  emitter{at = {0.8, 2.0, 0.35}, radius = 0.25, smoke = "ground_smoke"}
  -- flags show the air's own velocity where they stand: upwind, on the roof, in the wake
  flag{at = {1.6, 1.2, 0}, height = 1.4}
  flag{at = {2.9, 1.6, 1.8}, height = 0.6}
  flag{at = {4.6, 2.3, 0}, height = 1.4}

  fluid_metric{name = "smoke behind the building", field = "smoke", op = "sum", box = {3.6, 0, 0, 6, 4, 2}}
  fluid_metric{name = "hottest air (°C above)", field = "temp", op = "max"}
  fluid_metric{name = "wind over the roof (m/s)", field = "speed", op = "max", box = {2.6, 1.4, 1.9, 3.6, 2.6, 2.4}}

  camera{yaw = -55, pitch = 22, distance = 0.85, fov = 45}
  light{yaw = 300, pitch = 50, ambient = 0.45, strength = 0.75, ground = true, grid = true}
end
