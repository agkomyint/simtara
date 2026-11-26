-- A fire's smoke plume, in 3D, in real units.
--   A room-sized box (2.4 m x 2.4 m, 4 m high, open at the top) with a fire on the floor.
--   The fire heats the air; hot air is lighter, so it rises and pulls fresh air in from the sides.
--   The rising column curls into a mushroom and breaks into swirls: the same physics that
--   real fire models use (buoyancy, incompressible flow, vorticity).
-- Try: more heat (faster, taller plume), a side breeze (it bends), a beam that blocks it.
function setup()
  param{name = "heat", value = 450, min = 0, max = 600, about = "fire temperature above the air, °C", owner = "nature"}
  param{name = "smoke", value = 3, min = 0, max = 10, about = "smoke the fire makes", owner = "nature"}
  param{name = "breeze", value = 0, min = 0, max = 1.5, about = "a side wind, m/s", owner = "nature"}
  param{name = "swirl", value = 0.5, min = 0, max = 1.5, about = "small swirls (vorticity confinement)", owner = "nature"}
  param{name = "cooling", value = 0.25, min = 0, max = 1, about = "how fast hot air cools", owner = "nature"}

  fluid{size = {48, 48, 80}, cell = 0.05, dt = 1 / 30, wind = "breeze", open_top = true, vorticity = "swirl", cooling = "cooling", fade = 0.02}
  solid{box = {0.7, 0.7, 2.6, 1.3, 1.7, 2.75}}               -- a beam the smoke has to get around
  emitter{at = {1.2, 1.2, 0.15}, radius = 0.25, smoke = "smoke", heat = "heat", velocity = {0, 0, 0.5}}

  fluid_metric{name = "smoke above the beam", field = "smoke", op = "sum", box = {0, 0, 2.75, 2.4, 2.4, 4}}
  fluid_metric{name = "hottest air (°C above)", field = "temp", op = "max"}
  fluid_metric{name = "fastest rising air (m/s)", field = "speed", op = "max"}

  camera{yaw = -40, pitch = 12, distance = 0.95, fov = 45}
  light{yaw = 320, pitch = 55, ambient = 0.45, strength = 0.75, ground = true, grid = true}
end
