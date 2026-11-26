-- Play in a pool: steer three balls through real water.
--   Keys 1, 2, 3 (or N for the next one) choose which ball you steer; it glows at its edge.
--   ARROW KEYS (or W A S D) move it across the pool, E and Q move it up and down.
--   "Up" on the keyboard is away from the camera. Click the 3D view first so it has the keys.
--   Push a ball through the water and watch the bow wave, the wake, and the waves bouncing off
--   the walls; lift one out and drop it in for a splash; ram the others with it.
--   A boat floats on the waves you make: buoyancy (Archimedes) lifts it, the water drags it along.
-- Each ball also has sliders in Controls (its velocity), so they can be driven from there or from
-- a script too.
function setup()
  param{name = "gravity", value = 9.81, min = 0, max = 20, about = "m/s² (Earth 9.81, Moon 1.6)", owner = "nature"}

  -- a pool 2 m x 1.2 m (0.8 m deep box), water 0.35 m deep; 4 cm cells
  fluid{kind = "water", size = {50, 30, 20}, cell = 0.04, dt = 1 / 50, gravity = "gravity", iterations = 50}
  water{box = {0, 0, 0, 2.0, 1.2, 0.35}}

  -- three balls you can steer: a (yellow), b (red), c (blue)
  local balls = {{"a", 0.45, 0.35, 0.14}, {"b", 0.45, 0.85, 0.12}, {"c", 1.0, 0.6, 0.10}}
  for _, b in ipairs(balls) do
    local name = b[1]
    for _, axis in ipairs({"x", "y", "z"}) do
      param{name = name .. "_v" .. axis, value = 0, min = -2, max = 2, about = "ball " .. name .. ", velocity along " .. axis .. " (keys)", owner = "you"}
    end
    object{name = name, shape = "ball", at = {b[2], b[3], 0.35}, radius = b[4],
           velocity = {name .. "_vx", name .. "_vy", name .. "_vz"}, keys = true, speed = 1.2}
  end
  -- and a boat that floats (green)
  object{name = "boat", shape = "box", at = {1.5, 0.85, 0.5}, size = {0.3, 0.14, 0.08}, float = true, density = 400}

  fluid_metric{name = "fastest water (m/s)", field = "speed", op = "max"}

  camera{yaw = -60, pitch = 32, distance = 0.85, fov = 45}
  light{yaw = 300, pitch = 55, ambient = 0.4, strength = 0.8, ground = true, grid = false}
end