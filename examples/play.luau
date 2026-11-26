-- Play in a pool: steer a ball through real water.
--   ARROW KEYS (or W A S D) move the yellow ball across the pool, E and Q move it up and down.
--   Push it through the water and watch the bow wave, the wake, and the waves bouncing off the
--   walls. Lift it out and drop it in for a splash. "Up" on the keyboard is away from the camera.
--   A floating ball and a boat ride the waves you make: buoyancy (Archimedes) lifts them, the
--   water drags them along.
-- The water is real: it pushes back on what floats, and a ball denser than water would sink.
function setup()
  param{name = "ball_vx", value = 0, min = -2, max = 2, about = "steered ball, velocity across (keys)", owner = "you"}
  param{name = "ball_vy", value = 0, min = -2, max = 2, about = "steered ball, velocity along (keys)", owner = "you"}
  param{name = "ball_vz", value = 0, min = -2, max = 2, about = "steered ball, velocity up (E / Q)", owner = "you"}
  param{name = "gravity", value = 9.81, min = 0, max = 20, about = "m/s² (Earth 9.81, Moon 1.6)", owner = "nature"}

  -- a pool 2 m x 1.2 m, 0.8 m deep box, water 0.35 m deep; 4 cm cells
  fluid{kind = "water", size = {50, 30, 20}, cell = 0.04, dt = 1 / 50, gravity = "gravity", iterations = 50}
  water{box = {0, 0, 0, 2.0, 1.2, 0.35}}

  object{name = "ball", shape = "ball", at = {0.5, 0.6, 0.35}, radius = 0.14,
         velocity = {"ball_vx", "ball_vy", "ball_vz"}, keys = true, speed = 1.2}
  object{name = "float", shape = "ball", at = {1.4, 0.4, 0.5}, radius = 0.09, float = true, density = 350}
  object{name = "boat", shape = "box", at = {1.5, 0.85, 0.5}, size = {0.3, 0.14, 0.08}, float = true, density = 400}

  fluid_metric{name = "fastest water (m/s)", field = "speed", op = "max"}

  camera{yaw = -60, pitch = 32, distance = 0.85, fov = 45}
  light{yaw = 300, pitch = 55, ambient = 0.4, strength = 0.8, ground = true, grid = false}
end

