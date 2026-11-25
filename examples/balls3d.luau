-- Bouncing things in 3D. The camera and light belong to the universe: turn them in the Universe tab,
-- or place them from the script with camera{...} and light{...}. Shapes: sphere, cube, pyramid,
-- or any .obj in this project's models/ folder (here: ship).
function setup()
  param{name = "gravity", value = 0.04, min = 0, max = 0.3, about = "pull towards the ground", owner = "nature"}
  param{name = "bounce", value = 0.85, min = 0, max = 1, about = "how much speed a bounce keeps", owner = "nature"}
  param{name = "wind", value = 0.0, min = -0.3, max = 0.3, about = "push along x", owner = "nature"}
  param{name = "shake", value = 0, min = 0, max = 1, about = "random kicks to every ball", owner = "human"}

  local shapes = {"sphere", "cube", "pyramid", "ship"}
  for i = 1, 120 do
    spawn{kind = "ball", props = {
      x = random() * 20, y = random() * 20, z = random() * 8,
      vx = (random() - 0.5) * 0.1, vy = (random() - 0.5) * 0.1, vz = 0,
      heading = random() * 360, shape = shapes[(i % 4) + 1]}}
  end

  add_rule{name = "kick", kind = "ball", set = {vz = "vz - gravity + shake * (random() - 0.5) * 0.3"}}
  add_rule{name = "move", kind = "ball", set = {
    x = "(x + vx + wind + 20) % 20", y = "(y + vy + 20) % 20", z = "max(0, z + vz)", heading = "(heading + 3) % 360"}}
  add_rule{name = "bounce", kind = "ball", set = {vz = "if(z <= 0 and vz < 0, 0 - vz * bounce, vz)"}}

  define_metric{name = "average height", kind = "ball", prop = "z", op = "mean"}
  camera{yaw = 35, pitch = 25, distance = 0.9}
  light{yaw = 300, pitch = 55, ambient = 0.3, strength = 0.8}
  view{x = "x", y = "y", z = "z", color = "z", size = "z", shape = "shape", mode = "3d"}
end
