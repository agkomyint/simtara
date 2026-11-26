-- Water: a dam breaks, in 3D, in real units.
--   A tank 2.56 m long, 0.96 m wide and 1.28 m high. At the start a wall of water 0.7 m long and
--   0.9 m high stands at one end; then it is let go. It crashes into a block, splashes up the far
--   wall, slops back and finally settles flat.
--   Up to 300,000 particles carry the water (APIC, a particle-grid method used for film water), solved on the GPU;
--   a grid keeps it incompressible and holds its volume. Colour: blue water, white spray.
-- Try: turn on the tap, or lower gravity (Moon: 1.6).
-- Detail: "fast" (4 cm cells, ~80,000 particles, close to real time) or
-- "fine" (2.5 cm cells, ~300,000 particles, sharper splashes, about a fifth of real time).
local DETAIL = "fast"

function setup()
  param{name = "gravity", value = 9.81, min = 0, max = 20, about = "m/s² (Earth 9.81, Moon 1.6)", owner = "nature"}
  param{name = "tap", value = 0, min = 0, max = 0.02, about = "a tap pouring in, m³/s", owner = "people"}

  -- the tank is 2.56 x 0.96 x 1.28 m; DETAIL (top of the file) sets the size of a cell
  local cell = DETAIL == "fine" and 0.025 or 0.04
  local function cells(metres) return math.floor(metres / cell + 0.5) end
  fluid{kind = "water", size = {cells(2.56), cells(0.96), cells(1.28)}, cell = cell, dt = 1 / 50, gravity = "gravity", iterations = 60}
  water{box = {0, 0, 0, 0.7, 0.96, 0.9}}
  solid{box = {1.5, 0.3, 0, 1.75, 0.66, 0.25}}              -- a block on the floor
  emitter{at = {2.3, 0.48, 1.1}, radius = 0.06, water = "tap", velocity = {-0.5, 0, -1}}

  fluid_metric{name = "water past the block (m³)", field = "water", op = "sum", box = {1.75, 0, 0, 2.56, 0.96, 1.28}}

  camera{yaw = -60, pitch = 25, distance = 0.9, fov = 45}
  light{yaw = 300, pitch = 55, ambient = 0.35, strength = 0.85, ground = true, grid = true}
end
