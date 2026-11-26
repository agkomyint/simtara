-- Pouring a real liquid into a glass, in 3D, at real size (metres, seconds, kg).
--   A glass 7.6 cm wide and 11 cm tall stands on a table. A stream pours in from above.
--   The liquid has the real viscosity and density of the one you choose below, from standard
--   tables at room temperature. Watch how water splashes and swirls, while honey piles up,
--   folds over itself and spreads slowly (its viscosity is about 4,000 times water's).
--
-- Choose the liquid (then press Run again):
--   "water", "sea water", "milk", "mercury", "olive oil", "glycerine", "honey", "lava"
local LIQUID = "honey"
--
-- Compare two liquids: Experiment tab, change LIQUID in the second run, compare "liquid in the glass".
-- Note: a single liquid falls the same at any density (mercury pours like water); density gives
-- its mass. What makes honey different is viscosity.

function setup()
  param{name = "pour", value = 40, min = 0, max = 120, about = "pour rate, millilitres per second", owner = "people"}
  param{name = "gravity", value = 9.81, min = 0, max = 20, about = "m/s² (Earth 9.81, Moon 1.6)", owner = "nature"}

  fluid{kind = "water", liquid = LIQUID, size = {40, 40, 46}, cell = 0.006, dt = 1 / 60, gravity = "gravity", iterations = 60}
  solid{box = {0, 0, 0, 0.24, 0.24, 0.012}}                       -- the table top
  solid{cup = {0.12, 0.12, 0.038, 0.006, 0.012, 0.12}}            -- the glass
  -- the stream: a 1 cm wide spout 15 cm above the table, pointing down
  emitter{at = {0.11, 0.12, 0.25}, radius = 0.006, water_ml = "pour", velocity = {0.05, 0, -0.6}}

  fluid_metric{name = "liquid in the glass (m³)", field = "water", op = "sum", box = {0.082, 0.082, 0.018, 0.158, 0.158, 0.12}}
  fluid_metric{name = "liquid spilled (m³)", field = "water", op = "sum", box = {0, 0, 0.012, 0.24, 0.24, 0.018}}
  fluid_metric{name = "fastest liquid (m/s)", field = "speed", op = "max"}

  camera{yaw = -60, pitch = 22, distance = 0.95, fov = 40}
  light{yaw = 300, pitch = 55, ambient = 0.4, strength = 0.85, ground = true, grid = false}
end

