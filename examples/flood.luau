-- A river valley floods: rain, a swollen river, and a dam you can open. Real units.
--   A valley 3 km long falls from west to east. Upstream a dam holds back a reservoir; a river
--   winds down the valley past a town of small blocks. Lower the dam's crest to let the
--   reservoir out (a dam break), turn up the rain or the river, and watch where the water goes,
--   how deep it gets in the streets and how fast it runs.
--   Solved with the shallow-water equations (as HEC-RAS 2D and other flood models do): water is
--   kept exactly, flows downhill, is slowed by the ground's roughness (Manning) and the edge of
--   the flood moves over dry land.
-- Try: dam crest down to 15 m (a breach), rain 50 mm/h, river flow 300 m³/s.
function setup()
  param{name = "dam", value = 42, min = 10, max = 45, about = "height of the dam's crest, m above the datum (lower it to let the water out)", owner = "people"}
  param{name = "rain", value = 0, min = 0, max = 100, about = "rain, millimetres per hour", owner = "nature"}
  param{name = "river", value = 40, min = 0, max = 400, about = "river flow coming into the valley, m³/s", owner = "nature"}
  param{name = "roughness", value = 0.035, min = 0.01, max = 0.12, about = "Manning roughness of the ground", owner = "nature"}

  flood{size = {300, 150}, cell = 10, dt = 2, manning = "roughness", vertical = 3, sea_level = -100,
        edges = {west = "wall", north = "wall", south = "wall", east = "open"}}
  -- the valley: falling to the east, steep sides, a winding river bed
  ground{ramp = {0, 0, 60, 3000, 0, 20}}
  ground{channel = {0, 750, 3000, 750}, width = 1500, depth = 35}
  ground{noise = 4, scale = 300, seed = 7}
  ground{channel = {0, 750, 600, 700, 1200, 800, 1700, 700, 2200, 760, 2600, 820, 3000, 760}, width = 40, depth = 4}
  -- the reservoir behind the dam
  water_level{level = 40, box = {0, 0, 690, 1500}}
  gate{box = {690, 0, 710, 1500}, height = "dam"}
  -- the town on the north bank
  for i = 0, 9 do
    for j = 0, 4 do
      local x = 2000 + i * 50
      local y = 860 + j * 45
      ground{box = {x, y, x + 28, y + 28}, height = 6 + ((i + j * 2) % 3) * 3, building = true}
    end
  end
  river{at = {30, 750}, radius = 20, flow = "river", velocity = {1, 0}}
  rain{rate = "rain"}

  flood_metric{name = "town flooded (m²)", field = "wet", op = "sum", box = {2000, 860, 2500, 1100}}
  flood_metric{name = "deepest water in town (m)", field = "depth", op = "max", box = {2000, 860, 2500, 1100}}
  flood_metric{name = "fastest water in town (m/s)", field = "speed", op = "max", box = {2000, 860, 2500, 1100}}
  flood_metric{name = "water in the reservoir (m³)", field = "depth", op = "sum", box = {0, 0, 690, 1500}}

  camera{yaw = -125, pitch = 35, distance = 0.7, fov = 45}
  light{yaw = 150, pitch = 45, ambient = 0.45, strength = 0.8, ground = false, grid = false}
end
