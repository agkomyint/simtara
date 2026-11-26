-- A rock falls into the sea off a coastal town, in real units (metres, seconds).
--   The same coast as the tsunami example. Set "drop" to 1 and a rock hits the sea 1.5 km out.
--   Its kinetic energy (half m v squared) blasts a crater in the water; 15 % of the energy goes
--   into the crater (Ward & Asphaug 2000, the method used for asteroid-impact tsunamis). The
--   crater collapses into rings of waves that run at the speed of shallow-water waves,
--   sqrt(g x depth) (24 m/s here), lower as they spread, and climb the beach into the town.
--   Checked by tests: the crater holds exactly that energy, no water is made or lost, the rings
--   run at the wave speed and fall off as 1 / sqrt(distance).
--   Set "drop" back to 0 and up to 1 again for another rock. Watch impact.crater_radius_m.
-- Try: a 100 m rock at 1000 m/s, an iron rock (7800 kg/m3), a slow 300 m one.
function setup()
  param{name = "drop", value = 0, min = 0, max = 1, about = "set to 1 to drop the rock (back to 0, then 1 again, for another)", owner = "people"}
  param{name = "diameter", value = 40, min = 5, max = 300, about = "rock diameter, m", owner = "people"}
  param{name = "speed", value = 500, min = 50, max = 20000, about = "speed it hits the water at, m/s (asteroids 11000-20000)", owner = "people"}
  param{name = "rock", value = 3000, min = 500, max = 8000, about = "rock density, kg/m3 (ice 900, stone 3000, iron 7800)", owner = "people"}
  param{name = "roughness", value = 0.03, min = 0.01, max = 0.1, about = "Manning roughness of the ground (0.015 concrete, 0.03 grass, 0.1 forest)", owner = "nature"}
  param{name = "wind", value = 5, min = 0, max = 60, about = "wind speed 10 m above the sea, m/s (storm 25, hurricane 50+)", owner = "nature"}
  param{name = "wind_dir", value = 0, min = 0, max = 360, about = "direction the wind blows towards, degrees (0 = east, onto the shore; 90 = north)", owner = "nature"}

  flood{size = {200, 150}, cell = 20, dt = 1, manning = "roughness", vertical = 4, sea_level = 0, density = 1025,
        edges = {west = "open", north = "open", south = "open", east = "wall"}}
  -- sea floor rising to the beach, land rising behind it
  ground{ramp = {0, 0, -60, 2500, 0, -2}}
  ground{ramp = {2500, 0, -2, 2750, 0, 1.5}, within = {2500, 0, 2750, 3000}}
  ground{ramp = {2750, 0, 1.5, 4000, 0, 25}, within = {2750, 0, 4000, 3000}}
  ground{noise = 5, scale = 700, seed = 3}
  -- two headlands that make a bay
  ground{hill = {2700, 350, 650, 45}}
  ground{hill = {2650, 2700, 600, 40}}
  -- a river through the town to the sea
  ground{channel = {4000, 1600, 3300, 1450, 2900, 1550, 2500, 1500}, width = 50, depth = 3}
  -- the town: blocks of houses on a street grid
  for i = 0, 7 do
    for j = 0, 9 do
      local x = 2790 + i * 60
      local y = 1000 + j * 90
      if math.abs(y - 1500) > 60 then
        ground{box = {x, y, x + 35, y + 50}, height = 8 + ((i * 7 + j * 3) % 4) * 3, building = true}
      end
    end
  end
  water_level{level = 0}
  -- the wind drags on the sea (storm surge); the flags show where it blows
  wind{speed = "wind", direction = "wind_dir"}
  flag{at = {2760, 1450}, height = 40}
  flag{at = {3300, 1950}, height = 40}
  impact{at = {1200, 1500}, diameter = "diameter", speed = "speed", density = "rock", trigger = "drop"}

  flood_metric{name = "town flooded (m²)", field = "wet", op = "sum", box = {2790, 1000, 3270, 1900}}
  flood_metric{name = "deepest water in town (m)", field = "depth", op = "max", box = {2790, 1000, 3270, 1900}}
  flood_metric{name = "wave reaches town (s)", field = "arrival", op = "min", box = {2790, 1000, 3270, 1900}}
  flood_metric{name = "highest sea level (m)", field = "level", op = "max", box = {0, 0, 2400, 3000}}

  camera{yaw = -150, pitch = 32, distance = 0.75, fov = 45}
  light{yaw = 200, pitch = 40, ambient = 0.45, strength = 0.8, ground = false, grid = false}
end
