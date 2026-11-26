-- A tsunami hits a coastal town, in real units (metres, seconds).
--   A 4 km x 3 km stretch of coast: the sea floor rises from 60 m deep to a sandy beach, a town
--   of small blocks sits behind the beach between two headlands, hills rise behind it.
--   Offshore, an earthquake lifts the sea floor by a few metres. The sea above it rises with it
--   and spreads out as a tsunami: fast in deep water, slowing and growing as it reaches shallow
--   water, then running up the beach and through the streets.
--   Solved with the shallow-water equations (the method tsunami warning centres use), checked
--   against the Synolakis run-up experiment and exact dam-break solutions.
-- Try: a bigger uplift, rougher ground (more friction slows the flood), a storm wind of 40 m/s
-- onto the shore (it piles the sea up against the coast, as a storm surge does).
function setup()
  param{name = "uplift", value = 8, min = 0, max = 12, about = "how far the sea floor lifts in the earthquake, m", owner = "nature"}
  param{name = "roughness", value = 0.03, min = 0.01, max = 0.1, about = "Manning roughness of the ground (0.015 concrete, 0.03 grass, 0.1 forest)", owner = "nature"}
  param{name = "wind", value = 12, min = 0, max = 60, about = "wind speed 10 m above the sea, m/s (storm 25, hurricane 50+)", owner = "nature"}
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
  quake{at = {1000, 1500}, size = {500, 1100}, uplift = "uplift", when = 3}

  flood_metric{name = "town flooded (m²)", field = "wet", op = "sum", box = {2790, 1000, 3270, 1900}}
  flood_metric{name = "deepest water in town (m)", field = "depth", op = "max", box = {2790, 1000, 3270, 1900}}
  flood_metric{name = "wave reaches town (s)", field = "arrival", op = "min", box = {2790, 1000, 3270, 1900}}
  flood_metric{name = "highest sea level (m)", field = "level", op = "max", box = {0, 0, 2400, 3000}}

  camera{yaw = -150, pitch = 32, distance = 0.75, fov = 45}
  light{yaw = 200, pitch = 40, ambient = 0.45, strength = 0.8, ground = false, grid = false}
end
