-- Chemistry: two chemicals react and spread (Gray-Scott reaction-diffusion).
--   u is a food chemical, v feeds on it:   u + 2v -> 3v.   Food is added ("feed"), v is removed ("kill").
-- Different feed/kill settings grow spots, stripes, mazes or nothing. Press play, speed the clock up,
-- and move the Controls. Colour shows v.
local N = 64

function setup()
  param{name = "feed", value = 0.037, min = 0.01, max = 0.09, about = "food added per tick"}
  param{name = "kill", value = 0.065, min = 0.04, max = 0.07, about = "v removed per tick"}
  param{name = "diffuse_u", value = 0.5, min = 0.05, max = 0.9, about = "how fast u spreads"}
  param{name = "diffuse_v", value = 0.25, min = 0.02, max = 0.5, about = "how fast v spreads"}

  local id = {}
  for gy = 0, N - 1 do
    for gx = 0, N - 1 do
      local seed = (gx >= 28 and gx < 36 and gy >= 28 and gy < 36) or random() < 0.01
      id[gy * N + gx] = spawn{kind = "cell", props = {x = gx, y = gy, u = 1, v = seed and 0.5 or 0}}
    end
  end
  for gy = 0, N - 1 do
    for gx = 0, N - 1 do
      local me = id[gy * N + gx]
      -- the plate wraps around, so there are no edges
      relate{from = me, to = id[gy * N + (gx - 1) % N], kind = "next"}
      relate{from = me, to = id[gy * N + (gx + 1) % N], kind = "next"}
      relate{from = me, to = id[((gy - 1) % N) * N + gx], kind = "next"}
      relate{from = me, to = id[((gy + 1) % N) * N + gx], kind = "next"}
    end
  end
  add_rule{name = "react", kind = "cell", set = {
    u = "clamp(u + diffuse_u * (nmean(next, u) - u) - u * v * v + feed * (1 - u), 0, 1)",
    v = "clamp(v + diffuse_v * (nmean(next, v) - v) + u * v * v - (feed + kill) * v, 0, 1)"}}
  define_metric{name = "total v", kind = "cell", prop = "v", op = "sum"}
  define_metric{name = "total u", kind = "cell", prop = "u", op = "sum"}
  view{x = "x", y = "y", color = "v"}
end
