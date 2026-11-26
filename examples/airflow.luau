-- Air flowing around a building, in 3D: a small wind tunnel.
--   The tunnel is a 3D grid of air cells. Each cell has a velocity (vx, vy, vz) and a pressure p.
--   Every tick: air carries itself and smoke downstream (advection), rubs on its neighbours
--   (viscosity), and pressure pushes air out of crowded cells into empty ones (so air goes
--   around the building instead of through it). Wind blows in at x = 0 and leaves at the far end.
--   Smoke is released in streaks at the inlet so you can see the flow. Colour = speed.
-- This is a simplified fluid model (coarse grid, artificial compressibility), good for seeing
-- how flow behaves, not for engineering numbers.
-- Change SHAPE below to "two buildings" or "ball" for a different obstacle.
local NX, NY, NZ = 36, 12, 14
local SHAPE = "building"

local function solid_at(x, y, z)
  if z == 0 then return true end -- the ground
  if SHAPE == "ball" then
    return (x - 12) ^ 2 + (y - 5.5) ^ 2 + (z - 6) ^ 2 < 10
  elseif SHAPE == "two buildings" then
    return ((x >= 9 and x <= 11) or (x >= 17 and x <= 19)) and y >= 3 and y <= 8 and z <= 8
  end
  return x >= 10 and x <= 13 and y >= 4 and y <= 7 and z <= 8
end

-- formula pieces, built here so the rules stay readable
local function n(rel, q) return "nmean(" .. rel .. ", " .. q .. ")" end
local function upwind(q)
  return "max(vx, 0) * (" .. q .. " - " .. n("xm", q) .. ") + min(vx, 0) * (" .. n("xp", q) .. " - " .. q .. ")"
    .. " + max(vy, 0) * (" .. q .. " - " .. n("ym", q) .. ") + min(vy, 0) * (" .. n("yp", q) .. " - " .. q .. ")"
    .. " + max(vz, 0) * (" .. q .. " - " .. n("zm", q) .. ") + min(vz, 0) * (" .. n("zp", q) .. " - " .. q .. ")"
end
local function laplace(q)
  return "(" .. n("xm", q) .. " + " .. n("xp", q) .. " + " .. n("ym", q) .. " + " .. n("yp", q)
    .. " + " .. n("zm", q) .. " + " .. n("zp", q) .. " - 6 * " .. q .. ")"
end
local function grad(q, lo, hi) return "0.5 * (" .. n(hi, q) .. " - " .. n(lo, q) .. ")" end
local DIV = "(" .. grad("vx", "xm", "xp") .. " + " .. grad("vy", "ym", "yp") .. " + " .. grad("vz", "zm", "zp") .. ")"

function setup()
  param{name = "wind", value = 0.3, min = 0, max = 0.45, about = "speed of the air blowing in (cells per tick)", owner = "nature"}
  param{name = "viscosity", value = 0.04, min = 0, max = 0.15, about = "how thick the air is; low = more swirls", owner = "nature"}
  param{name = "gusts", value = 0.01, min = 0, max = 0.05, about = "random turbulence added to the air", owner = "nature"}
  param{name = "stiffness", value = 0.45, min = 0.1, max = 0.6, about = "how strongly pressure keeps air from piling up", owner = "nature"}
  param{name = "smoke_amount", value = 1, min = 0, max = 1, about = "smoke released at the inlet", owner = "human"}
  param{name = "smoke_fade", value = 0.004, min = 0, max = 0.05, about = "how fast smoke thins out", owner = "nature"}

  -- the cells
  local id = {}
  local function at(x, y, z) return id[(z * NY + y) * NX + x] end
  for z = 0, NZ - 1 do
    for y = 0, NY - 1 do
      for x = 0, NX - 1 do
        local solid = solid_at(x, y, z)
        local streak = x == 0 and not solid and y % 3 == 1 and z % 3 == 1
        id[(z * NY + y) * NX + x] = spawn{kind = "cell", props = {
          x = x, y = y, z = z,
          vx = solid and 0 or 0.3, vy = 0, vz = 0, p = 0, speed = 0,
          smoke = 0, puff = solid and 1 or 0, source = streak and 1 or 0,
          solid = solid and 1 or 0, inlet = x == 0 and 1 or 0, outlet = x == NX - 1 and 1 or 0,
          shape = solid and "cube" or "sphere"}}
      end
    end
  end
  -- each cell knows its neighbour in each direction; at the tunnel's edge it points at itself,
  -- so nothing flows through the walls of the tunnel
  for z = 0, NZ - 1 do
    for y = 0, NY - 1 do
      for x = 0, NX - 1 do
        local me = at(x, y, z)
        relate{from = me, to = at(math.max(x - 1, 0), y, z), kind = "xm"}
        relate{from = me, to = at(math.min(x + 1, NX - 1), y, z), kind = "xp"}
        relate{from = me, to = at(x, math.max(y - 1, 0), z), kind = "ym"}
        relate{from = me, to = at(x, math.min(y + 1, NY - 1), z), kind = "yp"}
        relate{from = me, to = at(x, y, math.max(z - 1, 0)), kind = "zm"}
        relate{from = me, to = at(x, y, math.min(z + 1, NZ - 1)), kind = "zp"}
      end
    end
  end

  -- probes: one reads the air pressing on the front of the obstacle, one the air in its wake
  local front = spawn{kind = "front_probe", props = {x = 2, y = NY / 2, z = NZ - 2, p = 0, speed = 0, shape = "pyramid"}}
  local wake = spawn{kind = "wake_probe", props = {x = NX - 3, y = NY / 2, z = NZ - 2, p = 0, speed = 0, shape = "pyramid"}}
  for z = 1, NZ - 1 do
    for y = 0, NY - 1 do
      local first, last = nil, nil
      for x = 1, NX - 1 do
        if solid_at(x, y, z) then
          first = first or x
          last = x
        end
      end
      if first then
        relate{from = front, to = at(first - 1, y, z), kind = "sees"}
        relate{from = wake, to = at(math.min(last + 3, NX - 1), y, z), kind = "sees"}
      end
    end
  end

  -- 1. air carries itself and the smoke along (upwind differences, all read the old state)
  add_rule{name = "carry", kind = "cell", set = {
    vx = "vx - (" .. upwind("vx") .. ")",
    vy = "vy - (" .. upwind("vy") .. ")",
    vz = "vz - (" .. upwind("vz") .. ")",
    smoke = "smoke - (" .. upwind("smoke") .. ")"}}
  -- 2. viscosity: neighbouring air drags on each other; smoke spreads a little; gusts stir it
  add_rule{name = "rub", kind = "cell", set = {
    vx = "vx + viscosity * " .. laplace("vx"),
    vy = "vy + viscosity * " .. laplace("vy") .. " + gusts * (random() - 0.5)",
    vz = "vz + viscosity * " .. laplace("vz") .. " + gusts * (random() - 0.5)",
    smoke = "smoke + 0.02 * " .. laplace("smoke")}}
  -- 3. pressure: air piling up raises pressure, pressure pushes air away (two rounds per tick)
  for round = 1, 2 do
    add_rule{name = "squeeze " .. round, kind = "cell", set = {p = "0.99 * p - stiffness * " .. DIV}}
    add_rule{name = "push " .. round, kind = "cell", set = {
      vx = "vx - stiffness * " .. grad("p", "xm", "xp"),
      vy = "vy - stiffness * " .. grad("p", "ym", "yp"),
      vz = "vz - stiffness * " .. grad("p", "zm", "zp")}}
  end
  -- 4. edges: no air moves inside solids, wind and smoke come in at the inlet, the outlet is open
  add_rule{name = "edges", kind = "cell", set = {
    -- speeds are capped at half a cell per tick, the most this grid can carry without blowing up
    vx = "if(solid > 0, 0, if(inlet > 0, wind, clamp(vx, 0 - 0.5, 0.5)))",
    vy = "if(solid > 0 or inlet > 0, 0, clamp(vy, 0 - 0.5, 0.5))",
    vz = "if(solid > 0 or inlet > 0, 0, clamp(vz, 0 - 0.5, 0.5))",
    p = "if(outlet > 0, 0, clamp(p, 0 - 2, 2))",
    smoke = "if(solid > 0, 0, if(inlet > 0, source * smoke_amount, clamp(smoke * (1 - smoke_fade), 0, 1)))"}}
  add_rule{name = "measure", kind = "cell", set = {
    speed = "sqrt(vx * vx + vy * vy + vz * vz)",
    puff = "if(solid > 0, 1, smoke)"}}
  add_rule{name = "read front", kind = "front_probe", set = {p = "nmean(sees, p)", speed = "nmean(sees, speed)"}}
  add_rule{name = "read wake", kind = "wake_probe", set = {p = "nmean(sees, p)", speed = "nmean(sees, speed)"}}

  define_metric{name = "pressure on front", kind = "front_probe", prop = "p", op = "mean"}
  define_metric{name = "wind behind obstacle", kind = "wake_probe", prop = "speed", op = "mean"}
  define_metric{name = "fastest air", kind = "cell", prop = "speed", op = "max"}
  define_metric{name = "average air speed", kind = "cell", prop = "speed", op = "mean"}
  define_metric{name = "smoke in tunnel", kind = "cell", prop = "smoke", op = "sum"}

  camera{yaw = 30, pitch = 28, distance = 0.95, fov = 45}
  light{yaw = 300, pitch = 55, ambient = 0.35, strength = 0.8, ground = false, grid = false}
  view{x = "x", y = "y", z = "z", color = "speed", size = "puff", shape = "shape", mode = "3d"}
end
