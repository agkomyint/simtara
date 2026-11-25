-- A forest fire, from physics and chemistry.
--   Combustion needs three things: fuel, heat above the ignition temperature, and oxygen.
--   Burning uses up fuel and releases heat. Heat conducts to neighbours and leaks to the air.
-- Turn the Controls while it runs: oxygen to 0 smothers the fire, rain cools and wets it,
-- a low ignition temperature makes the forest explosive.
-- Controls have owners: "nature" sets the conditions, "firefighters" are what a person can do.
-- Use "Play as" to try the fire as the fire crew: only their controls can be moved.
local N = 60

function setup()
  param{name = "ignite_temp", value = 180, min = 100, max = 600, about = "temperature where fuel starts to burn", owner = "nature"}
  param{name = "heat_release", value = 150, min = 0, max = 400, about = "heat given out per burning cell per tick", owner = "nature"}
  param{name = "burn_rate", value = 0.05, min = 0.005, max = 0.3, about = "fuel burnt per tick", owner = "nature"}
  param{name = "oxygen", value = 1.0, min = 0, max = 1, about = "0 = no air, nothing burns", owner = "nature"}
  param{name = "conduction", value = 0.25, min = 0, max = 0.5, about = "how fast heat moves to neighbours", owner = "nature"}
  param{name = "air_cooling", value = 0.03, min = 0, max = 0.3, about = "heat lost to the air", owner = "nature"}
  param{name = "rain", value = 0, min = 0, max = 1, about = "cools cells and puts fires out", owner = "nature"}
  param{name = "ambient", value = 20, min = 0, max = 40, about = "air temperature", owner = "nature"}

  -- what the fire crew can do: spray water on burning cells, and clear fuel next to the fire
  param{name = "water_drop", value = 0, min = 0, max = 5, about = "water sprayed on burning cells (costs water)", owner = "firefighters"}
  param{name = "crews", value = 0, min = 0, max = 30, about = "ground crews cutting a firebreak (each costs 10 per tick)", owner = "firefighters"}

  spawn{kind = "ops", props = {spent = 0}}
  local id = {}
  for gy = 0, N - 1 do
    for gx = 0, N - 1 do
      local river = (gx >= 28 and gx <= 30) and gy > 10
      local fuel = river and 0 or (0.5 + random() * 0.5)
      local spark = (gx - 12) ^ 2 + (gy - 30) ^ 2 < 4
      id[gy * N + gx] = spawn{kind = "cell", props = {x = gx, y = gy, fuel = fuel, temp = spark and 600 or 20, burning = 0, water_used = 0}}
    end
  end
  for gy = 0, N - 1 do
    for gx = 0, N - 1 do
      local me = id[gy * N + gx]
      if gx > 0 then relate{from = me, to = id[gy * N + gx - 1], kind = "next"} end
      if gx < N - 1 then relate{from = me, to = id[gy * N + gx + 1], kind = "next"} end
      if gy > 0 then relate{from = me, to = id[(gy - 1) * N + gx], kind = "next"} end
      if gy < N - 1 then relate{from = me, to = id[(gy + 1) * N + gx], kind = "next"} end
    end
  end

  -- chemistry: a cell burns while it has fuel, enough heat, and air
  add_rule{name = "ignite", kind = "cell", set = {burning = "if(fuel > 0.02 and temp > ignite_temp and oxygen > 0.05 and rain < 0.8, 1, 0)"}}
  -- burning consumes fuel (faster with more oxygen) and releases heat
  add_rule{name = "combust", kind = "cell", set = {
    fuel = "max(0, fuel - burning * burn_rate * oxygen)",
    temp = "temp + burning * heat_release * oxygen"}}
  -- physics: conduction to neighbours, cooling to the air, rain
  add_rule{name = "heat_flow", kind = "cell", set = {
    temp = "max(ambient, temp + conduction * (nmean(next, temp) - temp) - air_cooling * (temp - ambient) - rain * 0.15 * (temp - ambient))"}}

  -- each crew gets to a few cells near the fire per tick, chance-based, so more crews = bigger firebreak
  add_rule{name = "water", kind = "cell", set = {
    water_used = "water_used + water_drop * burning",
    temp = "max(ambient, temp - water_drop * 25 * burning)"}}
  add_rule{name = "clear", kind = "cell", set = {fuel = "if(temp > 60 and burning < 1 and random() < crews * 0.02, 0, fuel)"}}
  add_rule{name = "pay", kind = "ops", set = {spent = "spent + crews * 10"}}

  define_metric{name = "cells burning", kind = "cell", prop = "burning", op = "sum"}
  define_metric{name = "fuel left", kind = "cell", prop = "fuel", op = "sum"}
  define_metric{name = "crew cost", kind = "ops", prop = "spent", op = "sum"}
  define_metric{name = "water used", kind = "cell", prop = "water_used", op = "sum"}
  define_metric{name = "hottest cell", kind = "cell", prop = "temp", op = "max"}
  view{x = "x", y = "y", color = "temp", size = "fuel"}
end
