-- An idea spreads through 1,500 people. Test: simtara script examples/idea.luau --ticks 120 --expect "people who know > 100"
local PEOPLE = 1500
function setup()
  local ids = {}
  for i = 1, PEOPLE do ids[i] = spawn{kind = "person", props = {knows = (i <= 3)}} end
  for i = 1, PEOPLE do
    for _ = 1, 6 do relate{from = ids[i], to = ids[1 + math.floor(random() * PEOPLE)], kind = "friend"} end
  end
  define_metric{name = "people who know", kind = "person", prop = "knows", op = "sum"}
  add_rule{name = "spread", kind = "person", when = "knows == 0 and random() < 0.12 * ncount(friend, knows)", set = {knows = "true"}}
end
