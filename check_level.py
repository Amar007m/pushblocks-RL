from pushblocks_core import LEVELS, parse_level, State, astar_plan, is_solved

level_name = "L1"
level = parse_level(LEVELS[level_name])
state = State(player=level.player_start, blocks=frozenset(level.blocks_start))

plan = astar_plan(level, state)

print("Level:", level_name)
print("Blocks:", len(state.blocks), "Goals:", len(level.goals))
print("A* plan length:", None if plan is None else len(plan))
print("Solved initially:", is_solved(state.blocks, level.goals))