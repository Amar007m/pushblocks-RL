import random
from train_rl import PushBlocksGymEnv

env = PushBlocksGymEnv(level_name="L1", max_steps=150)

obs, _ = env.reset()
pot_changes = 0
total_steps = 5000

last_pot = getattr(env, "prev_potential", None)

for t in range(total_steps):
    a = random.randint(0, 3)
    obs, r, terminated, truncated, info = env.step(a)

    # Falls du potential im info mitgibst, nimm info["potential"]
    # sonst nimm env.prev_potential
    pot = getattr(env, "prev_potential", None)

    if last_pot is not None and pot is not None and pot != last_pot:
        pot_changes += 1
    last_pot = pot

    if terminated or truncated:
        obs, _ = env.reset()
        last_pot = getattr(env, "prev_potential", None)

print("Potential changes:", pot_changes, "out of", total_steps, "steps")
