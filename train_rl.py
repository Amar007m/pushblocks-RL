# train_rl.py
# PPO Training für PushBlocks (Level L1) – mit Multi-Process Parallelisierung (SubprocVecEnv)
#
# ✅ Warum das schneller ist:
# - Mehrere Environments laufen parallel (z.B. 8 Stück)
# - PPO sammelt dadurch viel mehr Erfahrung pro "Wall-Clock"-Zeit
#
# Install:
#   pip install gymnasium stable-baselines3 torch numpy
#
# Run:
#   python train_rl.py
#
# Output:
#   pushblocks_ppo_L1.zip
# Optional checkpoints:
#   checkpoints/ppo_L1_XXXX_steps.zip

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from pushblocks_core import LEVELS, parse_level, State, build_obs, apply_action, is_solved


# -------------------------
# Gym Environment
# -------------------------
class PushBlocksGymEnv(gym.Env):
    """
    Observation: flattened vector of (4, H, W) => (4*H*W,)
    Action: 0..3 (L, R, U, D)
    """
    metadata = {"render_modes": []}

    def __init__(self, level_name="L1", max_steps=150):
        super().__init__()
        self.level = parse_level(LEVELS[level_name])
        self.max_steps = max_steps

        self.obs_size = 4 * self.level.h * self.level.w
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.state: State = None  # type: ignore
        self.steps = 0
        self.prev_on_goal = 0
        self.last_action = None

    def _min_goal_dist(self, block_pos):
        bx, by = block_pos
        return min(abs(bx-gx) + abs(by-gy) for (gx, gy) in self.level.goals)
    
    def _potential(self, state: State) -> float:
    # kleiner ist besser
        return float(sum(self._min_goal_dist(b) for b in state.blocks))


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_action = None
        self.state = State(
            player=self.level.player_start,
            blocks=frozenset(self.level.blocks_start),
        )
        self.steps = 0
        self.prev_on_goal = self._count_on_goal(self.state)
        obs = build_obs(self.level, self.state).flatten().astype(np.float32)
        self.prev_potential = self._potential(self.state)
        print("start solved?", is_solved(self.state.blocks, self.level.goals))
        return obs, {}

    def _count_on_goal(self, state: State) -> int:
        return sum((b in self.level.goals) for b in state.blocks)

    def step(self, action):
        self.steps += 1

        # --- weights (tunen) ---
        STEP_PENALTY = -0.02           # gegen Rumlaufen
        ILLEGAL_PENALTY = -0.2
        NO_PUSH_PENALTY = -0.01        # klein halten, sonst wird er push-fixiert
        PUSH_BONUS = 0.05              # nur fürs "Push passiert"
        PROGRESS_COEF = 0.7            # shaping über Distanz
        GOOD_PUSH_BONUS = 0.15         # extra wenn push Fortschritt bringt
        BAD_PUSH_PENALTY = -0.15
        REVERSE_PENALTY = -0.03
        REV = {0:1, 1:0, 2:3, 3:2}       # wenn push schlechter macht

        reward = 0.0
        terminated = False
        truncated = False

        # potential BEFORE
        pot_before = self._potential(self.state)

        blocks_before = self.state.blocks

        # base step penalty
        reward += STEP_PENALTY

        nxt = apply_action(self.level, self.state, int(action))
        self.last_action = int(action)
        if nxt is None:
            reward += ILLEGAL_PENALTY
            nxt = self.state
        self.state = nxt

        blocks_after = self.state.blocks
        pushed = (blocks_after != blocks_before)

        # potential AFTER
        pot_after = self._potential(self.state)

        # reward progress (smaller potential is better)
        delta = pot_before - pot_after       # >0 means progress
        reward += PROGRESS_COEF * delta
        

        # push-specific reward
        if pushed:
            reward += PUSH_BONUS
            if delta > 0:
                reward += GOOD_PUSH_BONUS
            elif delta < 0:
                reward += BAD_PUSH_PENALTY
        else:
            reward += NO_PUSH_PENALTY

        # shaping: boxes on goals
        on_goal = self._count_on_goal(self.state)
        if on_goal > self.prev_on_goal:
            reward += 1.0 * (on_goal - self.prev_on_goal)
        elif on_goal < self.prev_on_goal:
            reward -= 2.0 * (self.prev_on_goal - on_goal)
        self.prev_on_goal = on_goal

        # penalty for reversing last action
        if self.last_action is not None and REV[action] == self.last_action:
            reward += REVERSE_PENALTY

        solved = is_solved(self.state.blocks, self.level.goals)
        if solved:
            reward += 50.0
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True
            reward -= 5.0

        obs = build_obs(self.level, self.state).flatten().astype(np.float32)
        info = {"solved": solved, "on_goal": on_goal, "pushed": pushed, "potential": pot_after}

        return obs, reward, terminated, truncated, info


# -------------------------
# Callback: solved rate logging
# -------------------------
class SolvedRateCallback(BaseCallback):
    """
    Prints solved-rate every N timesteps (global timesteps across all envs).
    """
    def __init__(self, check_every=10_000):
        super().__init__()
        self.check_every = check_every
        self.episodes = 0
        self.solved_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if done:
                self.episodes += 1
                if info.get("solved", False):
                    self.solved_count += 1

        if self.num_timesteps % self.check_every == 0 and self.episodes > 0:
            rate = self.solved_count / self.episodes
            print(f"[SolvedRate] episodes={self.episodes} "
                  f"solved={self.solved_count} "
                  f"rate={rate:.3f} "
                  f"pushed={info.get('push_happend', 0)} "
                  f"potential={info.get('potential', 0):.2f}")

        return True


# -------------------------
# Factory for SubprocVecEnv
# IMPORTANT for Windows: must be top-level function
# -------------------------
def make_env(level_name: str, max_steps: int, seed: int):
    def _init():
        env = PushBlocksGymEnv(level_name=level_name, max_steps=max_steps)
        env.reset(seed=seed)
        return env
    return _init


def main():
    level_name = "L2"

    # --- SPEED SETTINGS ---
    n_envs = 2          # number of parallel processes (try 4/8/12 depending on CPU)
    max_steps = 150     # shorter episodes train faster
    total_timesteps = 20_000  # start smaller; increase if needed

    # Build parallel env
    env = SubprocVecEnv([make_env(level_name, max_steps, seed=1000 + i) for i in range(n_envs)])


    # For debugging, you can use DummyVecEnv (no subprocesses)
    #env = DummyVecEnv([make_env(level_name, max_steps, seed=1000 + i) for i in range(n_envs)])
    
    
    # PPO settings:
    # With n_envs parallel envs, the effective batch collected per rollout is:
    #   rollout_steps = n_steps * n_envs
    # Keep rollout ~ 2048-8192 to be reasonable.
    finetune_target = f'pushblocks_ppo_{level_name}.zip'
    pretrained = "pushblocks_ppo_L1.zip"

    if os.path.exists(finetune_target):
        print(f"🔄 Loading existing model: {finetune_target}")
        model = PPO.load(finetune_target, env=env)
    elif os.path.exists(pretrained):
        print(f"🔄 Loading existing model: {pretrained}")
        model = PPO.load(pretrained, env=env)
    else:
        print(f"🚀 Training new model for level {level_name}...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=256,          # per env
            batch_size=256,       # training minibatch size
            learning_rate=3e-4,
            gamma=0.99,
            ent_coef=0.05,         # encourage exploration
        )
        

    # Optional: checkpoints so you can test early without waiting
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,            # global timesteps
        save_path="checkpoints",
        name_prefix=f"ppo_{level_name}",
    )

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=False,
        callback=[SolvedRateCallback(check_every=5_000), checkpoint_cb],
    )
    
    model.save(f"pushblocks_ppo_{level_name}")
    print(f"✅ Saved final model: pushblocks_ppo_{level_name}.zip")


if __name__ == "__main__":
    # Windows needs this guard for SubprocVecEnv
    main()
