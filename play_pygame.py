# =========================
# FILE 3: play_pygame.py
# =========================
# pygame Demo: HUMAN / ASTAR / RL
#
# Install:
#   pip install pygame stable-baselines3 gymnasium torch numpy
#
# Run:
#   python play_pygame.py
#
# Keys:
#   1 = HUMAN
#   2 = ASTAR
#   3 = RL (loads pushblocks_ppo_Lx.zip if present)
#   F1/F2/F3 = switch Level (L1/L2/L3)
#   ESC = reset current level
#   Q = quit

import pygame
from stable_baselines3 import PPO

from pushblocks_core import (
    LEVELS, parse_level, State, build_obs, apply_action, is_solved, astar_plan
)

pygame.init()

CELL = 32   # rendering scale (logic is grid-based)
FPS = 30

# Colors
BLACK = (0, 0, 0)
WALL = (90, 90, 90)
GOAL = (0, 160, 0)
BOX = (245, 245, 245)
BOX_ON_GOAL = (255, 220, 0)
PLAYER = (220, 50, 50)
TEXT = (220, 220, 220)


def make_window(level):
    w = level.w * CELL
    h = level.h * CELL
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("PushBlocks (Human / A* / RL)")
    return screen


def draw(screen, level, state: State):
    screen.fill(BLACK)

    # tiles (walls/goals)
    for y in range(level.h):
        for x in range(level.w):
            r = pygame.Rect(x * CELL, y * CELL, CELL, CELL)
            if (x, y) in level.walls:
                pygame.draw.rect(screen, WALL, r)
            elif (x, y) in level.goals:
                pygame.draw.rect(screen, GOAL, r)

    # boxes
    for (x, y) in state.blocks:
        r = pygame.Rect(x * CELL + 4, y * CELL + 4, CELL - 8, CELL - 8)
        color = BOX_ON_GOAL if (x, y) in level.goals else BOX
        pygame.draw.rect(screen, color, r)

    # player
    px, py = state.player
    pr = pygame.Rect(px * CELL + 6, py * CELL + 6, CELL - 12, CELL - 12)
    pygame.draw.rect(screen, PLAYER, pr)


def get_human_action():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return 0
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return 1
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        return 2
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        return 3
    return None


def main():
    level_name = "L1"
    level = parse_level(LEVELS[level_name])
    state = State(player=level.player_start, blocks=frozenset(level.blocks_start))

    screen = make_window(level)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    mode = "HUMAN"  # HUMAN / ASTAR / RL
    plan = []
    plan_failed = False

    rl_model = None
    rl_model_name = f"pushblocks_ppo_{level_name}.zip"

    # Slow down A*/RL moves for nicer demo (one move every N frames)
    ai_step_every = 3
    frame = 0

    running = True
    while running:
        frame += 1

        # --- events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

                if event.key == pygame.K_1:
                    mode = "HUMAN"
                    plan = []
                    plan_failed = False

                if event.key == pygame.K_2:
                    mode = "ASTAR"
                    plan = []
                    plan_failed = False

                if event.key == pygame.K_3:
                    mode = "RL"
                    plan = []
                    plan_failed = False

                if event.key == pygame.K_ESCAPE:
                    state = State(player=level.player_start, blocks=frozenset(level.blocks_start))
                    plan = []
                    plan_failed = False

                # Level switching
                if event.key == pygame.K_F1:
                    level_name = "L1"
                if event.key == pygame.K_F2:
                    level_name = "L2"
                if event.key == pygame.K_F3:
                    level_name = "L3"

                if event.key in (pygame.K_F1, pygame.K_F2, pygame.K_F3):
                    level = parse_level(LEVELS[level_name])
                    state = State(player=level.player_start, blocks=frozenset(level.blocks_start))
                    screen = make_window(level)
                    plan = []
                    plan_failed = False
                    rl_model = None
                    rl_model_name = f"pushblocks_ppo_{level_name}.zip"

        # --- decide action ---
        action = None

        if not is_solved(state.blocks, level.goals):
            if mode == "HUMAN":
                action = get_human_action()

            elif mode == "ASTAR":
                if frame % ai_step_every == 0:
                    if not plan and not plan_failed:
                        plan = astar_plan(level, state) or []
                        if not plan:
                            plan_failed = True
                    if plan:
                        action = plan.pop(0)

            elif mode == "RL":
                if frame % ai_step_every == 0:
                    if rl_model is None:
                        try:
                            rl_model = PPO.load(rl_model_name)
                        except Exception:
                            rl_model = None

                    if rl_model is not None:
                        # IMPORTANT: Must match training input: flattened obs vector
                        obs = build_obs(level, state).flatten()
                        action, _ = rl_model.predict(obs, deterministic=True)
                        action = int(action)

        # --- apply action ---
        if action is not None:
            nxt = apply_action(level, state, action)
            if nxt is not None:
                state = nxt  # illegal actions are ignored

        # --- render ---
        draw(screen, level, state)

        info = f"Level: {level_name} | Mode: {mode} | Solved: {is_solved(state.blocks, level.goals)}"
        if mode == "ASTAR" and plan_failed and not is_solved(state.blocks, level.goals):
            info += " | A*: no plan"
        if mode == "RL" and rl_model is None:
            info += f" | RL model missing: {rl_model_name}"

        text = font.render(info, True, TEXT)
        screen.blit(text, (8, 8))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
