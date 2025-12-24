# =========================
# FILE 1: pushblocks_core.py
# =========================
# Gemeinsame Logik (Grid-basiert) für:
# - A* Solver (klassische KI / Planung)
# - RL (Gym Env nutzt diese Logik)
# - pygame Demo (Rendering ruft apply_action + build_obs)
#
# Level-Format (ASCII):
#   # = Wall
#   . = Floor
#   P = Player start
#   B = Block/Box
#   G = Goal
#
# Ziel:
#   Alle Blöcke stehen auf Goal-Feldern (Sokoban-like).
#
# Actions:
#   0 = left, 1 = right, 2 = up, 3 = down

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, FrozenSet, Dict
import numpy as np
import heapq
import itertools

Pos = Tuple[int, int]  # (x, y) grid cell


LEVELS = {

        "L0": [
            "############",
            "#..........#",
            "#....G.....#",
            "#....B.....#",
            "#....P.....#",
            "#..........#",
            "############",
        ],
    
        "L1": [
        "############",
        "#....G.....#",
        "#....G.....#",
        "#..B.......#",
        "#.....P....#",
        "#..........#",
        "############",
    ],
    "L2": [
        "############",
        "#....G.....#",
        "#....G.....#",
        "#..B...B...#",
        "#.....P....#",
        "#..........#",
        "############",
    ],
    "L3": [
        "################",
        "#..G......G....#",
        "#..G......G....#",
        "#......B.......#",
        "#..B......P....#",
        "#..............#",
        "################",
    ],
}

# grid actions (dx, dy)
ACTIONS_GRID: List[Pos] = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # L, R, U, D


@dataclass
class Level:
    w: int
    h: int
    walls: Set[Pos]
    goals: Set[Pos]
    player_start: Pos
    blocks_start: Set[Pos]


def parse_level(map_lines: List[str]) -> Level:
    h = len(map_lines)
    w = len(map_lines[0])
    walls: Set[Pos] = set()
    goals: Set[Pos] = set()
    blocks: Set[Pos] = set()
    player: Optional[Pos] = None

    for y, row in enumerate(map_lines):
        if len(row) != w:
            raise ValueError("All level rows must have same length.")
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "G":
                goals.add((x, y))
            elif ch == "B":
                blocks.add((x, y))
            elif ch == "P":
                player = (x, y)
            # '.' (or other) = floor

    if player is None:
        raise ValueError("Level must contain a Player start 'P'.")

    return Level(w=w, h=h, walls=walls, goals=goals, player_start=player, blocks_start=blocks)


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass(frozen=True)
class State:
    player: Pos
    blocks: FrozenSet[Pos]


def is_solved(blocks: FrozenSet[Pos], goals: Set[Pos]) -> bool:
    return all(b in goals for b in blocks)


def apply_action(level: Level, state: State, action_id: int) -> Optional[State]:
    """
    Apply one action:
    - move player if free
    - push a block if next cell has a block and the cell behind is free
    - walls and blocks block movement
    Returns new state or None if illegal.
    """
    dx, dy = ACTIONS_GRID[action_id]
    px, py = state.player
    nx, ny = px + dx, py + dy
    np_ = (nx, ny)

    if np_ in level.walls:
        return None

    blocks = set(state.blocks)

    if np_ in blocks:
        bp = (nx + dx, ny + dy)
        if bp in level.walls or bp in blocks:
            return None
        blocks.remove(np_)
        blocks.add(bp)
        return State(player=np_, blocks=frozenset(blocks))

    return State(player=np_, blocks=state.blocks)


def build_obs(level: Level, state: State) -> np.ndarray:
    """
    Observation for RL:
      4 channels: walls, goals, blocks, player
      shape (4, H, W) float32 with values 0/1
    """
    obs = np.zeros((4, level.h, level.w), dtype=np.float32)

    for (x, y) in level.walls:
        obs[0, y, x] = 1.0
    for (x, y) in level.goals:
        obs[1, y, x] = 1.0
    for (x, y) in state.blocks:
        obs[2, y, x] = 1.0
    px, py = state.player
    obs[3, py, px] = 1.0

    return obs


# ----------------------------
# A* solver
# ----------------------------
def deadlock_corner(level: Level, block: Pos) -> bool:
    """
    Minimal deadlock:
    Block in a wall-corner and not on goal => impossible (prune).
    """
    if block in level.goals:
        return False
    x, y = block
    left = (x - 1, y) in level.walls
    right = (x + 1, y) in level.walls
    up = (x, y - 1) in level.walls
    down = (x, y + 1) in level.walls
    return (left and up) or (left and down) or (right and up) or (right and down)


def heuristic(level: Level, state: State) -> int:
    """
    Sum of min distances from each block to any goal.
    """
    goals = list(level.goals)
    h = 0
    for b in state.blocks:
        h += min(manhattan(b, g) for g in goals)
    return h


def astar_plan(level: Level, start: State, max_expansions: int = 200_000) -> Optional[List[int]]:
    """
    Returns list of action ids (0..3) or None if no plan.
    """
    if any(deadlock_corner(level, b) for b in start.blocks):
        return None

    open_heap: List[Tuple[int, int, int, State]] = []
    g: Dict[State, int] = {start: 0}
    counter = itertools.count()

    f0 = heuristic(level, start)
    heapq.heappush(open_heap, (f0, 0, next(counter), start))

    came_from: Dict[State, State] = {}
    came_action: Dict[State, int] = {}

    expansions = 0
    while open_heap:
        _, cur_g, _, cur = heapq.heappop(open_heap)
        expansions += 1
        if expansions > max_expansions:
            return None

        # skip outdated entries
        if cur_g != g.get(cur, 10**9):
            continue

        if is_solved(cur.blocks, level.goals):
            actions: List[int] = []
            s = cur
            while s in came_from:
                actions.append(came_action[s])
                s = came_from[s]
            actions.reverse()
            return actions

        for ai in range(4):
            nxt = apply_action(level, cur, ai)
            if nxt is None:
                continue
            if any(deadlock_corner(level, b) for b in nxt.blocks):
                continue

            ng = cur_g + 1
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                heapq.heappush(open_heap, (ng + heuristic(level, nxt), ng, next(counter), nxt))
                came_from[nxt] = cur
                came_action[nxt] = ai

    return None
