import random
import numpy as np
from typing import List

# Action constants
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
STAY = 4

ACTIONS = [UP, RIGHT, DOWN, LEFT]

class GridWorld:
    def __init__(self, grid_size=10, n_ghosts=2, pellet_prob=0.2):
        self.grid_size = grid_size
        self.n_ghosts = n_ghosts
        self.pellet_prob = pellet_prob
        self.reset()

    def reset(self):
        n = self.grid_size

        # Board setup
        self.pellets = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                if random.random() < self.pellet_prob:
                    self.pellets[i, j] = 2  # regular pellet

        # Place two boosters (power pellets) in fixed corners
        booster_positions = [(0, 0), (n - 1, n - 1)]
        for bi, bj in booster_positions:
            self.pellets[bi, bj] = 3  # booster pellet

        # Pac-Man starting position
        self.pacman = (n // 2, n // 2)

        # Ghost starting positions
        corners = [(0, n - 1), (n - 1, 0)]
        self.ghosts = [corners[i % len(corners)] for i in range(self.n_ghosts)]

        # Game state trackers
        self.dead_ghosts = {}  # ghost index -> respawn timer
        self.powered_timer = 0
        self.steps = 0

        return self.get_state()

    def in_bounds(self, pos):
        i, j = pos
        n = self.grid_size
        return 0 <= i < n and 0 <= j < n

    def move_pos(self, pos, action):
        i, j = pos
        if action == UP:
            return (i - 1, j)
        if action == DOWN:
            return (i + 1, j)
        if action == LEFT:
            return (i, j - 1)
        if action == RIGHT:
            return (i, j + 1)
        return (i, j)

    def step(self, pacman_action, ghost_actions: List[int] = None):
        """
        Apply Pac-Man action and ghost actions (if provided).
        Returns: state, reward, done, info
        """
        reward = 0.0
        done = False
        info = {}
        n = self.grid_size

        # Step counter
        self.steps += 1

        # Decrease powered timer if active
        if self.powered_timer > 0:
            self.powered_timer -= 1

        # Move Pac-Man
        new_p = self.move_pos(self.pacman, pacman_action)
        if self.in_bounds(new_p):
            self.pacman = new_p

        # Collect pellet or booster
        i, j = self.pacman
        if self.pellets[i, j] == 2:  # regular pellet
            reward += 1.0
            self.pellets[i, j] = 0
        elif self.pellets[i, j] == 3:  # booster
            reward += 5.0
            self.pellets[i, j] = 0
            self.powered_timer = 25  # steps of power mode

        # Move ghosts
        new_ghosts = []
        for idx, g in enumerate(self.ghosts):
            # Handle dead ghosts
            if idx in self.dead_ghosts:
                self.dead_ghosts[idx] -= 1
                if self.dead_ghosts[idx] <= 0:
                    corners = [(0, n - 1), (n - 1, 0)]
                    new_ghosts.append(corners[idx % len(corners)])
                    del self.dead_ghosts[idx]
                else:
                    new_ghosts.append((-1, -1))  # off-board
                continue

            # Move living ghosts
            if ghost_actions:
                a = ghost_actions.pop(0)
                newg = self.move_pos(g, a)
                if self.in_bounds(newg):
                    new_ghosts.append(newg)
                else:
                    new_ghosts.append(g)
            else:
                new_ghosts.append(self._ghost_move_simple(g))

        self.ghosts = new_ghosts

        # Collisions
        for idx, g in enumerate(self.ghosts):
            if g == (-1, -1):
                continue
            if g == self.pacman:
                if self.powered_timer > 0:
                    reward += 50.0  # eat ghost bonus
                    self.dead_ghosts[idx] = 30  # longer respawn
                    self.ghosts[idx] = (-1, -1)
                else:
                    reward -= 20.0
                    done = True
                    info['died'] = True

        # Win condition
        if np.sum(self.pellets > 0) == 0:
            done = True
            info['completed'] = True
            reward += 50.0

        # Timeout
        if self.steps > self.grid_size * self.grid_size * 4:
            done = True
            info['timeout'] = True

        return self.get_state(), reward, done, info

    def _ghost_move_simple(self, ghost_pos):
        gi, gj = ghost_pos
        pi, pj = self.pacman
        di = np.sign(pi - gi)
        dj = np.sign(pj - gj)

        # Prefer larger axis distance
        if abs(pi - gi) >= abs(pj - gj):
            candidate = (gi + di, gj)
            if self.in_bounds(candidate):
                return candidate
            candidate2 = (gi, gj + dj)
            if self.in_bounds(candidate2):
                return candidate2
        else:
            candidate = (gi, gj + dj)
            if self.in_bounds(candidate):
                return candidate
            candidate2 = (gi + di, gj)
            if self.in_bounds(candidate2):
                return candidate2

        return ghost_pos

    def get_state(self):
        """
        State channels:
        0 - regular pellets
        1 - boosters
        2 - pacman
        3 - ghosts
        4 - powered_flag
        """
        n = self.grid_size

        pellets = (self.pellets == 2).astype(np.float32)
        boosters = (self.pellets == 3).astype(np.float32)

        pac = np.zeros((n, n), dtype=np.float32)
        pac[self.pacman] = 1.0

        ghosts = np.zeros((n, n), dtype=np.float32)
        for g in self.ghosts:
            if g != (-1, -1):
                ghosts[g] = 1.0

        powered = np.full((n, n), float(self.powered_timer > 0), dtype=np.float32)

        return np.stack([pellets, boosters, pac, ghosts, powered], axis=0)

    def action_size(self):
        return 4  # up/down/left/right

    def observation_shape(self):
        return (5, self.grid_size, self.grid_size)
