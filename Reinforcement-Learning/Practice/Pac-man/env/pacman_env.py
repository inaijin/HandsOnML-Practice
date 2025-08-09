from game.game import GridWorld
from game.renderer import Renderer

# Just Like Gymnasium !
class PacmanEnv:
    def __init__(self, grid_size=10, n_ghosts=2):
        self.grid = GridWorld(grid_size=grid_size, n_ghosts=n_ghosts)
        self.renderer = Renderer(grid_size=grid_size)
        self._last_state = None

    def reset(self):
        state = self.grid.reset()
        self._last_state = state
        return state

    def step(self, action):
        next_state, reward, done, info = self.grid.step(action)
        self._last_state = next_state
        return next_state, reward, done, info

    def render(self, delay=50):
        if self._last_state is not None:
            self.renderer.render(self._last_state, delay=delay)

    def observation_space_shape(self):
        return self.grid.observation_shape()

    def action_size(self):
        return self.grid.action_size()
