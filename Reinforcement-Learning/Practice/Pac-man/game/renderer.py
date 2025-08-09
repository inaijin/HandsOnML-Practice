import pygame

CELL_SIZE = 32
MARGIN = 2

class Renderer:
    def __init__(self, grid_size=10, title="PacMan RL"):
        pygame.init()
        self.grid_size = grid_size
        w = grid_size * CELL_SIZE
        h = grid_size * CELL_SIZE
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

    def _is_booster(self, state, i, j):
        # boosters are now in channel 1
        return state[1, i, j] > 0.5

    def render(self, state, delay=50):
        """
        Expects state with channels:
        0 - pellets (regular)
        1 - boosters (power pellets)
        2 - pacman
        3 - ghosts
        4 - powered flag (global)
        """
        pellets = state[0]
        boosters = state[1]
        pac = state[2]
        ghosts = state[3]
        powered_flag = bool(state[4, 0, 0] > 0.5)

        n = self.grid_size
        self.screen.fill((0, 0, 0))

        for i in range(n):
            for j in range(n):
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE - MARGIN, CELL_SIZE - MARGIN)

                # draw regular pellet
                if pellets[i, j] > 0.5:
                    pygame.draw.circle(self.screen, (200, 200, 0), rect.center, 4)

                # draw booster (power pellet)
                if boosters[i, j] > 0.5:
                    pygame.draw.circle(self.screen, (0, 0, 255), rect.center, CELL_SIZE // 4)

                # draw pacman (blue when powered)
                if pac[i, j] > 0.5:
                    color = (0, 0, 255) if powered_flag else (255, 255, 0)
                    pygame.draw.circle(self.screen, color, rect.center, CELL_SIZE // 3)

                # draw ghost (blue when edible)
                if ghosts[i, j] > 0.5:
                    ghost_color = (0, 0, 255) if powered_flag else (255, 0, 0)
                    pygame.draw.rect(self.screen, ghost_color, rect.inflate(-CELL_SIZE // 3, -CELL_SIZE // 3))

        pygame.display.flip()

        # Keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # preserve your original timing behavior
        self.clock.tick(1000 // max(1, delay))
