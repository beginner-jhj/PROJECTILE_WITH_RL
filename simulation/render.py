import pygame

class Renderer:
    def __init__(self, width, height):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height
        self.pygame = pygame

    def draw_circle(self, point, circle_size=5):
        pygame.draw.circle(self.screen, (0, 100, 240), (int(point.x), int(point.y)), circle_size)

    def update(self):
        pygame.display.flip()
        self.clock.tick(60)

    def clear(self):
        self.screen.fill((255, 255, 255))

    def render(self, trajectory):
        self.clear()
        for point in trajectory:
            self.draw_circle(point)
        self.update()