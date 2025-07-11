import pygame
import time
class Renderer:
    def __init__(self, width, height):
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height
        self.pygame = pygame

    def draw_circle(self, point, circle_size=5, log=False):
        if log:
            print(f"Drawing circle at: ({point.x}, {point.y}) \n")
        pygame.draw.circle(self.screen, (0, 100, 240), (int(point.x), int(point.y)), circle_size)

    def update(self):
        pygame.display.flip()
        self.clock.tick(60)

    def clear(self):
        self.screen = self.pygame.display.set_mode((self.width, self.height))
        self.screen.fill((255, 255, 255))

    def render(self, trajectory, log=False):
        running = True
        while running:
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    running = False

            self.clear()
            for point in trajectory:
                self.draw_circle(point, log=log)
                self.update()
            
            time.sleep(0.1)
            self.pygame.quit()
            running = False
        self.pygame.init()
