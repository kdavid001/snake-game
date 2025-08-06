import pygame
import random

BLOCK_SIZE = 20

class Food:
    def __init__(self, screen_width, screen_height):
        self.radius = 8
        self.color = pygame.Color("red")
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.respawn()

    # Respawn food
    def respawn(self):
        cols = self.screen_width // BLOCK_SIZE
        rows = self.screen_height // BLOCK_SIZE
        self.x = random.randint(0, cols - 1) * BLOCK_SIZE
        self.y = random.randint(0, rows - 1) * BLOCK_SIZE

        self.rect = pygame.Rect(
            self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2
        )
    # initially Draw food
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)