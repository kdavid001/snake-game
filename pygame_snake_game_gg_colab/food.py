import pygame
import random

class Food:
    def __init__(self, screen_width, screen_height):
        self.radius = 8
        self.color = pygame.Color("red")
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.respawn()

    # Respawn food
    def respawn(self):
        self.x = random.randint(self.radius, self.screen_width - self.radius)
        self.y = random.randint(self.radius, self.screen_height - self.radius)
        self.rect = pygame.Rect(
            self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2
        )
    # initially Draw food
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)