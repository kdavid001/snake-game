import pygame


class GameOver:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.font = pygame.font.SysFont('Arial', 20)
        self.text = self.font.render('GAME OVER press Space to restart', True, (255, 0, 0))
        self.rect = self.text.get_rect(center=(width // 2, height / 2))

    def render(self, screen):
        screen.blit(self.text, self.rect)
        pygame.display.flip()
