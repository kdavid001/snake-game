import pygame


class Scoreboard:
    def __init__(self):
        with open("highscore_for_snake_game.txt") as score:
            self.high_score = score.read()
            self.high_score = int(self.high_score)
        self.score = 0
        self.font = pygame.font.Font(None, 48)
        self.high_score_font = pygame.font.Font(None, 48)
        self.score_text = self.font.render(f"Score: {0}", True, (255, 255, 255))
        self.high_score_text = self.font.render(f"High Score: {0}", True, (255, 255, 255))
        self.score_rect = self.score_text.get_rect(center=(400, 50))
        self.high_score_rect = self.high_score_text.get_rect(center=(600, 50))

    def update(self, screen, fps):
        self.score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.high_score_text = self.font.render(f"High Score: {self.high_score}", True, (255, 255, 255))
        screen.blit(self.score_text, self.score_rect)
        screen.blit(self.high_score_text, self.high_score_rect)

        if fps is not None:
            self.debug_font = pygame.font.Font(None, 28)
            fps_text = self.debug_font.render(f"FPS: {int(fps)}", True, "green")
            screen.blit(fps_text, (10, 10))

    # def result(self, screen):
    #     self.score += 1
    #     self.update(screen)

    def reset(self):

        if self.score > int(self.high_score):
            self.high_score = self.score
            with open("highscore_for_snake_game.txt", 'w') as score:
                self.high_score = str(self.high_score)
                score.write(f"{self.high_score}")
        self.score = 0
