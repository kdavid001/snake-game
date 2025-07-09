import pygame


class Scoreboard:
    def __init__(self):
        try:
            with open("highscore_for_snake_game.txt", "r") as score_file:
                content = score_file.read().strip()
                self.high_score = int(content) if content else 0
        except FileNotFoundError:
            # if File doesn't exist yet, create it with score 0
            with open("highscore_for_snake_game.txt", "w") as score_file:
                score_file.write("0")
            self.high_score = 0

        self.score = 0
        self.font = pygame.font.Font(None, 48)
        self.high_score_font = pygame.font.Font(None, 48)
        self.score_text = self.font.render(f"Score: {0}", True, (255, 255, 255))
        self.high_score_text = self.font.render(f"High Score: {0}", True, (255, 255, 255))
        self.score_rect = self.score_text.get_rect(center=(400, 50))
        self.high_score_rect = self.high_score_text.get_rect(center=(600, 50))

    # To update the scoreboard
    def update(self, screen, fps):

        self.score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.high_score_text = self.font.render(f"High Score: {self.high_score}", True, (255, 255, 255))
        screen.blit(self.score_text, self.score_rect)
        screen.blit(self.high_score_text, self.high_score_rect)

        # print(f"FPS before updaate: {fps}")
        if fps is not None:
            self.debug_font = pygame.font.Font(None, 28)
            fps_text = self.debug_font.render(f"FPS: {int(fps)}", True, "green")
            screen.blit(fps_text, (10, 10))

    # Reset scoreboard function
    def reset(self):
        if self.score > int(self.high_score):
            self.high_score = self.score
            with open("highscore_for_snake_game.txt", 'w') as score_file:
                score_file.write(str(self.high_score))  # Save as string
        self.score = 0  # Reset the current score

    # score increment
    def increase_score(self):
        self.score += 1
        return self.score

    def get_score(self):
        return self.score

    def get_high_score(self):
        return self.high_score
