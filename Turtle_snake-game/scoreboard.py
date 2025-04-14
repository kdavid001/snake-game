from turtle import *


class Score(Turtle):
    def __init__(self):
        super().__init__()
        self.speed('fastest')
        with open("highscore_for_snake_game.txt") as score:
            self.highscore = score.read()
            self.highscore = int(self.highscore)
        self.clear()
        self.penup()
        self.setposition(0, 250)
        self.color('white')
        self.hideturtle()
        self.score = 0
        self.update()

    def update(self):
        self.clear()
        self.write(arg=f"Score = {self.score}  High Score = {self.highscore}", move=False, align='center', font=('Arial', 30, 'normal'))

    def results(self):
        self.score += 5
        self.update()

    # trying to give extra points might as well just create another file dedicated for this.
    def bonus_results(self):
        self.score += 15
        self.update()

    def reset(self):
        if self.score > self.highscore:
            self.highscore = self.score
            with open("highscore_for_snake_game.txt", 'w') as score:
                self.highscore = str(self.highscore)
                score.write(f"{self.highscore}")
        self.score = 0
        self.update()

    def game_over(self):
        self.goto(0, 0)
        self.write("Game Over", align="center", font=("Arial", 30, 'normal'))
