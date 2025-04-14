import turtle
from turtle import Turtle

class Gameover(Turtle):
    def __init__(self):
        super().__init__()
        self.color("white")
        self.hideturtle()
        self.penup()
        self.goto(0, 0)
        self.write("GAME OVER", align="center", font=("Courier", 24, "bold"))

        # Optional: close after 3 seconds
        turtle.Screen().ontimer(turtle.bye, 3000)