import turtle
from turtle import *
import random


class Food(Turtle):

    def __init__(self):
        super().__init__()
        self.shape('circle')
        self.penup()
        self.shapesize(stretch_len=0.5, stretch_wid=0.5)
        self.color('red')
        self.speed('fastest')
        self.new_position()
        self.clear()

    def new_position(self):
        random_x = random.randint(-280, 280)
        random_y = random.randint(-280, 280)
        self.setposition(random_x, random_y)


