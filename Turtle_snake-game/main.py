import random

from turtle import *
from snake import Snake
import time
from food import Food
from scoreboard import Score
from pwg import Gameover

screen = Screen()
screen.title('welcome to snake Game')
screen.setup(width=600, height=600)
screen.bgcolor("black")
screen.tracer(0)

food_2 = Food()
food_2.color('white')
food_2.shapesize(1, 1)
snake = Snake()
food = Food()
scoreboard = Score()
food_2.hideturtle()

# not useful for now
# def countdown(t):
#     while t > 0:
#         print(t)
#         time.sleep(1)  # Pause execution for 1 second
#         t -= 1
#     print("Time's up!")

def hide_bonus_food():
    food_2.hideturtle()
    global bonus_active
    bonus_active = False


# Set the initial time in seconds
initial_time = 5
bonus_active = False
bonus_timer = 5000  # 5 seconds in milliseconds
a = 1
game_is_on = True
while game_is_on:
    screen.update()
    time.sleep(0.1)

    # detection with food
    if snake.head.distance(food) < 15:
        scoreboard.update()
        snake.extend()
        scoreboard.results()
        food.new_position()
        a += 1

    # detection with bonus point/food
    if a % 5 == 0 and not bonus_active:
        bonus_active = True
        food_2.new_position()
        food_2.showturtle()
        screen.ontimer(lambda: hide_bonus_food(), bonus_timer)

    if bonus_active and snake.head.distance(food_2) < 25:
        scoreboard.update()
        snake.extend()
        snake.extend()
        scoreboard.bonus_results()
        food_2.hideturtle()
        bonus_active = False
        a += 1

    # detection with wall
    if snake.head.xcor() > 280 or snake.head.xcor() < -280 or snake.head.ycor() > 280 or snake.head.ycor() < - 280:
        user_pick = screen.textinput("TRY AGAIN.", "would you like to try again? y/n").lower()
        if user_pick == "y":
            scoreboard.reset()
            snake.reset()
        if user_pick == "n":
            game_is_on = False
            pwg = Gameover()

    # detection with tail
    for _ in snake.segments:
        if _ == snake.head:
            pass
        elif snake.head.distance(_) < 15:
            scoreboard.reset()
            snake.reset()

    screen.listen()
    snake.move()
    screen.onkey(snake.up, "Up")
    screen.onkey(snake.down, "Down")
    screen.onkey(snake.left, "Left")
    screen.onkey(snake.right, "Right")

screen.exitonclick()
