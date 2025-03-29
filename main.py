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


# you can try to add a timer for the bonus to disappear here but note the stuff seems to be lagging check why,
# could be too much memory being used.
def countdown(t):
    while t > 0:
        print(t)
        time.sleep(1)  # Pause execution for 1 second
        t -= 1
    print("Time's up!")


# Set the initial time in seconds
initial_time = 5

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

    # its showing but not recording.
    if a % 5 == 0:
        print(a)
        food_2.showturtle()
        if snake.head.distance(food_2) < 15:
            scoreboard.update()
            snake.extend()
            scoreboard.bonus_results()
            food_2.new_position()
            food_2.clear()
            food_2.hideturtle()
            a += 1
            if a % 5 == 0:
                food_2.hideturtle()

    # you can try to add a timer for the bonus to disappear here but note the stuff seems to be lagging check why,
    # could be too much memory being used.
    # else:
    #   if countdown(initial_time) == 0:
    #   food_2.clear() food_2.hideturtle()

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
