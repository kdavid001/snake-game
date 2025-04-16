import pygame
from snake import Snake
from food import Food
from scoreboard import Scoreboard
import math
import random

pygame.font.init()
font = pygame.font.SysFont('Arial', 30)

screen = pygame.display.set_mode((800, 600))
import math

class SnakeGame:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.block_size = 20
        self.snake = Snake(self.block_size, (self.block_size, self.block_size))
        self.food = Food(self.width, self.height)
        self.scoreboard = Scoreboard()  # Scoreboard instance created once
        self.done = False

    def reset(self):
        # Generate random starting positions for snake and food
        start_x = random.randint(self.block_size, self.width - self.block_size)
        start_y = random.randint(self.block_size, self.height - self.block_size)

        # Reset the snake and food
        self.snake = Snake(self.block_size, (start_x, start_y))
        self.food = Food(self.width, self.height)

        # Reset the scoreboard (preserves high score)
        self.scoreboard.reset()

        # Game is no longer done
        self.done = False

        # Return the updated state of the game
        return self.get_state()

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True

        # Handle direction from action
        actions = ['up', 'down', 'left', 'right']
        self.snake.change_direction(actions[action])

        self.snake.move(1 / 15)  # fixed delta time for now
        reward = -1
        head = self.snake.body[0]

        # Calculate Euclidean distance to the food
        # Calculate Euclidean distance to the food
        food_x, food_y = self.food.rect.x, self.food.rect.y
        snake_x, snake_y = head.x, head.y
        distance_to_food = math.sqrt((food_x - snake_x) ** 2 + (food_y - snake_y) ** 2)

        # Initialize previous distance if needed
        if not hasattr(self, 'previous_distance'):
            self.previous_distance = distance_to_food

        # Calculate distance change
        distance_change = self.previous_distance - distance_to_food

        # Reward/penalty based on movement toward/away from food
        if abs(distance_change) > 2:  # Only consider meaningful movements
            if distance_change > 0:  # Moved closer
                reward += distance_change * 0.15
            else:  # Moved away
                reward -= 0.05 * abs(distance_change)  # Smaller penalty for moving away

        # Bonus for being very close (helps with final approach)
        if distance_to_food < 15:
            reward += 0.1 * (15 - distance_to_food)

        self.previous_distance = distance_to_food

        # Check collisions
        if (head.left < 0 or head.right > self.width or
            head.top < 0 or head.bottom > self.height or
            any(head.colliderect(seg) for seg in self.snake.body[3:])):
            self.done = True
            # self.reset()
            reward = -50
            return self.get_state(), reward, self.done

        # To Check food collision
        if head.colliderect(self.food.rect):
            self.scoreboard.increase_score()
            reward = +100
            self.snake.add_segment()
            self.food.respawn()

            # To Ensure it doesn't spawn on the snake
            while any(seg.colliderect(self.food.rect) for seg in self.snake.body):
                self.food.respawn()

        return self.get_state(), reward, self.done

    def render(self, screen, fps):
        screen.fill("black")
        self.food.draw(screen)
        self.snake.draw(screen)
        self.scoreboard.update(screen, fps)  # Update scoreboard with current FPS

    def get_state(self):
        head = self.snake.body[0]
        food = self.food.rect
        return {
            "snake_head": (head.x, head.y),
            "snake_body": [(seg.x, seg.y) for seg in self.snake.body],
            "food": (food.x, food.y),
            "score": self.scoreboard.get_score(),
            "highscore": self.scoreboard.get_high_score()
        }