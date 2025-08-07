import pygame
from snake import Snake
from food import Food
from scoreboard import Scoreboard
import math
import random

pygame.font.init()
font = pygame.font.SysFont('Arial', 30)

screen = pygame.display.set_mode((800, 600))


class SnakeGame:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.block_size = 20
        self.snake = Snake(self.block_size, (self.block_size, self.block_size))
        self.food = Food(self.width, self.height)
        self.scoreboard = Scoreboard()
        self.done = False
        # trying to prevent looping
        self.steps_since_last_food = 0

    def reset(self):
        start_x = random.randint(0, (self.width // self.block_size) - 1) * self.block_size
        start_y = random.randint(0, (self.height // self.block_size) - 1) * self.block_size
        self.previous_distance = None
        # Reset the snake and food
        self.snake = Snake(self.block_size, (start_x, start_y))
        self.food = Food(self.width, self.height)

        # Reset the scoreboard
        self.scoreboard.reset()

        # Game is no longer done
        self.done = False
        self.steps_since_last_food = 0
        return self.get_state()

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True

        # Handle direction from action
        actions = ['up', 'down', 'left', 'right']
        self.snake.change_direction(actions[action])

        self.snake.move(1 / 15)  # fixed delta time for now
        self.steps_since_last_food += 1
        reward = -1
        head = self.snake.body[0]

        # to Calculate Euclidean distance to the food
        food_x, food_y = self.food.rect.x, self.food.rect.y
        snake_x, snake_y = head.x, head.y
        distance_to_food = math.sqrt((food_x - snake_x) ** 2 + (food_y - snake_y) ** 2)

        if self.previous_distance is None:
            self.previous_distance = distance_to_food

        # Calculate distance change
        distance_change = self.previous_distance - distance_to_food

        # Reward/penalty based on movement toward/away from food
        if abs(distance_change) > 2:  # Only considers meaningful movements
            if distance_change > 0:  # Moved closer
                reward += distance_change * 0.15
            else:  # Moved away
                reward -= 0.05 * abs(distance_change)  # Smaller penalty for moving away

        # Bonus for being very close (helps with final approach)
        if distance_to_food < 15:
            decay_factor = max(0.0, 1.0 - (self.steps_since_last_food / 200))
            reward_bonus = 0.05 * max(0, (15 - math.floor(distance_to_food))) * decay_factor
            reward += reward_bonus
        #   reward += 0.1 * (15 - distance_to_food)

        self.previous_distance = distance_to_food

        # Check collisions
        if (head.left < 0 or head.right > self.width or
                head.top < 0 or head.bottom > self.height or
                any(head.colliderect(seg) for seg in self.snake.body[3:])):
            self.done = True
            # self.reset()
            reward = -50
            return self.get_state(), reward, self.done

        # just added this,After collision checks and food checks
        # Dynamic step limit to allow more time as snake grows
        base_steps = 100
        steps_per_segment = 10
        max_steps = base_steps + len(self.snake.body) * steps_per_segment

        if self.steps_since_last_food > max_steps:
            reward -= 10
            self.done = True
            return self.get_state(), reward, self.done

        # To Check food collision
        # Always keep this reward last -> to show the bot the main goal
        if head.colliderect(self.food.rect):
            self.scoreboard.increase_score()
            reward = +100
            self.snake.add_segment()
            self.food.respawn()
            # added this line
            self.steps_since_last_food = 0

            # To Ensure the food doesn't spawn on the snake
            attempts = 0
            while any(seg.colliderect(self.food.rect) for seg in self.snake.body) and attempts < 100:
                self.food.respawn()
                attempts += 1

        return self.get_state(), reward, self.done

    def render(self, screen, fps):
        screen.fill("black")
        self.food.draw(screen)
        self.snake.draw(screen)
        self.scoreboard.update(screen, fps)

    def get_state(self):
        head = self.snake.body[0]
        food = self.food.rect
        return {
            "snake_head": (head.x, head.y),
            "snake_body": [(seg.x, seg.y) for seg in self.snake.body],
            "food": (food.x, food.y),
            "direction": self.snake.direction,
            "score": self.scoreboard.get_score(),
            "highscore": self.scoreboard.get_high_score()
        }
