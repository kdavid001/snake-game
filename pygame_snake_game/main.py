import pygame
from food import Food
from scoreboard import Scoreboard
from snake import Snake

pygame.init()
pygame.display.set_caption("Snake Game")

width, height = 800, 600
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# game objects
food = Food(width, height)
scoreboard = Scoreboard()
snake = Snake(20, (20, 20))

game_active = True
game_over = False
font = pygame.font.Font(None, 28)

while True:
    dt = clock.tick(60) / 1000  # Delta time in seconds

    # FPS display
    fps = clock.get_fps()
    # print(fps)
    # fps_text = font.render(f"FPS: {int(fps)}", True, "green")
    # screen.blit(fps_text, (10, 10))  # Top-left corner

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        if game_over and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Reset game state
                food = Food(width, height)
                scoreboard.reset()
                snake = Snake(20, ((20, 20)))
                game_active = True
                game_over = False

    if game_active:
        # Input handling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            snake.change_direction("up")
        elif keys[pygame.K_DOWN]:
            snake.change_direction("down")
        elif keys[pygame.K_LEFT]:
            snake.change_direction("left")
        elif keys[pygame.K_RIGHT]:
            snake.change_direction("right")

        # Update snake position
        snake.move(dt)

        # Wall collision
        head = snake.body[0]
        if (head.left < 0 or head.right > width or
                head.top < 0 or head.bottom > height):
            game_active = False
            game_over = True


        # Self-collision (critical fix here)
        # In the collision check section:
        # Self-collision check (skip first 3 segments)
        if len(snake.body) > 3:
            head = snake.body[0]
            for segment in snake.body[3:]:
                if head.colliderect(segment):
                    game_active = False
                    game_over = True
                    break

        # In the food collision handling section:
        if snake.body[0].colliderect(food.rect):
            food = Food(width, height)
            # Ensure food doesn't spawn on snake
            while any(segment.colliderect(food.rect) for segment in snake.body):
                food = Food(width, height)
            # scoreboard.result(screen)
            scoreboard.score += 1
            scoreboard.update(screen, fps)
            snake.add_segment()  # Corrected method name

    # Drawing
    screen.fill("black")

    if game_active:
        food.draw(screen)
        snake.draw(screen)
    else:
        # Game over text
        text = font.render("Game Over! Press SPACE to restart", True, "white")
        screen.blit(text, (width // 2 - text.get_width() // 2, height // 2 - text.get_height() // 2))

    scoreboard.update(screen, fps=fps)
    pygame.display.flip()
