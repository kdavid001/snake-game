import pygame


pygame.init()
width = 800
height = 600
screen = pygame.display.set_mode((width, height))

clock = pygame.time.Clock()
running = True
dt = 0

player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    screen.fill("Blue")

    pygame.draw.circle(screen, "red", player_pos, 100)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        player_pos.y -= 300 * dt
    keys = pygame.key.get_pressed()
    if keys[pygame.K_DOWN]:
        player_pos.y += 300 * dt
    keys = pygame.key.get_pressed()
    if keys[pygame.K_RIGHT]:
        player_pos.x += 300 * dt
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player_pos.x -= 300 * dt


    pygame.display.flip()

    dt = clock.tick(60)/ 1000

pygame.quit()


