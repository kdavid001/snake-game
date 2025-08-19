import pygame

class Snake:
    def __init__(self, block_size, start_pos=(20, 20)):
        self.block_size = block_size
        self.color = "white"
        self.speed = 200  # pixels per second
        self.direction = "right"
        self.grow = False
        self.accumulated_time = 0

        # Calculate time per step based on speed and block size
        self.time_per_step = 1 / (self.speed / self.block_size)

        # Initialize body with grid-aligned positions
        x, y = start_pos
        self.body = [
            pygame.Rect(x, y, self.block_size, self.block_size),
            pygame.Rect(x - self.block_size, y, self.block_size, self.block_size),
            pygame.Rect(x - 2 * self.block_size, y, self.block_size, self.block_size)
        ]

    def move(self, dt):
        self.accumulated_time += dt
        while self.accumulated_time >= self.time_per_step:
            self.accumulated_time -= self.time_per_step

            # Get the current head
            head = self.body[0].copy()

            # Align to grid to avoid drift
            head.x = (head.x // self.block_size) * self.block_size
            head.y = (head.y // self.block_size) * self.block_size

            # Move in the current direction
            if self.direction == "up":
                head.y -= self.block_size
            elif self.direction == "down":
                head.y += self.block_size
            elif self.direction == "left":
                head.x -= self.block_size
            elif self.direction == "right":
                head.x += self.block_size

            # Insert new head
            self.body.insert(0, head)

            # Remove last segment if not growing
            if not self.grow:
                self.body.pop()
            else:
                self.grow = False

    def add_segment(self):  # Corrected method name
        self.grow = True

    # Rest of the Snake class remains the same
    def change_direction(self, new_direction):
        opposites = {"up": "down", "down": "up",
                     "left": "right", "right": "left"}
        if new_direction != opposites[self.direction]:
            self.direction = new_direction

    def draw(self, surface):
        for segment in self.body:
            pygame.draw.rect(surface, self.color, segment)

