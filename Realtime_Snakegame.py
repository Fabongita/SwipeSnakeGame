import cv2
import mediapipe as mp
import pygame
import random
import numpy as np

# ------------------------
# Hand Tracking Setup
# ------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# ------------------------
# Pygame Setup
# ------------------------
pygame.init()
WIDTH, HEIGHT = 640, 480
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Swipe Snake Game")
font = pygame.font.SysFont(None, 30)
clock = pygame.time.Clock()

# ------------------------
# Colors
# ------------------------
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED   = (255, 0, 0)
BLACK = (0, 0, 0)
GRAY  = (200, 200, 200)
DARK_GRAY = (150, 150, 150)
# Power-up colors:
BLUE = (0, 0, 255)       # bonus
YELLOW = (255, 255, 0)   # slow
PURPLE = (128, 0, 128)   # invincible

# ------------------------
# Game Settings
# ------------------------
speed = 10                   # Base snake speed (normal_speed)
normal_speed = speed         # Store normal speed for restoration
SWIPE_THRESHOLD = 15         # Minimum smoothed movement to trigger a swipe
SWIPE_COOLDOWN = 250         # Milliseconds between swipes
SMOOTHING_FACTOR = 0.5       # Exponential smoothing factor (0–1)

# Scores
score = 0
high_score = 0

# Snake Variables
snake_pos  = [100, 50]
snake_body = [[100, 50], [90, 50], [80, 50]]
snake_direction = "RIGHT"
food_pos   = [random.randrange(1, WIDTH // 10) * 10,
              random.randrange(1, HEIGHT // 10) * 10]
food_spawn = True

# Swipe detection variables
last_finger_x = None
last_finger_y = None
last_swipe_time = 0
smoothed_dx = 0
smoothed_dy = 0

# Optional snake head image (None by default)
snake_head_img = None

# New: Power-up variables
powerup_item = None  # will store dict {"type": type, "pos": (x, y), "spawn_time": t}
active_powerup = None  # can be "slow" or "invincible"; "bonus" is applied instantly
active_powerup_end_time = 0

# ------------------------
# Game States
# "start", "picture", "game", "gameover"
# ------------------------
state = "start"

# Initialize Camera
cap = cv2.VideoCapture(0)

# ------------------------
# Helper Functions
# ------------------------
def reset_game():
    """Reset snake positions, spawn new food, and reset score."""
    global snake_pos, snake_body, snake_direction, food_pos, food_spawn, score
    snake_pos  = [100, 50]
    snake_body = [[100, 50], [90, 50], [80, 50]]
    snake_direction = "RIGHT"
    food_pos = [random.randrange(1, WIDTH // 10) * 10,
                random.randrange(1, HEIGHT // 10) * 10]
    food_spawn = True
    score = 0

def draw_score():
    """Display current and high scores."""
    score_text = font.render(f"Score: {score}  High Score: {high_score}", True, BLACK)
    win.blit(score_text, (10, 10))

def crop_head_image(frame, center, radius):
    """
    Crop a circular region from the given frame.
    `frame` is assumed to be in BGR format.
    Returns a pygame.Surface of size (20, 20) with per-pixel alpha.
    """
    cx, cy = center
    # Calculate ROI coordinates (ensure they lie within frame bounds)
    x1 = max(cx - radius, 0)
    y1 = max(cy - radius, 0)
    x2 = min(cx + radius, frame.shape[1])
    y2 = min(cy + radius, frame.shape[0])
    
    roi = frame[y1:y2, x1:x2].copy()  # Crop ROI
    h, w = roi.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), min(w, h)//2, (255), -1)
    roi_bgra = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
    roi_bgra[:, :, 3] = mask
    roi_rgba = cv2.cvtColor(roi_bgra, cv2.COLOR_BGRA2RGBA)
    head_surf = pygame.image.frombuffer(roi_rgba.tobytes(), (w, h), "RGBA")
    head_surf = pygame.transform.smoothscale(head_surf, (20, 20))
    return head_surf

def draw_button(surface, text, x, y, w, h, color, hover_color):
    """
    Draw a button (rectangle + text) on 'surface'.
    Returns a pygame.Rect for collision detection.
    """
    mouse = pygame.mouse.get_pos()
    # Draw button with hover effect
    if x < mouse[0] < x+w and y < mouse[1] < y+h:
        pygame.draw.rect(surface, hover_color, (x, y, w, h))
    else:
        pygame.draw.rect(surface, color, (x, y, w, h))
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=(x + w//2, y + h//2))
    surface.blit(text_surface, text_rect)
    return pygame.Rect(x, y, w, h)

def spawn_powerup():
    """Randomly spawn a power-up if none exists."""
    global powerup_item
    # Spawn with a low probability each frame if none exists
    if powerup_item is None and random.random() < 0.005:
        powerup_type = random.choice(["bonus", "slow", "invincible"])
        x = random.randrange(0, WIDTH // 10) * 10
        y = random.randrange(0, HEIGHT // 10) * 10
        powerup_item = {"type": powerup_type, "pos": (x, y), "spawn_time": pygame.time.get_ticks()}

# ------------------------
# Main Loop
# ------------------------
running = True
while running:
    # Process global events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if state == "start":
        # --- Start Screen ---
        win.fill(WHITE)
        title_text = font.render("Swipe Snake Game", True, BLACK)
        win.blit(title_text, (WIDTH//2 - title_text.get_width()//2, HEIGHT//4))
        btn_start_rect = draw_button(win, "Start Game", WIDTH//2 - 60, HEIGHT//2, 120, 40, GRAY, DARK_GRAY)
        btn_custom_rect = draw_button(win, "Customize Snake Head", WIDTH//2 - 100, HEIGHT//2 + 60, 200, 40, GRAY, DARK_GRAY)
        pygame.display.flip()
        clock.tick(30)
        mouse_pos = pygame.mouse.get_pos()
        mouse_click = pygame.mouse.get_pressed()
        if mouse_click[0] == 1:
            if btn_start_rect.collidepoint(mouse_pos):
                state = "game"
                reset_game()
                last_finger_x = None
                last_finger_y = None
                last_swipe_time = 0
                smoothed_dx = 0
                smoothed_dy = 0
                # Reset power-up variables
                powerup_item = None
                active_powerup = None
                active_powerup_end_time = 0
                speed = normal_speed
            elif btn_custom_rect.collidepoint(mouse_pos):
                state = "picture"
        continue

    elif state == "picture":
        # --- Picture Screen: Capture custom snake head ---
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.image.frombuffer(frame_rgb.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
        frame_surface = pygame.transform.scale(frame_surface, (WIDTH, HEIGHT))
        win.blit(frame_surface, (0, 0))
        center = (WIDTH//2, HEIGHT//2)
        radius = 100
        pygame.draw.circle(win, RED, center, radius, 3)
        btn_capture_rect = draw_button(win, "Capture", WIDTH//2 - 60, HEIGHT - 120, 120, 40, GRAY, DARK_GRAY)
        btn_back_rect = draw_button(win, "Back", WIDTH//2 - 40, HEIGHT - 60, 80, 40, GRAY, DARK_GRAY)
        pygame.display.flip()
        clock.tick(30)
        mouse_pos = pygame.mouse.get_pos()
        mouse_click = pygame.mouse.get_pressed()
        if mouse_click[0] == 1:
            if btn_capture_rect.collidepoint(mouse_pos):
                snake_head_img = crop_head_image(frame, center, radius)
                state = "start"
            elif btn_back_rect.collidepoint(mouse_pos):
                state = "start"
        continue

    elif state == "game":
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[8]
                finger_x = int(index_tip.x * WIDTH)
                finger_y = int(index_tip.y * HEIGHT)
                cv2.circle(frame, (finger_x, finger_y), 10, (0, 0, 255), -1)
                if last_finger_x is None or last_finger_y is None:
                    last_finger_x, last_finger_y = finger_x, finger_y
                raw_dx = finger_x - last_finger_x
                raw_dy = finger_y - last_finger_y
                smoothed_dx = SMOOTHING_FACTOR * raw_dx + (1 - SMOOTHING_FACTOR) * smoothed_dx
                smoothed_dy = SMOOTHING_FACTOR * raw_dy + (1 - SMOOTHING_FACTOR) * smoothed_dy
                current_time = pygame.time.get_ticks()
                if (abs(smoothed_dx) > SWIPE_THRESHOLD or abs(smoothed_dy) > SWIPE_THRESHOLD) and \
                   (current_time - last_swipe_time > SWIPE_COOLDOWN):
                    if abs(smoothed_dx) > abs(smoothed_dy):
                        if smoothed_dx < 0 and snake_direction != "RIGHT":
                            snake_direction = "LEFT"
                        elif smoothed_dx > 0 and snake_direction != "LEFT":
                            snake_direction = "RIGHT"
                    else:
                        if smoothed_dy < 0 and snake_direction != "DOWN":
                            snake_direction = "UP"
                        elif smoothed_dy > 0 and snake_direction != "UP":
                            snake_direction = "DOWN"
                    last_swipe_time = current_time
                last_finger_x, last_finger_y = finger_x, finger_y

        # Spawn a power-up if none exists
        spawn_powerup()

        # Check if snake collects power-up
        if powerup_item is not None:
            p_x, p_y = powerup_item["pos"]
            if snake_pos[0] == p_x and snake_pos[1] == p_y:
                if powerup_item["type"] == "bonus":
                    score += 5
                elif powerup_item["type"] == "slow":
                    active_powerup = "slow"
                    active_powerup_end_time = pygame.time.get_ticks() + 5000  # 5 sec
                    speed = max(5, normal_speed // 2)
                elif powerup_item["type"] == "invincible":
                    active_powerup = "invincible"
                    active_powerup_end_time = pygame.time.get_ticks() + 5000  # 5 sec
                powerup_item = None

        # Check active power-up effect expiration
        if active_powerup is not None and pygame.time.get_ticks() > active_powerup_end_time:
            if active_powerup == "slow":
                speed = normal_speed
            active_powerup = None

        # Move snake based on current direction
        if snake_direction == "UP":
            snake_pos[1] -= 10
        elif snake_direction == "DOWN":
            snake_pos[1] += 10
        elif snake_direction == "LEFT":
            snake_pos[0] -= 10
        elif snake_direction == "RIGHT":
            snake_pos[0] += 10

        snake_body.insert(0, list(snake_pos))
        if snake_pos == food_pos:
            score += 1
            if score > high_score:
                high_score = score
            food_spawn = False
        else:
            snake_body.pop()

        if not food_spawn:
            food_pos = [random.randrange(1, WIDTH // 10) * 10,
                        random.randrange(1, HEIGHT // 10) * 10]
        food_spawn = True

        # Collision checks: always check walls; check self-collision only if not invincible
        if snake_pos[0] < 0 or snake_pos[0] > WIDTH - 10 or snake_pos[1] < 0 or snake_pos[1] > HEIGHT - 10:
            state = "gameover"
        if active_powerup != "invincible":
            for block in snake_body[1:]:
                if snake_pos == block:
                    state = "gameover"

        win.fill(WHITE)
        # Draw snake (use custom head image if available)
        for i, pos in enumerate(snake_body):
            if i == 0 and snake_head_img is not None:
                win.blit(snake_head_img, (pos[0], pos[1]))
            else:
                pygame.draw.rect(win, GREEN, pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(win, RED, pygame.Rect(food_pos[0], food_pos[1], 10, 10))
        # Draw power-up if it exists
        if powerup_item is not None:
            p_type = powerup_item["type"]
            p_x, p_y = powerup_item["pos"]
            if p_type == "bonus":
                color = BLUE
            elif p_type == "slow":
                color = YELLOW
            elif p_type == "invincible":
                color = PURPLE
            pygame.draw.rect(win, color, pygame.Rect(p_x, p_y, 10, 10))
        draw_score()
        pygame.display.flip()
        clock.tick(speed)
        cv2.imshow("Finger Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
        continue

    elif state == "gameover":
        win.fill(WHITE)
        gameover_text = font.render("Game Over!", True, BLACK)
        win.blit(gameover_text, (WIDTH//2 - gameover_text.get_width()//2, HEIGHT//4))
        btn_play_rect = draw_button(win, "Play Again", WIDTH//2 - 60, HEIGHT//2, 120, 40, GRAY, DARK_GRAY)
        btn_menu_rect = draw_button(win, "Main Menu", WIDTH//2 - 60, HEIGHT//2 + 60, 120, 40, GRAY, DARK_GRAY)
        btn_quit_rect = draw_button(win, "Quit", WIDTH//2 - 30, HEIGHT//2 + 120, 60, 40, GRAY, DARK_GRAY)
        pygame.display.flip()
        clock.tick(30)
        mouse_pos = pygame.mouse.get_pos()
        mouse_click = pygame.mouse.get_pressed()
        if mouse_click[0] == 1:
            if btn_play_rect.collidepoint(mouse_pos):
                reset_game()
                state = "game"
                last_finger_x = None
                last_finger_y = None
                last_swipe_time = 0
                smoothed_dx = 0
                smoothed_dy = 0
                powerup_item = None
                active_powerup = None
                active_powerup_end_time = 0
                speed = normal_speed
            elif btn_menu_rect.collidepoint(mouse_pos):
                state = "start"
            elif btn_quit_rect.collidepoint(mouse_pos):
                running = False
        continue

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
