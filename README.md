# Realtime Snake Game with Hand Tracking

## Features Overview
This project implements a real-time Snake game using hand tracking through OpenCV and MediaPipe.

### **1. Import Libraries**
- Import `cv2`, `mediapipe`, `pygame`, `random`, and `numpy`.

### **2. Initialize Hand Tracking**
- Set up MediaPipe Hands with confidence thresholds and max one hand.

### **3. Initialize Pygame**
- Create the game window, set its dimensions, title, and frame rate control.

### **4. Define Colors**
- Set color variables for snake, food, background, and text.

### **5. Set Game Parameters**
- Define speed, swipe threshold, cooldown, and smoothing factor.

### **6. Initialize Score Variables**
- Track current score and high score.

### **7. Set Up Snake and Food**
- Initialize the snake’s position, body, direction, and food position.

### **8. Define Swipe Detection Variables**
- Track last finger position, swipe timing, and smoothed movement.

### **9. Prepare Custom Snake Head Option**
- Add a variable for the snake head image (if customized).

### **10. Manage Game States**
- Create states like `start`, `picture`, `game`, and `game over`.

### **11. Initialize Camera**
- Use OpenCV to capture video from the webcam.

### **12. Define Helper Functions**
- `reset_game()`: Resets snake, food, and score when starting a new game.
- `draw_score()`: Displays the current and high score on screen.
- `crop_head_image()`: Captures and processes a custom snake head from the camera.
- `draw_button()`: Draws interactive buttons for menus.

### **13. Create Main Game Loop**
Controls the game’s behavior based on the current state.

#### **Start State**
- Display start menu and handle button clicks.

#### **Picture State**
- Show webcam feed and let the player capture a custom snake head.

#### **Game State**
- Capture hand position using MediaPipe.
- Calculate smoothed movement for swipe detection.
- Change snake direction based on swipe input.
- Update snake position and body.
- Check for collisions and eating food.
- Spawn power-ups (like invincibility or time-slow).

#### **Game Over State**
- Display "Game Over" screen with options to restart, return to the menu, or quit.

### **14. Cleanup**
- Release camera, close OpenCV windows, and quit Pygame when exiting.
