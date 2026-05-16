import pyautogui
import time

print("Switch to PowerPoint slideshow within 3 seconds...")
time.sleep(3)

pyautogui.press("right")
print("Moved to next slide.")