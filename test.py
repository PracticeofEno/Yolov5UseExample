import ctypes, time
import torch
import win32gui
import win32ui
import win32api
import win32con
from ctypes import windll
from PIL import Image, ImageGrab
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# # Model
model = torch.hub.load('./yolov5', 'custom', path='./test5.pt/weights/best.pt', source='local')

# # Images
# im1 = Image.open('./images/1.png')  # PIL image

# # Inference
# results = model(im1, size=640) # batch of images

# # Results
# results.print()  


# for result in results.pred:
#     x1, y1, x2, y2 = result[:4].int().tolist()
#     print(results.pandas().xyxy[0])
#     # Do something with the coordinates


hwnd = win32gui.FindWindow(None, "마비노기")
print(hwnd)

bbox = win32gui.GetWindowRect(hwnd)
img = ImageGrab.grab(bbox)

results = model(img, size=640)
tmp = results.pandas().xyxy[0]

bbox_coords = tmp.iloc[0].values
print(bbox_coords)
x1 = bbox_coords[0]
y1 = bbox_coords[1]
x2 = bbox_coords[2]
y2 = bbox_coords[3]

print(f'x1 = {x1} y1 = {y1}, x2={x2} y2={y2}')

mouse_x = int(x1) + (int(x2 - x1)) // 2
mouse_y = int(y2) - int((y2 - y1) * 0.17)

ctypes.windll.user32.SetCursorPos(mouse_x, mouse_y)
win32gui.SetForegroundWindow(hwnd)
time.sleep(2)
ctypes.windll.user32.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,mouse_x,mouse_y,0,0)
time.sleep(0.1)
ctypes.windll.user32.mouse_event(win32con.MOUSEEVENTF_LEFTUP,mouse_x,mouse_y,0,0)
print(mouse_x, mouse_y)
results.show()