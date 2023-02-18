import cv2
import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.hub.load('./yolov5', 'custom', path='./test.pt/weights/best.pt', source='local')

# initialize video capture from default camera
cap = cv2.VideoCapture(0)

# check if camera is opened successfully
if not cap.isOpened():
    print("Unable to open camera")
    exit()

# loop through frames from camera
while True:
    # read frame from camera
    ret, frame = cap.read()

    # check if frame is successfully read
    if not ret:
        print("Unable to read frame from camera")
        break
   
    results = model(frame, size=640)
    
    # Get the bounding box coordinates and class labels for each detected object
    bboxes = results.xyxy[0].cpu().numpy()
    
    for i, bbox in enumerate(bboxes):
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        th = bbox[4]
        if (th > 0.5):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, str("tree"), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # display the frame in a window
    cv2.imshow("Camera", frame)

    # wait for key press and break if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# release the video capture object and close window
cap.release()
cv2.destroyAllWindows()