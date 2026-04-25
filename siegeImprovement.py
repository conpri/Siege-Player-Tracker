#Intended to take in a video and display it, asking if the thing on screen is an operator and taking a snapshot of the image and place it in the training folder with labels.
import cv2
from ultralytics import YOLO
import os
from tkinter import messagebox
import tkinter as tk
from tkinter import Tk, filedialog

root = tk.Tk()
root.withdraw()

#Just incase of glitch
cv2.destroyAllWindows()

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

modelPath= "best.pt"
model = YOLO(modelPath)

# Step 4: Load the video
Tk().withdraw()
video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4"), ("All files", "*.*")]) 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties to potentially save the output video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#width = 2560
#height = 1440

# Step 5: Process video frame by frame and detect objects
cv2.namedWindow("YOLO Operator Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Operator Tracker", width, height)
frameDelay = 61
imageCounter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frameDelay <= 60:
        frameDelay += 1
    # Run YOLO model inference on the frame
    results = model(frame)
    frameAnalytics = results[0]

    # The 'ultralytics' library results object has a built-in plot method
    # which draws the bounding boxes and labels directly onto the frame
    annotated_frame = results[0].plot()

    # Display the frame with bounding boxes
    cv2.imshow("YOLO Operator Tracker", annotated_frame)

    if((frameAnalytics.boxes.conf > 0).any() and frameDelay >= 60):
        frameDelay = 0
        answer = messagebox.askyesno("Operator", "Do you want this frame?")
        if answer:
            while os.path.exists(f"Siege Dataset\\train\\images\\ImprovementPic{imageCounter}.jpg") or os.path.exists(f"Siege Dataset\\train\\labels\\ImprovementPic{imageCounter}.txt"):
                imageCounter += 1    
            cv2.imwrite(f"Siege Dataset\\train\\images\\ImprovementPic{imageCounter}.jpg", frame)

            
            boxes = frameAnalytics.boxes.xyxy.cpu().numpy()
            classes = frameAnalytics.boxes.cls.cpu().numpy()
            h, w = frame.shape[:2]
            with open(f"Siege Dataset\\train\\labels\\ImprovementPic{imageCounter}.txt", "x") as f:
                answer = messagebox.askyesno("Operator", "Is this an operator?")
                if answer:
                    for box, cls in zip(boxes, classes):
                        x1, y1, x2, y2 = box

                        # Convert to YOLO format
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h

                        # Write line: class x_center y_center width height
                        f.write(f"{int(cls)} {x_center} {y_center} {width} {height}\n")



    # Optional: Write the frame to the output video
    #out.write(annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 6: Release resources
cap.release()
# Optional: 
#out.release()
cv2.destroyAllWindows()
