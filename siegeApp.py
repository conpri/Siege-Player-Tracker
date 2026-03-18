import cv2
from ultralytics import YOLO
import torch

#Checks for GPU
print("HIP available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("HIP version:", torch.version.hip)
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

#Just incase of glitch
cv2.destroyAllWindows()

modelPath = "/mnt/c/Users/conpr.CONNORSPC/Documents/Siege/Content/train_copy/weights/best.pt" #WSL version
#modelPath= "C:\\Users\\conpr.CONNORSPC\\Documents\\Siege\\content\\train_copy\\weights\\best.pt" # Windows Version
model = YOLO(modelPath)

video_path = "/mnt/c/Users/conpr.CONNORSPC/Documents/Siege/Video.mp4" #WSL version
#video_path = "C:\\Users\\conpr.CONNORSPC\\Documents\\Siege\\Video.mp4" #Windows Version
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties to save the output video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#width = 2560
#height = 1440

#Output video writer so I can see my work
fourcc = cv2.VideoWriter_fourcc(*'mp4v')


vidURL = "/mnt/c/Users/conpr.CONNORSPC/Documents/Siege/output_video_with_boxes.mp4" #WSL version
#vidURL = "C:\\Users\\conpr.CONNORSPC\\Documents\\Siege\\output_video_with_boxes.mp4" #Windows version
out = cv2.VideoWriter(vidURL, fourcc, fps, (width, height))

#Stuff to be able to resize window
cv2.namedWindow("YOLO Siege Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Siege Tracker", width, height)
# Video goes frame by frame to check for operators
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model inference on the frame
    results = model(frame)

    # Draws bounding boxes onto each frame that it detects something
    annotated_frame = results[0].plot()

    # Display the frame with bounding boxes
    cv2.imshow("YOLO Siege Tracker", annotated_frame)

    # Optional: Write the frame to the output video
    out.write(annotated_frame)

    # Breaks loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources so things dont break
cap.release()
out.release()
cv2.destroyAllWindows()
