import cv2
from ultralytics import YOLO
import torch
from tkinter import Tk, filedialog
import os
from moviepy import VideoFileClip


print("HIP available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("HIP version:", torch.version.hip)
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

frameCounter = 150

#Just incase of glitch
cv2.destroyAllWindows()

#Change dir so can be ran from folder
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

#modelPath = "/mnt/c/Users/conpr.CONNORSPC/Documents/Siege/model3/Content/train_copy/weights/best.pt" #WSL version
modelPath= "model3\\content\\train_copy\\weights\\best.pt" # Windows Version
model = YOLO(modelPath)

# Step 4: Load the video
Tk().withdraw()
video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4"), ("All files", "*.*")]) 
cap = cv2.VideoCapture(video_path)
audio = VideoFileClip(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties to potentially save the output video
fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#width = 2560
#height = 1440

# Optional: Define an output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')


#vidURL = "/mnt/c/Users/conpr.CONNORSPC/Documents/Siege/output_video_with_boxes.mp4" #WSL version
vidURL = "output_video_with_boxes.mp4" #Windows version
out = cv2.VideoWriter(vidURL, fourcc, fps, (width, height))

# Step 5: Process video frame by frame and detect objects
cv2.namedWindow("YOLO Operator Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Operator Tracker", width, height)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model inference on the frame
    # 'stream=True' can be used in some cases, but 'model(frame)' works for single frames
    results = model(frame)

    # The 'ultralytics' library results object has a built-in plot method
    # which draws the bounding boxes and labels directly onto the frame
    if (results[0].boxes.conf > .45).any() or frameCounter < fps*5:
        frameCounter += 1
        annotated_frame = results[0].plot()
        if (results[0].boxes.conf > .45).any():
            frameCounter = 0
    else:
        annotated_frame = frame

    # Display the frame with bounding boxes
    cv2.imshow("YOLO Operator Tracker", annotated_frame)

    # Optional: Write the frame to the output video
    out.write(annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Step 6: Release resources
cap.release()
# Optional: 
out.release()
cv2.destroyAllWindows()
video_target = VideoFileClip("output_video_with_boxes.mp4")
final = video_target.with_audio(audio.audio)
final.write_videofile("finalOutput.mp4", codec="libx264", audio_codec="aac")
