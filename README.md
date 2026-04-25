# Siege-Player-Tracker
## Description
A computer vision model to track Rainbow 6 Siege operators. Target Audience for current design is analyzing film, whether an operator swung and you didnt see it at the time or it was just almost unnoticable based off the angle they were holding.

## Techstack
### Software & Tools
Jupyter Notebook (Python) — Used during the early exploration phase to visualize images and inspect data

Google Colab (Python) — Environment used for training the YOLO model with GPU acceleration

VS Code (Python) — Main development workspace where all post‑training scripts were written and refined

LabelImg — Annotation tool used to label images and generate YOLO‑formatted dataset files

### Frameworks
OpenCV (cv2) — Handles frame‑by‑frame video extraction, visualization, and writing processed video outputs

Ultralytics YOLO — Object detection model used for training and inference

Tkinter — Provides a simple GUI file‑selection dialog for choosing input videos

MoviePy — Merges the original audio track back into the final processed video

## Installs Required
Ultralytics - py -m pip install ultralytics

OpenCV - py -m pip install opencv-python

MoviePy - py -m pip install moviepy

NumPy - py -m pip install numpy

PyTorcy - py -m pip install torch torchvision torchaudio

## How to use program
1. Install best.pt and siegeApp.py
2. Have files in same directory
3. Have a clip ready to place into program (Doesn't need to be in the same directory)
4. Do py (Full path of program) in Command Prompt
5. Select video clip once File Selector pops up
6. Finished video clip will be in the same directory as your program once it's done running.
