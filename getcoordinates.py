import cv2
import numpy as np
import os

# Define a callback function to get mouse events
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at (x, y): ({x}, {y})")

# Open the video file
video_path = os.path.join('.', 'data', 'accident.mp4')
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Couldn't open video file.")
    exit()

# Create a window to display the video
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', get_coordinates)

while True:
    ret, frame = cap.read()
    frame_width = int(cap.get(3))  # 3 corresponds to CV_CAP_PROP_FRAME_WIDTH
    frame_height = int(cap.get(4)) 
    print(frame_width, frame_height)
    if not ret:
        print("End of video.")
        break

    roi_vertices = np.array([[899, 100], [1051,141], [960,549], [863, 405]], np.int32)
    # roi_vertices = np.array([[899, 100], [1051,141], [952,553], [857, 404]], np.int32)
    roi_x = 900
    roi_y = 100
    cv2.polylines(frame, [roi_vertices], True, (0, 0, 255), 1)
    # roi_vertices = np.array([[863, 405], [960,549], [606,598], [582, 443]], np.int32)
    # roi_vertices = np.array([[573, 400], [857,404], [952,553], [611, 659]], np.int32)
    roi_vertices = np.array([[580, 440], [857,404], [952,553], [611, 659]], np.int32)
    cv2.polylines(frame, [roi_vertices], True, (0, 0, 255), 1)

    # Display the current frame
    cv2.imshow('Video', frame)

    # Press 'q' to quit
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()