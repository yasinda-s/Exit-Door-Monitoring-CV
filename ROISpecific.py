#sets a virtually accurate ROI for the exit area AND door and counts the number of people within the region
import os
import cv2
from ultralytics import YOLO
from tracker import Tracker
import numpy as np

#setup video, cap_out, yolo and deepsort
video_path = os.path.join('.', 'data', 'CCTV_in.mp4')
video_out_path = os.path.join('.', 'CCTV_out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
                          (1280, 720))

model = YOLO("yolov8n.pt")
# tracker = Tracker()

detection_threshold = 0.3

#function for exit door
def exitDoorROI():
    roi_vertices = np.array([[710, 31], [810,36], [790,220], [700, 215]], np.int32)
    roi_x = 700
    roi_y = 36
    cv2.polylines(frame, [roi_vertices], True, (0, 0, 255), 1)
    cv2.putText(frame, "Exit Door", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

#function for exit area ROI
def exitAreaROI(frame):
    roi_vertices = np.array([[520, 340], [640,220], [800,230], [783, 380]], np.int32)
    cv2.fillPoly(frame, [roi_vertices], (0, 165, 255))
    return roi_vertices

#function for detecting people (if feet within exit area)
def peopleDetectionYOLO(frame, roiVertices):
    results = model(frame)
    detections = []
    human_exit_count = 0
    roi_x1y1, roi_x2y2, roi_x3y3, roi_x4y4 = roiVertices[0],roiVertices[1],roiVertices[2],roiVertices[3] 
    roi_x, roi_y = min(roi_x1y1[0], roi_x2y2[0], roi_x3y3[0], roi_x4y4[0]), min(roi_x1y1[1], roi_x2y2[1], roi_x3y3[1], roi_x4y4[1])
    _, roi_height = max(roi_x1y1[0], roi_x2y2[0], roi_x3y3[0], roi_x4y4[0]) - roi_x, max(roi_x1y1[1], roi_x2y2[1], roi_x3y3[1], roi_x4y4[1]) - roi_y
    
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if class_id==0 and score>=detection_threshold:
                    feet_y = int(y2)
                    if roi_y <= feet_y <= roi_y + roi_height:
                        human_exit_count += 1
                        print("Human within exit region.")

    cv2.putText(frame, f"Humans in exit area: {human_exit_count}", (5, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1)
    return detections

#function for tracking people
def peopleTrackingDeepSort(frame, detections):
    tracker.update(frame, detections)
    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = bbox
        track_id = track.track_id
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 3)
        label = f"Person {track_id}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)


# main loop for processing
while ret:
    frame = cv2.resize(frame, (1280, 720))

    exitDoorROI()
    roiVertices = exitAreaROI(frame)
    detections = peopleDetectionYOLO(frame, roiVertices)
    # peopleTrackingDeepSort(frame, detections)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()