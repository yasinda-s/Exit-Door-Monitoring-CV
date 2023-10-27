#does ROI for door and exit area + checks if humans are within the exit ROI
import os
import cv2
from ultralytics import YOLO
# from tracker import Tracker

#setup video, cap_out, yolo and deepsort
video_path = os.path.join('.', 'data', 'CCTV_in.mp4')
video_out_path = os.path.join('.', 'CCTV_out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("yolov8n.pt")
# tracker = Tracker()

detection_threshold = 0.3

#function for exit door
def exitDoorROI():
    roi_x = 700
    roi_y = 36
    roi_width = 96
    roi_height = 190
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 0, 255), 2)
    cv2.putText(frame, "Exit Door", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

#function for exit area ROI
def exitAreaROI():
    roi_x = 500
    roi_y = 180
    roi_width = 280
    roi_height = 200
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0,165,255), 2)
    cv2.putText(frame, "Exit Area", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,165,255), 1)
    return [roi_x, roi_y, roi_width, roi_height]

#function for detecting people (crossing area ROI)
def peopleDetectionYOLO(frame, areaROICoordinates):
    results = model(frame)
    detections = []
    roi_x, roi_y, roi_width, roi_height = areaROICoordinates[0],areaROICoordinates[1],areaROICoordinates[2],areaROICoordinates[3] 
    human_exit_count = 0 
    
    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if class_id==0 and score>=detection_threshold:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    if roi_x <= center_x <= roi_x + roi_width and roi_y <= center_y <= roi_y + roi_height:
                        human_exit_count += 1
                        print("Human within exit Region")
                    detections.append([x1, y1, x2, y2, score])

    cv2.putText(frame, f"Humans in exit area: {human_exit_count}", (5, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1)
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
    areaROICoordinates = exitAreaROI()
    detections = peopleDetectionYOLO(frame, areaROICoordinates)
    # peopleTrackingDeepSort(frame, detections)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()