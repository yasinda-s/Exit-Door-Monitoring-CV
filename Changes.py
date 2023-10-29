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
tracker = Tracker()
detection_threshold = 0.3
people_movement = {} #dictionary for tracking people - use this to check if the person is incoming or outgoing the exit
people_last_10_steps_in_exit_door = {} #dictionary for tracking people - use this to check if the person is incoming or outgoing the exit
frame_count = 0

#function for exit door
def exitDoorROI(frame):
    roi_vertices = np.array([[710, 31], [810,36], [790,220], [700, 215]], np.int32)
    roi_x = 700
    roi_y = 36
    cv2.polylines(frame, [roi_vertices], True, (0, 0, 255), 1)
    cv2.putText(frame, "Exit Door", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    # frame_count += 1
    return roi_vertices

#function for exit area ROI
def exitAreaROI():
    roi_vertices = np.array([[520, 340], [640,220], [800,230], [783, 380]], np.int32)
    return roi_vertices

#function for detecting people (if feet within exit area)
def peopleDetectionYOLO(frame, exitAreaVertices, exitDoorVertices):
    results = model(frame)
    detections = []
    human_area_count = 0
    human_door_count = 0

    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r #top left and bottom right coordinates of each object
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            class_id = int(class_id)
            if class_id==0 and score>=detection_threshold:
                    feet_coordinates = (((x1+x2)/2), y2-5)
                    if cv2.pointPolygonTest(exitAreaVertices, feet_coordinates, False) > 0:
                        human_area_count += 1
                    if cv2.pointPolygonTest(exitDoorVertices, feet_coordinates, False) > 0:
                        human_door_count += 1
                    detections.append([x1, y1, x2, y2, score])

    cv2.putText(frame, f"Workers in exit area: {human_area_count}", (5, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1)
    cv2.putText(frame, f"Workers through exit door: {human_door_count}", (5, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1)
    return detections

#function for tracking people
def peopleTrackingDeepSort(frame, detections, exitDoorVertices, exitAreaVertices):
    tracker.update(frame, detections)
    totalOutgoing = 0
    totalIncoming = 0
    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = bbox
        track_id = track.track_id

        if track_id not in people_movement:
            people_movement[track_id] = []
        else:
            people_movement[track_id].append([x1, y1, x2, y2])

        countOutgoing, countIncoming = countPersonOutgoing(exitDoorVertices, exitAreaVertices, track_id)

        totalOutgoing += countOutgoing
        totalIncoming += countIncoming

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
        label = f"Staff Worker {track_id}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.putText(frame, f"Workers outgoing: {totalOutgoing}", (5, 95), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1)
    cv2.putText(frame, f"Workers incoming: {totalIncoming}", (5, 130), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1)

def countPersonOutgoing(exitDoorVertices, exitAreaVertices, track_id): #Check why this is not working, this works now, check why
    last_coordinates = people_movement.get(track_id, [])[-10:]
    countOutgoing = 0
    countIncoming = 0
    # Check if there are at least 2 coordinates to compare
    if len(last_coordinates) < 2:
        return countOutgoing, countIncoming
    # Calculate the feet coordinates for the last 10 coordinates
    feet_coordinates = []
    for coordinate in last_coordinates:
        feet_coordinates.append(((coordinate[0] + coordinate[2]) / 2, coordinate[3] - 5))

    if track_id not in people_last_10_steps_in_exit_door:
        people_last_10_steps_in_exit_door[track_id] = False
    else:
        for coordinate in feet_coordinates:
            if cv2.pointPolygonTest(exitDoorVertices, coordinate, False) > 0:
                people_last_10_steps_in_exit_door[track_id] = True

    if cv2.pointPolygonTest(exitAreaVertices, feet_coordinates[-1], False) > 0 and people_last_10_steps_in_exit_door[track_id] == True:
        countOutgoing += 1
        print(f"Person {track_id} is OUTGOING from exit door")
        # print(countOutgoing)
        # return countOutgoing
    elif cv2.pointPolygonTest(exitAreaVertices, feet_coordinates[-1], False) > 0 and people_last_10_steps_in_exit_door[track_id] == False:
        countIncoming += 1
        print(f"Person {track_id} is INCOMING to exit door")
        # return countIncoming

    return countOutgoing, countIncoming

# main loop for processing
while ret:
    frame = cv2.resize(frame, (1280, 720))
    frame_count += 1
    print("Frame Count - ", frame_count)
    exitDoorVertices = exitDoorROI(frame)
    exitAreaVertices = exitAreaROI()
    detections = peopleDetectionYOLO(frame, exitAreaVertices, exitDoorVertices)
    peopleTrackingDeepSort(frame, detections, exitDoorVertices, exitAreaVertices)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

    cap_out.write(frame)
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()