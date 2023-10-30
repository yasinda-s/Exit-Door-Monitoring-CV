#sets a virtually accurate ROI for the exit area AND door and counts the number of people within the region
import os
import cv2
from ultralytics import YOLO
from tracker import Tracker
import numpy as np

#setup video, cap_out, yolo and deepsort
video_path = os.path.join('.', 'data', 'accident.mp4')
video_out_path = os.path.join('.', 'CCTV_out.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS),
                          (1280, 720))

model = YOLO("yolov8s-seg.pt") 
tracker = Tracker()
detection_threshold = 0.5
people_movement = {} 
people_last_10_steps_in_exit_door = {} 
people_last_10_steps_in_exit_area = {}
unique_incoming = []
unique_outgoing = []
frame_count = 0

#function for exit door
def exitDoorROI(frame):
    roi_vertices = np.array([[899, 100], [1051,141], [960,549], [863, 405]], np.int32)
    roi_x = 930
    roi_y = 110
    # cv2.polylines(frame, [roi_vertices], True, (0, 0, 255), 1)
    cv2.putText(frame, "Exit Door", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    # frame_count += 1
    return roi_vertices

#function for exit area ROI
def exitAreaROI():
    # roi_vertices = np.array([[899, 100], [1051,141], [952,553], [857, 404]], np.int32)
    # roi_vertices = np.array([[863, 405], [960,549], [606,598], [582, 420]], np.int32)
    # roi_vertices = np.array([[573, 400], [857,404], [952,553], [611, 659]], np.int32)
    roi_vertices = np.array([[580, 440], [857,404], [952,553], [611, 659]], np.int32)
    # cv2.polylines(frame, [roi_vertices], True, (0, 0, 255), 1)
    return roi_vertices

#function for detecting people (if feet within exit area)
def peopleDetectionYOLO(frame, exitAreaVertices, exitDoorVertices):
    # results = model(frame)
    results = model.predict(frame, classes=0)
    detections = []
    human_area_count = 0
    human_door_count = 0

    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r #top left and bottom right coordinates of each object
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            class_id = int(class_id)
            if score>=detection_threshold:
                    feet_coordinates = (((x1+x2)/2), y2-5)
                    if cv2.pointPolygonTest(exitAreaVertices, feet_coordinates, False) > 0:
                        human_area_count += 1
                    if cv2.pointPolygonTest(exitDoorVertices, feet_coordinates, False) > 0:
                        human_door_count += 1
                    detections.append([x1, y1, x2, y2, score])

    cv2.putText(frame, f"Workers currently in exit area: {human_area_count}", (5, 405), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(frame, f"Workers currently in exit door: {human_door_count}", (5, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    return detections

#function for tracking people
def peopleTrackingDeepSort(frame, detections, exitDoorVertices, exitAreaVertices):
    tracker.update(frame, detections)
    totalOutgoing = 0
    totalIncoming = 0
    # len_outgoing = 0
    # len_incoming = 0

    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = bbox
        track_id = track.track_id

        if track_id not in people_movement:
            people_movement[track_id] = []
            people_movement[track_id].append([x1, y1, x2, y2])
        else:
            people_movement[track_id].append([x1, y1, x2, y2])

        countOutgoing, countIncoming = countPersonOutgoing(exitDoorVertices, exitAreaVertices, track_id)
        totalOutgoing += countOutgoing
        totalIncoming += countIncoming
        # len_outgoing += len_outgoingfunc
        # len_incoming += len_incomingfunc

        # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
        label = f"WH3{track_id}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Workers currently outgoing: {totalOutgoing}", (5, 475), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(frame, f"Workers currently incoming: {totalIncoming}", (5, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(frame, f"Record of outgoing workers: {len(unique_outgoing)}", (5, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(frame, f"Record of incoming workers: {len(unique_incoming)}", (5, 580), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

def countPersonOutgoing(exitDoorVertices, exitAreaVertices, track_id):
    print("EXIT AREA HISTORY ", people_last_10_steps_in_exit_area)
    print("EXIT DOOR HISTORY ", people_last_10_steps_in_exit_door)
    last_coordinates = people_movement.get(track_id, [])[-30:]
    countOutgoing = 0
    countIncoming = 0

    if len(last_coordinates) < 2:
        return countOutgoing, countIncoming

    feet_coordinates = []
    for coordinate in last_coordinates:
        feet_coordinates.append(((coordinate[0] + coordinate[2]) / 2, (coordinate[3]-5)))

    if track_id not in people_last_10_steps_in_exit_door:
        people_last_10_steps_in_exit_door[track_id] = False

    if any(cv2.pointPolygonTest(exitDoorVertices, coordinate, False) > 0 for coordinate in feet_coordinates):
        people_last_10_steps_in_exit_door[track_id] = True
    else:
        people_last_10_steps_in_exit_door[track_id] = False

    if track_id not in people_last_10_steps_in_exit_area:
        people_last_10_steps_in_exit_area[track_id] = False

    if any(cv2.pointPolygonTest(exitAreaVertices, coordinate, False) > 0 for coordinate in feet_coordinates):
        people_last_10_steps_in_exit_area[track_id] = True
    else:
        people_last_10_steps_in_exit_area[track_id] = False

    if cv2.pointPolygonTest(exitAreaVertices, feet_coordinates[-1], False) > 0 and people_last_10_steps_in_exit_door[track_id] == True:
        countOutgoing += 1
        print(f"Person {track_id} is OUTGOING from exit door")
        if track_id not in unique_outgoing:
            unique_outgoing.append(track_id)
            # print(f"Unique outgoing: {unique_outgoing}")
    elif cv2.pointPolygonTest(exitDoorVertices, feet_coordinates[-1], False) > 0 and people_last_10_steps_in_exit_area[track_id] == True:
        countIncoming += 1
        print(f"Person {track_id} is INCOMING to exit door")
        if track_id not in unique_incoming:
            unique_incoming.append(track_id)
            # print(f"Unique incoming: {unique_incoming}")

    # print(countOutgoing, countIncoming, len(unique_outgoing), len(unique_incoming))
    return countOutgoing, countIncoming

# main loop for processing
while ret:
    frame = cv2.resize(frame, (1280, 720))
    frame_count += 1
    print("Frame Count - ", frame_count)
    cv2.rectangle(frame, (5, 380), (540, 595), (0, 0, 0), -1)
    exitDoorVertices = exitDoorROI(frame)
    exitAreaVertices = exitAreaROI()
    detections = peopleDetectionYOLO(frame, exitAreaVertices, exitDoorVertices)
    peopleTrackingDeepSort(frame, detections, exitDoorVertices, exitAreaVertices)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap_out.write(frame)
    ret, frame = cap.read()

print(f"Unique outgoing: {unique_outgoing}")
print(f"Unique incoming: {unique_incoming}")
cap.release()
cap_out.release()
cv2.destroyAllWindows()