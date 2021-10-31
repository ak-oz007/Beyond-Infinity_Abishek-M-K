
def posedetector(cap):
    import cv2
    import mediapipe as mp
    import numpy as np

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recoloring the image for mediapipe to work
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
            out = cv2.VideoWriter("Final Output.avi", fourcc, 5.0, (1280, 720))

            # Making  the  detecion
            results = pose.process(image)

            # Recoloring back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extracting landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                print(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])
            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            # imj = cv2.resize(image, (320, 480))
            out.write(image)
            # cv2.imshow('Mediapipe Feed!',image)
            cv2.imshow('Mediapipe Pose', cv2.flip(image, 1))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        # cap.release()
        # cv2.destroyAllWindows
