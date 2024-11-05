from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import threading
from pygame import mixer

# Head pose
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


# Head pose detection function
def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # Calculate Euler angles
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


# For Alarms
mixer.init()
eyes_closed_alarm_path = r"C:\Users\user\OneDrive\Desktop\FYP\Development Interface\Rainy\Real-Time---Driver-Fatigue-Detection-System\sound.wav"
yawning_alarm_path = r"C:\Users\user\OneDrive\Desktop\FYP\Development Interface\Rainy\Real-Time---Driver-Fatigue-Detection-System\music.wav"

def sound_eyes_closed_alarm():
    mixer.music.load(eyes_closed_alarm_path)
    mixer.music.play()

def sound_yawning_alarm():
    mixer.music.load(yawning_alarm_path)
    mixer.music.play()


# Constants for aspect ratios
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 45
MOU_AR_THRESH = 0.75


# Mouth aspect ratio calculation
def mouth_aspect_ratio(mou):
    # Compute the euclidean distances between the horizontal
    X = dist.euclidean(mou[0], mou[6])
    # Compute the euclidean distances between the vertical
    Y1 = dist.euclidean(mou[2], mou[10])
    Y2 = dist.euclidean(mou[4], mou[8])
    # Taking average
    Y = (Y1 + Y2) / 2.0
    # Compute mouth aspect ratio
    mar = Y / X
    return mar


# Eye aspect ratio calculation
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmarks (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear


# Initialize dlib's face detector (HOG-based) and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = r'C:\Users\user\OneDrive\Desktop\FYP\Development Interface\Rainy\Real-Time---Driver-Fatigue-Detection-System\shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# Grab the indexes of the facial landmarks for the left and right eye and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Initialize video capture
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize variables
COUNTER = 0
yawnStatus = False
yawns = 0

# Loop over capturing video
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame
    frame = imutils.resize(frame, width=640)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_yawn_status = yawnStatus

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    for rect in rects:
        # Determine the facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates, then compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouEAR = mouth_aspect_ratio(mouth)

        # Average the eye aspect ratio for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # Head pose detection
        reprojectdst, euler_angle = get_head_pose(shape)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # Draw lines for head pose
        for start, end in line_pairs:
            start_point = tuple(map(int, reprojectdst[start]))
            end_point = tuple(map(int, reprojectdst[end]))
            cv2.line(frame, start_point, end_point, (0, 0, 255))

        # Display Euler angles
        xx = euler_angle[0, 0]
        cv2.putText(frame, "X: " + "{:7.2f}".format(xx), (20, 420), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 0), thickness=2)
        yy = euler_angle[1, 0]
        cv2.putText(frame, "Y: " + "{:7.2f}".format(yy), (20, 445), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 0), thickness=2)
        cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 470), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 0), thickness=2)
        
        # Check if the pitch angle is below a certain threshold
        if xx > 10:  # You can adjust this threshold as needed
            cv2.putText(frame, "Head Down", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            s = threading.Thread(name='sound_yawning_alarm', target=sound_yawning_alarm)
            s.start()
            
        # Check to see if the eye aspect ratio is below the blink threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            cv2.putText(frame, "Eyes Closed ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # If the eyes were closed for a sufficient number of frames, trigger an alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                s = threading.Thread(name='sound_eyes_closed_alarm', target=sound_eyes_closed_alarm)
                s.start()

        # Otherwise, the eye aspect ratio is not below the blink threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            cv2.putText(frame, "Eyes Open ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # Yawning detections
        if mouEAR > MOU_AR_THRESH:
            cv2.putText(frame, "Yawning ", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            yawnStatus = True
            output_text = "Yawn Count: " + str(yawns + 1)
            cv2.putText(frame, output_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            yawnStatus = False
        
        if prev_yawn_status == True and yawnStatus == False:
            yawns += 1
        
        # Check if yawns exceed the threshold
        if yawns > 2:
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            s = threading.Thread(name='sound_yawning_alarm', target=sound_yawning_alarm)
            s.start()
            yawns = 0
        
        cv2.putText(frame, "MAR: {:.2f}".format(mouEAR), (480, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check for the 'q' key to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
