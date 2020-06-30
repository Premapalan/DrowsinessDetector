# importing libraries
import cv2
import dlib
import numpy as np

# path for predictor
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

# ensuring face detection works for single face

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return 'error'
    if len(rects) == 0:
        return 'No face'
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                   fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                   fontScale=0.4,
                   color=(0,0,255))
        cv2.circle(im, pos, 3, color=(0,255,255))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis = 0)
    return int(top_lip_mean[:, 1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis = 0)
    return int(bottom_lip_mean[:, 1])

def mouth_open(image):
    landmarks = get_landmarks(image)

    if landmarks == 'error':
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

# for right eyes
def top_eye_lid_right(landmarks):
    top_eye_lid_right_pts = []
    for i in range(42,44):
        top_eye_lid_right_pts.append(landmarks[i])
    top_eye_lid_right_all_pts = np.squeeze(np.asarray(top_eye_lid_right_pts))
    top_eye_lid_right_mean = np.mean(top_eye_lid_right_pts, axis = 0)
    return int(top_eye_lid_right_mean[:, 1])

def bottom_eye_lid_right(landmarks):
    bottom_eye_lid_right_pts = []
    for i in range(45,47):
        bottom_eye_lid_right_pts.append(landmarks[i])
    bottom_eye_lid_right_all_pts = np.squeeze(np.asarray(bottom_eye_lid_right_pts))
    bottom_eye_lid_right_mean = np.mean(bottom_eye_lid_right_pts, axis = 0)
    return int(bottom_eye_lid_right_mean[:, 1])

def right_eyes_open(image):
    landmarks = get_landmarks(image)

    if landmarks == 'error':
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_eye_lid_right_center = top_eye_lid_right(landmarks)
    bottom_eye_lid_right_center = bottom_eye_lid_right(landmarks)
    eye_lid_right_distance = abs(top_eye_lid_right_center - bottom_eye_lid_right_center)
    print('eye right distance:',eye_lid_right_distance)
    return image_with_landmarks, eye_lid_right_distance

# for left eyes
def top_eye_lid_left(landmarks):
    top_eye_lid_left_pts = []
    for i in range(36,38):
        top_eye_lid_left_pts.append(landmarks[i])
    top_eye_lid_left_all_pts = np.squeeze(np.asarray(top_eye_lid_left_pts))
    top_eye_lid_left_mean = np.mean(top_eye_lid_left_pts, axis = 0)
    return int(top_eye_lid_left_mean[:, 1])

def bottom_eye_lid_left(landmarks):
    bottom_eye_lid_left_pts = []
    for i in range(39,41):
        bottom_eye_lid_left_pts.append(landmarks[i])
    bottom_eye_lid_left_all_pts = np.squeeze(np.asarray(bottom_eye_lid_left_pts))
    bottom_eye_lid_left_mean = np.mean(bottom_eye_lid_left_pts, axis = 0)
    return int(bottom_eye_lid_left_mean[:, 1])

def left_eyes_open(image):
    landmarks = get_landmarks(image)

    if landmarks == 'error':
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_eye_lid_left_center = top_eye_lid_left(landmarks)
    bottom_eye_lid_left_center = bottom_eye_lid_left(landmarks)
    eye_lid_left_distance = abs(top_eye_lid_left_center - bottom_eye_lid_left_center)
    print('eye left distance:',eye_lid_left_distance)
    return image_with_landmarks, eye_lid_left_distance

# yawn detector main program trial and error
cap = cv2.VideoCapture(0)
yawns = 0
yawn_status = False
right_eye_open_status = False
left_eye_open_status = False


while True:
    ret, frame = cap.read()
    image_landmarks, lip_distance = mouth_open(frame)
    image_landmarks, eye_lid_right_distance = right_eyes_open(frame)
    image_landmarks, eye_lid_left_distance = left_eyes_open(frame)

    prev_yawn_status = yawn_status
    prev_right_eye_open_status = right_eye_open_status
    prev_left_eye_open_status = left_eye_open_status

    if lip_distance > 15:
        yawn_status = True

        cv2.putText(frame, 'subject is yawning', (10, 100),
                   cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)

#         output_text = 'Yawn Count: ' + str(yawns + 1)

#         cv2.putText(frame, output_text, (50,50),
#                    cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,127), 2)
    else:
        yawn_status = False

    if eye_lid_right_distance > 3:
        right_eye_open_status = True

        cv2.putText(frame, 'right eye is open', (10, 20),
                   cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1)

    else:
        right_eye_open_status = False
        cv2.putText(frame, 'right eye is closed', (10, 20),
                   cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255), 1)

    if eye_lid_left_distance > 3:
            left_eye_open_status = True

            cv2.putText(frame, 'left eye is open', (10, 40),
                       cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
    else:
        left_eye_open_status = False
        cv2.putText(frame, 'left eye is closed', (10, 40),
                   cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)

    if prev_yawn_status == True and yawn_status ==False:
        yawns += 1

    cv2.imshow('Live Landmarks', image_landmarks)
    cv2.imshow('Yawn Detection', frame)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
