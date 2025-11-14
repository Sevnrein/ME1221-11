import cv2
from fer.fer import FER

emotion_detector = FER()

cap = cv2.VideoCapture(0);

if not cap.isOpened():
    print("Error: Can't open the camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't read the video frame.")
        break

    faces = emotion_detector.detect_emotions(frame)
    for face in faces:
        box = face['box']
        emotion = face['emotions']
        cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
        max_emotion = {'none': 0} 
        for key in emotion:
            if emotion.get(key) > next(iter(max_emotion.values())):
                max_emotion.clear()
                max_emotion[key] = emotion.get(key)
        cv2.putText(frame, next(iter(max_emotion.keys())), (box[0], box[1]+box[3]+10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow('My Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("The program is closing...")
cap.release()
cv2.destroyAllWindows()
