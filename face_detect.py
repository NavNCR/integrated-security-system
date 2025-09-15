from mtcnn import MTCNN
detector = MTCNN()
faces = detector.detect_faces(frame)
for face in faces:
    x, y, w, h = face['box']
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
cv2.imshow('Face Detected', frame)
