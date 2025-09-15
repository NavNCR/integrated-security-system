import cv2
from mtcnn import MTCNN

# Initialize the MTCNN face detector
detector = MTCNN()

# Open the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

# Check if the camera was opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Start the main loop to read frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the frame was not read successfully, break the loop
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to RGB as MTCNN expects RGB input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the current frame
    faces = detector.detect_faces(rgb_frame)
    # 

    # Loop through all detected faces and draw a rectangle around them
    for face in faces:
        # Get the coordinates of the bounding box
        x, y, width, height = face['box']
        
        # Draw the rectangle on the original BGR frame
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Display the frame with the face detections
    cv2.imshow('Face Detection Project', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()