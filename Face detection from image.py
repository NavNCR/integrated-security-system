import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load the input image
image_path = r"C:\Users\HP\Downloads\your_image.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Convert the image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize the Face Detection model
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    # Perform face detection
    results = face_detection.process(image_rgb)

    # Check if faces are detected
    if not results.detections:
        print("No faces detected.")
    else:
        # Annotate the image with face detections
        annotated_image = image.copy()
        for detection in results.detections:
            mp_drawing.draw_detection(annotated_image, detection)

        # Display the annotated image
        cv2.imshow("Face Detection", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
