import cv2
from ultralytics import YOLO
import os

# Verify the YOLO model path
model_path = r"C:\Users\ASUS\OneDrive\Desktop\SDP\best.pt"
if not os.path.exists(model_path):
    print("Error: YOLO model file not found. Check the path.")
    exit()

# Load your YOLOv10 model
try:
    model = YOLO(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load YOLO model. Error: {e}")
    exit()

# Function to process frames and run YOLO detection
def process_frame(frame, model):
    try:
        # Resize the frame for faster processing
        frame_resized = cv2.resize(frame, (640, 480))
        
        # Convert frame to RGB for YOLO
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Run YOLO detection
        results = model.predict(img_rgb, conf=0.1, save=False)  # Adjust confidence as needed
        print(f"Detection results: {results}")  # Debugging: Print detection results

        # Draw bounding boxes and labels
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                label = result.names[box.cls[0]]

                # Draw bounding box and label
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_resized,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
    except Exception as e:
        print(f"Error while processing frame: {e}")
    return frame_resized

# Open webcam for live video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

if not cap.isOpened():
    print("Error: Could not access the webcam. Ensure it is connected and accessible.")
    exit()

print("Press 'q' to exit the live detection.")

try:
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        print(f"Frame size: {frame.shape}")  # Debugging: Print the frame dimensions

        # Process the frame and detect objects
        processed_frame = process_frame(frame, model)

        # Display the processed frame
        cv2.imshow('YOLOv10 Live Object Detection', processed_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam and OpenCV windows closed.")
