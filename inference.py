from ultralytics import YOLO
import cv2

def run_inference():
    print("Starting YOLO detection...")

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        annotated_frame = results[0].plot()
        cv2.imshow("Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
