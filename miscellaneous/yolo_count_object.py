from ultralytics import YOLO
import os
import cv2


class CountObjectInArea:
    def __init__(self, model_name, video_path):
        self.model_name = model_name
        self.video_path = video_path
        self.track_history = {}

    def init_model(self):
        self.model = YOLO(self.model_name)

    def start_capture(self):
        self.capture = cv2.VideoCapture(self.video_path)

    def listen_to_exit(self):
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.capture.release()
            cv2.destroyAllWindows()

    def __call__(self, *args, **kwds):
        self.init_model()
        self.start_capture()
        while self.capture.isOpened():
            success, frame = self.capture.read()
            if success:
                predictions = self.model.track(frame)
                prediction = predictions[0]
                boxes = prediction.boxes
                xyxy = boxes.xyxy.numpy()
                for box in xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])
                    color = (0, 255, 0)
                    thickness = 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.imshow("YOLO Object Tracking", frame)
                self.listen_to_exit()


if __name__ == "__main__":
    model_name = "yolo11n.pt"
    video_path = os.path.join(
        os.path.dirname(__file__),
        "./yolo_count_object.mp4",
    )

    counter = CountObjectInArea(model_name, video_path)
    counter()
