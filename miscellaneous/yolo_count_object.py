from collections import defaultdict
from ultralytics import YOLO
import os
import cv2


class CountObjectInArea:
    def __init__(self, model_name, video_path):
        self.model_name = model_name
        self.video_path = video_path
        self.object_counts = defaultdict(int)
        self.total_count = 0

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
                self.object_counts = defaultdict(int)
                self.total_count = 0
                predictions = self.model.track(frame)
                prediction = predictions[0]
                class_names = prediction.names
                boxes = prediction.boxes
                xyxy = boxes.xyxy
                cls = boxes.cls
                conf = boxes.conf

                for index in range(len(xyxy)):
                    box = xyxy[index]
                    class_id = cls[index]
                    class_name = class_names[int(class_id)]
                    confident = conf[index]
                    x1, y1, x2, y2 = map(int, box[:4])
                    color = (0, 255, 0)
                    thickness = 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(
                        frame,
                        f"{class_name} - {confident:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        thickness,
                    )
                    self.object_counts[class_name] += 1
                    self.total_count += 1

                y_offset = 30
                cv2.putText(
                    frame,
                    f"Total objects: {self.total_count}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

                for i, (class_name, count) in enumerate(self.object_counts.items()):
                    y_offset += 30
                    cv2.putText(
                        frame,
                        f"{class_name}: {count}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

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
