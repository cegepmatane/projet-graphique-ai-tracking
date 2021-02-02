from imageai.Detection import ObjectDetection
import tensorflow as tf
import os
import cv2

#FIX ERREURS CUBLAS
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

execution_path = os.getcwd()

detector = ObjectDetection()

detector.setModelTypeAsRetinaNet()
#detector.setModelTypeAsYOLOv3()
#detector.setModelTypeAsTinyYOLOv3()

detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.1.0.h5"))
#detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
#detector.setModelPath(os.path.join(execution_path, "yolo-tiny.h5"))

detector.loadModel(detection_speed="fast")

#custom = detector.CustomObjects(person=True, cell_phone=True, bottle=True, bed=True, book=True, cup=True)   custom_objects=custom,

camera = cv2.VideoCapture(1)

while True:
    ret, frame = camera.read()
    detected_image, detections = detector.detectObjectsFromImage(
                                                                 input_image=frame, input_type="array",
                                                                 output_type="array")
    """
    for detection in detections:
        (x1, y1, x2, y2) = detection["box_points"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, detection["name"], (x1 + 6, y1 - 6), font, 1.0, (0, 255, 0), 1)"""

    for eachObject in detections:
        print(eachObject["name"], " : ", eachObject["percentage_probability"])
        (x1, y1, x2, y2) = eachObject["box_points"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, eachObject["name"], (x1 + 6, y1 - 6), font, 1.0, (0, 255, 0), 1)

    cv2.imshow('Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
