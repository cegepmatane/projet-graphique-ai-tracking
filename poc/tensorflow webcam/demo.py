from imageai.Detection import ObjectDetection
import tensorflow as tf
import os
import cv2
import time

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
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel(detection_speed="fast")

custom = detector.CustomObjects(person=False)
camera = cv2.VideoCapture(0)
filtreObjets = False
font = cv2.FONT_HERSHEY_DUPLEX


#Main Loop
while True:
    start = time.time()
    ret, frame = camera.read()

    detected_image, detections = detector.detectObjectsFromImage(input_image=frame,
                                                                 input_type="array",
                                                                 output_type="array",
                                                                 minimum_percentage_probability=50)

    for eachObject in detections:
            #print(eachObject["name"], " : ", eachObject["percentage_probability"])
            (x1, y1, x2, y2) = eachObject["box_points"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, eachObject["name"] + " " + str(round(eachObject["percentage_probability"], 2)) + "%",
                       (x1 + 6, y1 - 6), font, 1.0, (0, 255, 0), 1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #FPS
    end = time.time()
    Inputframes = 30
    seconds = end - start
    fps = 1 / (seconds / 5)
    #print(str(round(seconds, 4)))
    cv2.putText(frame, str(round(fps)) + " fps", (7, 28), font, 1, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('Detection', frame)

camera.release()
cv2.destroyAllWindows()
