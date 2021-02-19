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

#Chargement de la source vidéo (ici la webcam)
execution_path = os.getcwd()
camera = cv2.VideoCapture(0)


#Paramétrage du modèle
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel(detection_speed="fast")

#Option supplémentaires au modèle
custom = detector.CustomObjects(person=False) #Il est possible de mettre 80 objets différents
filtreObjets = False
font = cv2.FONT_HERSHEY_DUPLEX

#Boucle principale
while True:
    #Début de la lecture du temps pour les fps et lecture du flux
    start = time.time()
    ret, frame = camera.read()

    #Analyse et détection des objets image par image
    detected_image, detections = detector.detectObjectsFromImage(input_image=frame,
                                                                 input_type="array",
                                                                 output_type="array",
                                                                 minimum_percentage_probability=50)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #Calcule et affchage des fps
    end = time.time()
    Inputframes = 30
    seconds = end - start
    fps = 1 / (seconds / 5)
    #print(str(round(seconds, 4)))
    cv2.putText(detected_image, str(round(fps)) + " fps", (7, 28), font, 1, (0, 0, 255), 3, cv2.LINE_AA)

    #Ouverture de la fenetre finale opencv
    cv2.imshow('Detection', detected_image)

#Libération de la mémoire et destruction de la fênetre
camera.release()
cv2.destroyAllWindows()
