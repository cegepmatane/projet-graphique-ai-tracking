from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()
detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()
detections = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "walk.mp4"),
                                             output_file_path=os.path.join(execution_path, "resultat"))

for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"])