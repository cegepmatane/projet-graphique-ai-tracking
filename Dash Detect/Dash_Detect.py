import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QApplication, QMainWindow

import numpy as np
import tensorflow as tf
from imageai.Detection import ObjectDetection
import cv2 as cv
import time

#Configuration pour eviter les erreur "CUBLAS_STATUS_ALLOC_FAILED"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Actuellement, la taille de la memoire a besoin d'etre la meme sur les 2 GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # La taille de la memoire doit etre calibré avant l'initialisation du GPU
        print(e)

COLONNES_FILTRES = 9
LIGNES_FILTRES = 9
FILTRES = [
    "bicycle",
    "car",
    "person",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
]


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.largeur_fenetre = 1300
        self.hauteur_fenetre = 800
        self.setGeometry(200, 100, 1300, 800)
        self.setMaximumSize(self.largeur_fenetre, self.hauteur_fenetre)
        self.setWindowTitle("Dash Detect Version 0.1")

        self.threadpool = QtCore.QThreadPool()
        self.threadVideo = ThreadDetectionVideo(self)
        self.threadVideo.setAutoDelete(False)

        self.threadVideo.signals.envoi_frame.connect(self.rafraichirFrameVideo)

        self.creerGui()

    def creerGui(self):
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)

        self.sections = QtWidgets.QVBoxLayout()
        self.sections.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.sections.setContentsMargins(0, 0, 0, 0)
        self.sections.setSpacing(0)
        self.sections.setObjectName("sections")

        self.conteneur_video = QtWidgets.QGroupBox()
        self.conteneur_video.setMaximumSize(QtCore.QSize(1280, 600))
        self.conteneur_video.setObjectName("conteneur_video")

        # self.lecteur_video = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.VideoSurface)
        # self.lecteur_video.setMuted(True)
        # self.lecteur_video.error.connect(self.gererErreur)

        # self.flux_video = QVideoWidget()
        # self.flux_video.setAspectRatioMode(QtCore.Qt.KeepAspectRatio)
        # self.flux_video.setMaximumSize(QtCore.QSize(1280,600))
        # self.flux_video.setObjectName("flux_video")

        self.frame_video = QtWidgets.QLabel(self.conteneur_video)
        self.frame_video.setGeometry(QtCore.QRect(160, 20, 871, 541))
        self.frame_video.setAlignment(QtCore.Qt.AlignCenter)
        self.frame_video.setText("Sélectionnez un fichier vidéo à analyser ou bien un flux en direct depuis une caméra")
        self.frame_video.setScaledContents(True)
        self.frame_video.setObjectName("frame_video")


        self.conteneur_controles = QtWidgets.QGroupBox()
        self.conteneur_controles.setMaximumSize(QtCore.QSize(1280, 300))
        self.conteneur_controles.setObjectName("conteneur_controles")

        self.controles_source = QtWidgets.QFrame(self.conteneur_controles)
        self.controles_source.setGeometry(QtCore.QRect(10, 20, 151, 171))
        self.controles_source.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.controles_source.setFrameShadow(QtWidgets.QFrame.Raised)
        self.controles_source.setObjectName("controles_source")
        self.btn_fichier = QtWidgets.QPushButton(self.controles_source)
        self.btn_fichier.setGeometry(QtCore.QRect(10, 20, 131, 31))
        self.btn_fichier.setObjectName("btn_fichier")
        self.btn_fichier.clicked.connect(self.naviguerFichiers)

        self.nom_fichier = QtWidgets.QLabel(self.controles_source)
        self.nom_fichier.setGeometry(QtCore.QRect(10, 55, 131, 25))
        self.nom_fichier.setObjectName("nom_fichier")

        self.btn_source = QtWidgets.QPushButton(self.controles_source)
        self.btn_source.setGeometry(QtCore.QRect(10, 90, 131, 31))
        self.btn_source.setObjectName("btn_source")

        self.controles_lecture = QtWidgets.QFrame(self.conteneur_controles)
        self.controles_lecture.setGeometry(QtCore.QRect(170, 20, 161, 171))
        self.controles_lecture.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.controles_lecture.setFrameShadow(QtWidgets.QFrame.Raised)
        self.controles_lecture.setObjectName("controles_lecture")
        self.btn_lecture = QtWidgets.QPushButton(self.controles_lecture)
        self.btn_lecture.setGeometry(QtCore.QRect(10, 20, 61, 61))
        self.btn_lecture.setObjectName("btn_lecture")
        self.btn_lecture.clicked.connect(self.threadVideo.mettreEnMarche)

        self.btn_pause = QtWidgets.QPushButton(self.controles_lecture)
        self.btn_pause.setGeometry(QtCore.QRect(90, 20, 61, 61))
        self.btn_pause.setObjectName("btn_pause")
        self.btn_pause.clicked.connect(self.threadVideo.mettreEnPause)

        self.btn_stop = QtWidgets.QPushButton(self.controles_lecture)
        self.btn_stop.setGeometry(QtCore.QRect(10, 90, 141, 31))
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.clicked.connect(self.threadVideo.arreter)


        self.controles_filtres = QtWidgets.QGroupBox(self.conteneur_controles)
        self.controles_filtres.setGeometry(QtCore.QRect(340, 20, 931, 271))
        self.controles_filtres.setObjectName("controles_filtres")
        self.tous_filtre = QtWidgets.QCheckBox(self.controles_filtres)
        self.tous_filtre.setGeometry(QtCore.QRect(20, 70, 50, 20))
        self.tous_filtre.setObjectName("tous_filtre")

        self.tous_filtre.stateChanged.connect(self.actionnerTousLesFiltres)

        self.filtres = QtWidgets.QFrame(self.controles_filtres)
        self.filtres.setGeometry(QtCore.QRect(170, 20, 161, 271))
        self.filtres.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.filtres.setFrameShadow(QtWidgets.QFrame.Raised)
        self.filtres.setObjectName("filtres")
        self.construireCheckboxesFiltres()

        # Attribution des noms
        self.conteneur_video.setTitle("Visualisation")
        self.conteneur_controles.setTitle("Panneau de contrôle")
        self.btn_fichier.setText("Fichier vidéo")
        self.btn_source.setText("Source vidéo")
        self.btn_lecture.setText("Lire")
        self.btn_pause.setText("Pause")
        self.btn_stop.setText("Stop")
        self.controles_filtres.setTitle("Filtres de détection")
        self.tous_filtre.setText("Tous")

        self.sections.addWidget(self.conteneur_video)
        self.sections.addWidget(self.conteneur_controles)
        self.centralwidget.setLayout(self.sections)

        # self.lecteur_video.setVideoOutput(self.flux_video)

    def construireCheckboxesFiltres(self):
        self.checkboxes_filtres = []
        index_filtre = 0

        for i in range(LIGNES_FILTRES):
            for j in range(COLONNES_FILTRES):
                if index_filtre >= len(FILTRES):
                    return
                chaine_filtre = FILTRES[index_filtre]
                x_offset = 85 * (j + 1)
                y_offset = 25 * (i + 1)
                nouveau_filtre = QtWidgets.QCheckBox(self.controles_filtres)
                nouveau_filtre.setGeometry(QtCore.QRect(x_offset, y_offset, 85, 20))
                nouveau_filtre.setObjectName("filtre_" + chaine_filtre)
                nouveau_filtre.setText(chaine_filtre)
                self.checkboxes_filtres.append(nouveau_filtre)
                index_filtre += 1

    # FONCTIONS POUR UN FICHIER VIDEO
    def naviguerFichiers(self):
        nom_fichier, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                       "Sélectionner un fichier",
                                       os.getcwd(),
                                       "Fichiers vidéo (*.mp4 *.avi *.mov *.qt *.wmv *.mkv)")

        self.chargerFichier(nom_fichier)

    def chargerFichier(self, fichier):
        if fichier != '':
            self.nom_fichier.setText(os.path.split(fichier)[1])
            self.frame_video.setText("")
            # self.sections.replaceWidget(self.conteneur_video, self.flux_video)
            # self.lecteur_video.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(fichier)))
            self.threadVideo.video = cv.VideoCapture(fichier)
            # self.lireVideo()
            self.threadpool.start(self.threadVideo)

    @QtCore.pyqtSlot(np.ndarray)
    def rafraichirFrameVideo(self, frame):
        pixmap = self.convertirVersPixmap(frame)
        self.frame_video.setPixmap(pixmap)

    # Convertir une frame de VideoCapture de OpenCV vers une Pixmap utilisable par pyQt5
    def convertirVersPixmap(self, image_cv):
        image_rgb = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)
        hauteur, largeur, canaux = image_rgb.shape
        octets_par_ligne = canaux * largeur
        image_format_qt = QtGui.QImage(image_rgb.data, largeur, hauteur, octets_par_ligne, QtGui.QImage.Format_RGB888)
        image_redimensionnee = image_format_qt.scaled(self.largeur_fenetre, self.hauteur_fenetre, QtCore.Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(image_redimensionnee)


    # CONTROLES DE LA LECTURE

    # def lireVideo(self):
    #     if self.lecteur_video.state() == QtMultimedia.QMediaPlayer.PausedState or QtMultimedia.QMediaPlayer.StoppedState:
    #         self.lecteur_video.play()
    #         self.threadVideo.mettreEnMarche()
    #
    # def mettreEnPauseVideo(self):
    #     if self.lecteur_video.state() == QtMultimedia.QMediaPlayer.PlayingState:
    #         self.lecteur_video.pause()
    #         self.threadVideo.mettreEnPause()
    #
    # def stopperVideo(self):
    #     if self.lecteur_video.state() == QtMultimedia.QMediaPlayer.PlayingState or QtMultimedia.QMediaPlayer.PausedState:
    #         self.lecteur_video.stop()
    #         self.threadVideo.arreter()

    # def gererErreur(self):
    #    print("Erreur: " + self.lecteur_video.errorString())

    # CONTROLES DES FILTRES
    def actionnerTousLesFiltres(self, etat):
        checked = True if QtCore.Qt.Checked == etat else False
        for filtre in self.checkboxes_filtres:
            filtre.setChecked(checked)


class Signals(QtCore.QObject):
    envoi_frame = QtCore.pyqtSignal(np.ndarray)


class ThreadDetectionVideo(QtCore.QRunnable):
    signals = Signals()

    def __init__(self, application):
        super(ThreadDetectionVideo, self).__init__()
        self.application = application
        self.execution_path = os.getcwd()
        self.detecteur = ObjectDetection()
        self.video = None
        self.font = cv.FONT_HERSHEY_DUPLEX
        self.detecteur.setModelTypeAsYOLOv3()
        self.detecteur.setModelPath(os.path.join(self.execution_path, "yolo.h5"))
        self.detecteur.loadModel(detection_speed="flash")

        self.en_pause = False
        self.est_arrete = False

    @QtCore.pyqtSlot()
    def run(self):
        while True:
            if not self.en_pause:
                self.detecterObjets()

            if cv.waitKey(1) & self.est_arrete:
                break

        # Libération de la mémoire et destruction de la fênetre
        self.video.release()
        # cv.destroyAllWindows()

        self.est_arrete = False

    def detecterObjets(self):
        # Début de la lecture du temps pour les fps et lecture du flux
        temps_debut = time.time()

        # Analyse et détection des objets image par image
        ret, frame = self.video.read()
        detected_image, detections = self.detecteur.detectObjectsFromImage(input_image=frame,
                                                                           input_type="array",
                                                                           output_type="array",
                                                                           minimum_percentage_probability=60)
        # Calcule et affchage des fps
        temps_fin = time.time()
        secondes = temps_fin - temps_debut
        fps = 1 / secondes
        cv.putText(detected_image, str(round(fps)) + " fps", (7, 28), self.font, 1, (0, 0, 255), 3, cv.LINE_AA)

        if ret:
            self.signals.envoi_frame.emit(detected_image)

    def mettreEnPause(self):
        self.en_pause = True

    def mettreEnMarche(self):
        self.en_pause = False

    def arreter(self):
        self.est_arrete = True


application = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(application.exec_())
