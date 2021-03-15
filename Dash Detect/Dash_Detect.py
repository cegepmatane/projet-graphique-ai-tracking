import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
import breeze_ressources
import numpy as np
import tensorflow as tf
from imageai.Detection import ObjectDetection
import cv2 as cv
import time

COLONNES_FILTRES = 10
LIGNES_FILTRES = 10
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
        self.largeur_fenetre = 1280
        self.hauteur_fenetre = 720
        self.setGeometry(200, 100, self.largeur_fenetre, self.hauteur_fenetre)
        self.setMinimumSize(800, 600)
        self.setWindowTitle("Dash Detect Version 0.2")
        self.setWindowIcon(QIcon("logo.png"))
        self.gif = QtGui.QMovie("chargement.gif")

        self.threadpool = QtCore.QThreadPool()
        self.threadVideo = ThreadDetectionVideo(self)
        self.threadVideo.setAutoDelete(False)
        self.threadVideo.signals.envoi_frame.connect(self.rafraichirFrameVideo)

        self.creerGui()

        # Activation de tous les filtres au démarrage
        self.tous_filtre.setChecked(True)

    def creerGui(self):
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)

        self.sections = QtWidgets.QVBoxLayout()
        self.sections.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.sections.setContentsMargins(10, 10, 10, 10)
        self.sections.setSpacing(5)
        self.sections.setObjectName("sections")

        # VIDEO
        self.conteneur_video = QtWidgets.QGroupBox("Visualisation")
        self.conteneur_video.setObjectName("conteneur_video")

        self.layout_conteneur_video = QtWidgets.QHBoxLayout()
        self.conteneur_video.setLayout(self.layout_conteneur_video)
        self.layout_conteneur_video.setAlignment(QtCore.Qt.AlignCenter)

        self.frame_video = QtWidgets.QLabel(
            "Sélectionnez un fichier vidéo à analyser ou bien un flux en direct depuis une caméra",
            self.conteneur_video
        )
        self.frame_video.setScaledContents(True)
        self.frame_video.setObjectName("frame_video")
        self.gif_chargement = QtWidgets.QLabel(self.conteneur_video)
        self.gif_chargement.setGeometry(5, 5, self.conteneur_video.width(), 80)
        self.gif_chargement.setAlignment(QtCore.Qt.AlignLeft)
        self.gif_chargement.setObjectName("frame_gif_chargement")

        self.layout_conteneur_video.addWidget(self.frame_video)
        self.layout_conteneur_video.addWidget(self.gif_chargement)

        # CONTROLES
        self.conteneur_controles = QtWidgets.QGroupBox("Panneau de contrôle")
        self.conteneur_controles.setObjectName("conteneur_controles")
        self.conteneur_controles.setMaximumHeight(250)

        self.layout_conteneur_controles = QtWidgets.QHBoxLayout()
        self.conteneur_controles.setLayout(self.layout_conteneur_controles)
        self.layout_conteneur_controles.setAlignment(QtCore.Qt.AlignCenter)

        # SOURCES
        self.controles_source = QtWidgets.QVBoxLayout()
        self.controles_source.setObjectName("controles_source")
        self.controles_source.setAlignment(QtCore.Qt.AlignCenter)

        self.btn_fichier = QtWidgets.QPushButton("Fichier vidéo")
        self.btn_fichier.setObjectName("btn_fichier")
        self.btn_fichier.clicked.connect(self.naviguerFichiers)

        """self.choix_modele = QtWidgets.QComboBox(self.conteneur_controles)
        self.choix_modele.setGeometry(QtCore.QRect(20, 175, 131, 31))
        self.choix_modele.setObjectName("selecteur_modele_detection")
        self.choix_modele.addItems(["Resnet 50", "Yolo V3", "Tiny Yolo V3"])
        self.choix_modele.setDisabled(True)"""

        self.nom_fichier = QtWidgets.QLabel()
        self.nom_fichier.setObjectName("nom_fichier")
        self.btn_fichier.setToolTip("Appuyer sur Stop pour relancer une analyse")
        self.btn_source = QtWidgets.QPushButton("Source vidéo")
        self.btn_source.setObjectName("btn_source")
        self.btn_source.clicked.connect(self.chargerWebcam)
        self.btn_source.setToolTip("Appuyer sur Stop pour relancer une analyse")
        self.btn_info = QtWidgets.QPushButton("À propos ⓘ")
        self.btn_info.setToolTip("Informations sur ce logiciel")
        self.btn_info.setObjectName("btn_informations")
        self.btn_info.clicked.connect(self.afficher_information)

        self.controles_source.addWidget(self.btn_fichier)
        self.controles_source.addWidget(self.nom_fichier)
        self.controles_source.addWidget(self.btn_source)
        self.controles_source.addWidget(self.btn_info)

        # LECTURE
        self.controles_lecture = QtWidgets.QGridLayout()
        self.controles_lecture.setObjectName("controles_lecture")
        self.controles_lecture.setAlignment(QtCore.Qt.AlignCenter)

        self.btn_lecture = QtWidgets.QPushButton("Lire")
        self.btn_lecture.setObjectName("btn_lecture")
        self.btn_lecture.clicked.connect(self.threadVideo.mettreEnMarche)
        self.btn_lecture.setIcon(QIcon("play.ico"))
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.setObjectName("btn_pause")
        self.btn_pause.clicked.connect(self.threadVideo.mettreEnPause)
        self.btn_pause.setIcon(QIcon("pause.ico"))
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.clicked.connect(self.threadVideo.arreter)
        self.btn_stop.setIcon(QIcon("stop.ico"))

        self.controles_lecture.addWidget(self.btn_lecture, 0, 0)
        self.controles_lecture.addWidget(self.btn_pause, 0, 1)
        self.controles_lecture.addWidget(self.btn_stop, 1, 0, 1, 2)

        # FILTRES
        self.controles_filtres = QtWidgets.QGroupBox("Filtres de détection")
        self.controles_filtres.setObjectName("controles_filtres")

        self.layout_filtres = QtWidgets.QHBoxLayout()
        self.controles_filtres.setLayout(self.layout_filtres)
        self.layout_filtres.setAlignment(QtCore.Qt.AlignCenter)

        self.tous_filtre = QtWidgets.QCheckBox("Tous")
        self.tous_filtre.setObjectName("tous_filtre")
        self.tous_filtre.stateChanged.connect(self.actionnerTousLesFiltres)
        self.filtres = QtWidgets.QGridLayout()
        self.filtres.setSpacing(10)
        self.filtres.setObjectName("filtres")
        self.construireCheckboxesFiltres()

        self.layout_filtres.addWidget(self.tous_filtre)
        self.layout_filtres.addLayout(self.filtres)

        # PLACEMENT CONTROLES
        self.layout_conteneur_controles.addLayout(self.controles_source, 1)
        self.layout_conteneur_controles.addLayout(self.controles_lecture, 1)
        self.layout_conteneur_controles.addWidget(self.controles_filtres, 5)

        self.sections.addWidget(self.conteneur_video)
        self.sections.addWidget(self.conteneur_controles)

        self.centralwidget.setLayout(self.sections)

    def construireCheckboxesFiltres(self):
        self.checkboxes_filtres = []
        index_filtre = 0

        for i in range(LIGNES_FILTRES):
            for j in range(COLONNES_FILTRES):
                if index_filtre >= len(FILTRES):
                    return
                # x_offset = 90 * (j + 1)
                # y_offset = 25 * (i + 1)
                nouveau_filtre = QtWidgets.QCheckBox(FILTRES[index_filtre])
                # nouveau_filtre.setGeometry(QtCore.QRect(x_offset, y_offset, 85, 20))
                nouveau_filtre.setObjectName("filtre_" + FILTRES[index_filtre])

                # Si une checkbox change d'état, les filtres de détection sont modifiés
                nouveau_filtre.stateChanged.connect(
                    lambda etat, checkbox=nouveau_filtre: self.threadVideo.definirFiltres(etat, checkbox)
                )

                # Ajout dans le layout grille self.filtres
                self.filtres.addWidget(nouveau_filtre, i, j)

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
            self.threadVideo.video = cv.VideoCapture(fichier)
            self.threadpool.start(self.threadVideo)
            self.entrerEnChargement()

    def chargerWebcam(self):
        # print("chargerWebcam")
        camera = cv.VideoCapture(0, cv.CAP_DSHOW)
        self.threadVideo.video = camera
        self.nom_fichier.setText("")
        self.threadpool.start(self.threadVideo)
        self.entrerEnChargement()

    def entrerEnChargement(self):
        self.frame_video.setText("Chargement en cours...Merci de patienter !")
        self.btn_source.setDisabled(True)
        self.btn_fichier.setDisabled(True)
        self.gif_chargement.setVisible(True)
        self.gif_chargement.setMovie(self.gif)
        self.gif.start()

    @QtCore.pyqtSlot(np.ndarray)
    def rafraichirFrameVideo(self, frame):
        pixmap = self.convertirVersPixmap(frame)
        self.frame_video.setPixmap(pixmap)
        self.btn_fichier.setDisabled(False)
        self.btn_source.setDisabled(False)
        self.gif_chargement.setVisible(False)

    # Convertir une frame de VideoCapture de OpenCV vers une Pixmap utilisable par pyQt5
    def convertirVersPixmap(self, image_cv):
        image_rgb = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)
        hauteur, largeur, canaux = image_rgb.shape
        octets_par_ligne = canaux * largeur
        image_format_qt = QtGui.QImage(image_rgb.data, largeur, hauteur, octets_par_ligne, QtGui.QImage.Format_RGB888)
        image_redimensionnee = image_format_qt.scaled(self.conteneur_video.width(), self.conteneur_video.height(), QtCore.Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(image_redimensionnee)

    # CONTROLES DES FILTRES
    def actionnerTousLesFiltres(self, etat):
        coche = True if QtCore.Qt.Checked == etat else False
        for filtre in self.checkboxes_filtres:
            filtre.setChecked(coche)

    def afficher_information(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Logiciel fait par Hy-Vong Georges Dit Rap et Guillaume d'Albignac")
        msg.setInformativeText("| © 2021 | Cégep de Matane |")
        msg.setWindowTitle("Informations sur Dash Detect")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()


# Les signaux sont définis dans une classe séparée car il faut qu'il soient lancé par un QObject
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

        self.filtres = {}

    @QtCore.pyqtSlot()
    def run(self):
        while True:
            if not self.en_pause:
                self.detecterObjets()
            if cv.waitKey(1) & self.est_arrete:
                break

        # Libération de la mémoire et destruction de la fênetre
        self.video.release()
        self.est_arrete = False

    def detecterObjets(self):
        # Début de la lecture du temps pour les fps et lecture du flux
        temps_debut = time.time()

        # Analyse et détection des objets image par image
        ret, frame = self.video.read()

        # Syntaxe à utiliser pour définir les objets dans de la détection sélective
        # filtres = {
        #     'person': 'valid'
        # }
        detected_image, detections = self.detecteur.detectObjectsFromImage(custom_objects=self.filtres,
                                                                           input_image=frame,
                                                                           input_type="array",
                                                                           output_type="array",
                                                                           minimum_percentage_probability=60)
        # Calcule et affchage des fps
        temps_fin = time.time()
        secondes = temps_fin - temps_debut
        # print(secondes)
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

    def definirFiltres(self, etat, checkbox):
        # Si la checkbox est décochée
        if etat == 0:
            self.filtres[checkbox.text()] = "invalid"
        # si la checkbox est cochée
        elif etat == 2:
            self.filtres[checkbox.text()] = "valid"


# Configuration pour eviter les erreur "CUBLAS_STATUS_ALLOC_FAILED"
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

application = QApplication(sys.argv)

# Définition du thème
fichier_style = QtCore.QFile(":/dark.qss")
fichier_style.open(QtCore.QFile.ReadOnly | QtCore.QFile.Text)
stream = QtCore.QTextStream(fichier_style)
application.setStyleSheet(stream.readAll())

window = MainWindow()
window.show()
sys.exit(application.exec_())
