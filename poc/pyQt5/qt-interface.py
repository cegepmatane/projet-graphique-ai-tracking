import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import cv2 as cv
from imutils.video import VideoStream


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
        self.setGeometry(100, 100, 1300, 800)
        self.setWindowTitle("Dash Detect PyQt5 GUI PoC")

        self.creerGui()

    def creerGui(self):
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 1280, 771))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")

        self.sections = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.sections.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.sections.setContentsMargins(0, 0, 0, 0)
        self.sections.setSpacing(0)
        self.sections.setObjectName("sections")

        self.conteneur_video = QtWidgets.QGroupBox(self.verticalLayoutWidget)
        self.conteneur_video.setMaximumSize(QtCore.QSize(1280, 600))
        self.conteneur_video.setObjectName("conteneur_video")

        self.video = QtWidgets.QLabel(self.conteneur_video)
        self.video.setGeometry(QtCore.QRect(160, 20, 871, 541))
        # self.video.setPixmap(QtGui.QPixmap("./fgpvtnw0zbz11.jpg"))
        self.video.setScaledContents(True)
        self.video.setObjectName("video")

        self.conteneur_controles = QtWidgets.QGroupBox(self.verticalLayoutWidget)
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
        self.btn_pause = QtWidgets.QPushButton(self.controles_lecture)
        self.btn_pause.setGeometry(QtCore.QRect(90, 20, 61, 61))
        self.btn_pause.setObjectName("btn_pause")
        self.btn_stop = QtWidgets.QPushButton(self.controles_lecture)
        self.btn_stop.setGeometry(QtCore.QRect(10, 90, 141, 31))
        self.btn_stop.setObjectName("btn_stop")

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
        self.setCentralWidget(self.centralwidget)

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

    def naviguerFichiers(self):
        nom_fichier, _ = QtWidgets.QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                "All Files (*);;Python Files (*.py)")
        if nom_fichier:
            print(nom_fichier)

    def actionnerTousLesFiltres(self, etat):
        checked = True if QtCore.Qt.Checked == etat else False
        for filtre in self.checkboxes_filtres:
            filtre.setChecked(checked)


application = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(application.exec_())
