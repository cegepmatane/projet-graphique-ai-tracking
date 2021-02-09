import tkinter as tk
from PIL import ImageTk, Image
import threading
import datetime
import cv2 as cv
import os
import argparse
from imutils.video import VideoStream


def buildWidget(parent, w_type, text, side, var=None):
    widget=None
    if w_type == "button":
        widget = tk.Button(parent, text=text)
    else:
        if w_type == "checkbox":
            widget = tk.Checkbutton(parent, text=text, variable=var)
    widget.pack(side=side, fill="both", padx=10, pady=10)
    return widget


class LiveFeed(tk.Frame):
    def __init__(self, window, outputPath):
        super().__init__(window)
        self.pack()

        self.window = window
        self.window.geometry("1366x768")
        self.window.resizable(width=False, height=False)
        # self.window.minsize(800, 600)
        self.window.configure(bg="#333333")
        self.window.title("Dash Detect GUI PoC")
        self.window.protocol("WM_DELETE_WINDOW", self.onClose)
        # self.window.bind("<Configure>", self.enforce_aspect_ratio)

        self.camera = VideoStream().start()
        self.outputPath = outputPath

        self.aspect_ratio = 16.0/9.0
        self.video_frame = None
        self.panel = None
        self.isClosing = False

        # Definition du thread pour la video
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        self.controlPanel = ControlPanel(self.window)

    def videoLoop(self):
        try:
            while not self.stopEvent.is_set():
                self.video_frame = self.camera.read()
                # self.video_frame = resize(self.video_frame, width=1000)

                image = cv.cvtColor(self.video_frame, cv.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                # if the panel is None, we need to initialize it
                if self.panel is None:
                    self.panel = tk.Label(image=image, bg="#222222")
                    self.panel.image = image
                    self.panel.pack(fill="both", padx=10, pady=10)
                    self.controlPanel.pack(fill="both", padx=10, pady=10)
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image
        except RuntimeError:
            print("[INFO] caught a RuntimeError")

    def takeSnapshot(self):
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y%m%d_%H-%M-%S"))
        picture = os.path.sep.join((self.outputPath, filename))
        # save the file
        cv.imwrite(picture, self.video_frame.copy())
        print("[INFO] saved {}".format(filename))

    def onClose(self):
        print("[INFO] closing...")
        self.stopEvent.set()
        self.camera.stop()
        self.window.quit()


class ControlPanel(tk.Frame):
    def __init__(self, window):
        super().__init__(window)

        self.window = window
        self.source_section = None
        self.livefeed_button = None
        self.file_button = None
        self.video_controls_section = None
        self.play_button = None
        self.pause_button = None
        self.stop_button = None
        self.filters_section = None
        self.all_filters_check = None
        self.filters = []

        self.aspect_ratio = 16.0/9.0

        self.buildGui()

    def buildGui(self):

        # Construction des sections
        # Section de sélection de la source vidéo
        self.source_section = tk.Frame(self, width=100, height=5, bg="purple")
        self.source_section.pack(side="left", fill="both")
        self.livefeed_button = buildWidget(self.source_section, "button", "Choisir la caméra", "top")
        self.file_button = buildWidget(self.source_section, "button", "Fichier", "top")

        self.video_controls_section = tk.Frame(self, width=100, height=5, bg="yellow")
        self.video_controls_section.pack(side="left", fill="both")
        play_pause_group = tk.Frame(self.video_controls_section, bg="yellow")
        self.play_button = buildWidget(play_pause_group, "button", "Lire", "left")
        self.pause_button = buildWidget(play_pause_group, "button", "Pause", "right")
        play_pause_group.pack()
        self.stop_button = buildWidget(self.video_controls_section, "button", "Stop", "top")

        self.filters_section = tk.Frame(self, height=5, bg="green")
        self.filters_section.pack(side="left", fill="both", expand="yes")
        all_filters_group = tk.Frame(self.filters_section, bg="green")
        other_filters_group = tk.Frame(self.filters_section, bg="cyan")
        var1 = "Yo"
        self.all_filters_check = buildWidget(all_filters_group, "checkbox", "Tous", "top", var1)
        all_filters_group.pack(side="left", fill="y")
        other_filters_group.pack(side="right", fill="both", expand="True")


# Recuperation des arguments au lancement en CLI
ap = argparse.ArgumentParser()
ap.add_argument(
    "-o",
    "--output",
    required=False,
    default=os.getcwd(),
    help="path to output directory to store snapshots"
)
args = vars(ap.parse_args())

root = tk.Tk()

livefeed = LiveFeed(root, args["output"])
livefeed.mainloop()