import sys
import os
import traceback
import mmcv
import cv2
import qimage2ndarray
import numpy as np
import json
import globals
from PyQt5.QtCore import Qt, pyqtSlot, QRunnable, QObject, QThreadPool, pyqtSignal, QDir, QTimer, QSize
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QFileDialog, QMainWindow, QSlider, QStyle, QVBoxLayout, QSpinBox,
                             QHBoxLayout, QPushButton, QWidget, QLabel, QCheckBox, QComboBox, QProgressBar,
                             QProgressDialog, QTabWidget, QListWidget, QMessageBox)
from MMPose import mmpose_inference
from mmpose.apis import init_pose_model, vis_pose_tracking_result
from mmpose.datasets import DatasetInfo


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(list)
    progress = pyqtSignal(dict)


class Worker(QRunnable):
    def __init__(self, in_file, tracking_type, pose, bbox_thr, tracking_thr, kpt_radius, line_thickness):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.in_file = in_file
        self.tracking_type = tracking_type
        self.pose = pose
        self.bbox_thr = bbox_thr
        self.tracking_thr = tracking_thr
        self.kptradius = kpt_radius
        self.linethickness = line_thickness
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.progress_callback = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = mmpose_inference(self.progress_callback, self.in_file, self.tracking_type, self.pose, self.bbox_thr,
                                      self.tracking_thr, self.kptradius, self.linethickness)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Human pose estimation and tracking')

        widget = QWidget()

        self.video = [[], []]
        self.temp_video = []
        self.running = False
        self.loaded_jsons = [{'pose': '--',
                              'track': '--',
                              'bbox_thr': 0,
                              'kpt_thr': 0},
                             {'pose': '--',
                              'track': '--',
                              'bbox_thr': 0,
                              'kpt_thr': 0}]
        self.frame_label = QLabel()
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.display_video_stream)
        self.pause = True
        self.fps = 0
        self.length = 0
        self.frameindex = -1
        self.url = ""
        self.originalsize = QSize()

        self.inferenceresults = {"pose": "",
                                 "track": "",
                                 "video_path": "",
                                 "bbox_thr": 0,
                                 "kpt_thr": 0,
                                 "results": []}

        file_menu = self.menuBar().addMenu("&File")
        self.open_action = file_menu.addAction("Open video")
        self.open_action.triggered.connect(self.open_file)
        self.exit_action = file_menu.addAction("Exit")
        self.exit_action.triggered.connect(self.exit_call)
        self.about_action = self.menuBar().addAction("About")
        self.about_action.triggered.connect(self.about)

        # INFERENCE WIDGET CREATION
        inferencewidget = QWidget()
        inferencewidget.layout = QVBoxLayout(inferencewidget)
        inferencewidget.setMaximumWidth(300)
        inferencewidget.setMinimumWidth(300)
        # Pose Method Combobox
        posemethod = QWidget()
        posemethod.layout = QHBoxLayout(posemethod)
        self._poselabel = QLabel("Pose estimation method:")
        self._posecombo = QComboBox()
        self._posecombo.addItems(["Hourglass", "HRNet", "DEKR", "HigherHRNet"])
        self._posecombo.insertSeparator(2)
        self._posecombo.setEnabled(False)
        posemethod.layout.addWidget(self._poselabel)
        posemethod.layout.addWidget(self._posecombo)
        # Tracking Radio Button
        self._track = QCheckBox()
        self._track.setText("Enable pose tracking")
        self._track.stateChanged.connect(self.enable_tracking)
        self._track.setEnabled(False)
        # Tracking Method Combobox
        trackingmethod = QWidget()
        trackingmethod.layout = QHBoxLayout(trackingmethod)
        self._tracklabel = QLabel("Pose estimation method:")
        self._trackcombo = QComboBox()
        self._trackcombo.addItems(["Greedily by IoU", "Greedily by distance", "Greedily by OKS", "ByteTrack", "OC-SORT",
                                   "QDTrack"])
        self._trackcombo.insertSeparator(3)
        self._trackcombo.setEnabled(False)
        trackingmethod.layout.addWidget(self._tracklabel)
        trackingmethod.layout.addWidget(self._trackcombo)
        # bbox_thr Slider
        bboxthrslider = QWidget()
        bboxthrslider.layout = QHBoxLayout(bboxthrslider)
        self._bboxslider = QSlider()
        self._bboxslider.setMaximum(100)
        self._bboxslider.setMinimum(0)
        self._bboxslider.setTickPosition(QSlider.TicksBelow)
        self._bboxslider.setValue(30)
        self._bboxslider.setOrientation(Qt.Horizontal)
        self._bboxslider.valueChanged.connect(self.bboxsliderchanged)
        self._bboxslider.setEnabled(False)
        self._bboxline = QLabel()
        self._bboxline.setFixedWidth(50)
        self._bboxline.setText(str(self._bboxslider.value() / 100) + "\n")
        bboxthrslider.layout.addWidget(self._bboxslider)
        bboxthrslider.layout.addSpacing(10)
        bboxthrslider.layout.addWidget(self._bboxline)
        self._bboxlabel = QLabel("Bounding box threshold for pose estimation")
        # kpt_thr Slider
        kptthrslider = QWidget()
        kptthrslider.layout = QHBoxLayout(kptthrslider)
        self._kptslider = QSlider()
        self._kptslider.setMaximum(100)
        self._kptslider.setMinimum(0)
        self._kptslider.setTickPosition(QSlider.TicksBelow)
        self._kptslider.setValue(30)
        self._kptslider.setOrientation(Qt.Horizontal)
        self._kptslider.valueChanged.connect(self.kptsliderchanged)
        self._kptslider.setEnabled(False)
        self._kptline = QLabel()
        self._kptline.setFixedWidth(50)
        self._kptline.setText(str(self._kptslider.value() / 100) + "\n")
        kptthrslider.layout.addWidget(self._kptslider)
        kptthrslider.layout.addSpacing(10)
        kptthrslider.layout.addWidget(self._kptline)
        self._kptlabel = QLabel("Keypoint threshold for pose tracking")
        # Keypoint radius
        kptradius = QWidget()
        kptradius.layout = QHBoxLayout(kptradius)
        self._kptradius = QSpinBox()
        self._kptradius.setRange(1, 10)
        self._kptradius.setValue(4)
        self._kptradius.setMaximumWidth(50)
        self._kptradius.setEnabled(False)
        self._kptradiuslabel = QLabel("Keypoint visualization radius:")
        kptradius.layout.addWidget(self._kptradiuslabel)
        kptradius.layout.addWidget(self._kptradius)
        # Line thickness
        linethickness = QWidget()
        linethickness.layout = QHBoxLayout(linethickness)
        self._linethickness = QSpinBox()
        self._linethickness.setRange(1, 5)
        self._linethickness.setValue(1)
        self._linethickness.setMaximumWidth(50)
        self._linethickness.setEnabled(False)
        self._linethicknesslabel = QLabel("Pose visualization thickness:")
        linethickness.layout.addWidget(self._linethicknesslabel)
        linethickness.layout.addWidget(self._linethickness)
        # Show Radio Button
        self._show = QCheckBox()
        self._show.setText("Show inference in real time")
        self._show.setEnabled(False)
        # Progress Bar
        self._progressbar = QProgressBar()
        self._progressbar.setFormat("%v/%m (%p%)")
        self._progressbar.setValue(0)
        self._progressbar.setAlignment(Qt.AlignCenter)
        self._progressbar.setTextVisible(False)
        # Start Stop Buttons
        startstop = QWidget()
        startstop.layout = QHBoxLayout(startstop)
        self._buttonstart = QPushButton("Run Inference")
        self._buttonstart.clicked.connect(self.start_inference)
        self._buttonstart.setEnabled(False)
        self._buttonstop = QPushButton("Stop Inference")
        self._buttonstop.setEnabled(False)
        self._buttonstop.clicked.connect(self.stop_inference)
        startstop.layout.addWidget(self._buttonstart)
        startstop.layout.addSpacing(10)
        startstop.layout.addWidget(self._buttonstop)
        # Put It All Together
        inferencewidget.layout.addWidget(posemethod)
        inferencewidget.layout.addWidget(self._track)
        inferencewidget.layout.addWidget(trackingmethod)
        inferencewidget.layout.addWidget(self._bboxlabel)
        inferencewidget.layout.addWidget(bboxthrslider)
        inferencewidget.layout.addWidget(self._kptlabel)
        inferencewidget.layout.addWidget(kptthrslider)
        inferencewidget.layout.addWidget(kptradius)
        inferencewidget.layout.addWidget(linethickness)
        inferencewidget.layout.addWidget(self._show)
        inferencewidget.layout.addWidget(self._progressbar)
        inferencewidget.layout.addWidget(startstop)

        # Visualization widget
        visualizationwidget = QWidget()
        visualizationwidget.layout = QVBoxLayout(visualizationwidget)
        self._resultlist = QListWidget()
        self._resultlist.itemClicked.connect(self.change_json)
        visualizationwidget.layout.addWidget(self._resultlist)
        self._infolabel1 = QLabel()
        self._infolabel1.setText('Result information')
        self._infolabel1.setAlignment(Qt.AlignCenter)
        self._infolabel2 = QLabel()
        self._infolabel2.setText('Pose method: --\n'
                                 'Tracking method: --\n'
                                 'Pose bounding box threshold: 0\n'
                                 'Tracking keypoint threshold: 0')
        visualizationwidget.layout.addWidget(self._infolabel1)
        visualizationwidget.layout.addWidget(self._infolabel2)
        # Keypoint radius
        kptradius1 = QWidget()
        kptradius1.layout = QHBoxLayout(kptradius1)
        self._kptradius1 = QSpinBox()
        self._kptradius1.setRange(1, 10)
        self._kptradius1.setValue(4)
        self._kptradius1.setMaximumWidth(50)
        self._kptradius1.setEnabled(False)
        self._kptradiuslabel1 = QLabel("Keypoint visualization radius:")
        kptradius1.layout.addWidget(self._kptradiuslabel1)
        kptradius1.layout.addWidget(self._kptradius1)
        # Line thickness
        linethickness1 = QWidget()
        linethickness1.layout = QHBoxLayout(linethickness1)
        self._linethickness1 = QSpinBox()
        self._linethickness1.setRange(1, 5)
        self._linethickness1.setValue(1)
        self._linethickness1.setMaximumWidth(50)
        self._linethickness1.setEnabled(False)
        self._linethicknesslabel1 = QLabel("Pose visualization thickness:")
        linethickness1.layout.addWidget(self._linethicknesslabel1)
        linethickness1.layout.addWidget(self._linethickness1)
        visualizationwidget.layout.addWidget(kptradius1)
        visualizationwidget.layout.addWidget(linethickness1)
        # JSON Load and Save Buttons
        jsonbuttons = QWidget()
        jsonbuttons.layout = QHBoxLayout(jsonbuttons)
        self._buttonopenjson = QPushButton("Add JSON file")
        self._buttonopenjson.setEnabled(False)
        self._buttonopenjson.clicked.connect(self.open_json)
        self._buttonremovejson = QPushButton("Remove file")
        self._buttonremovejson.setEnabled(False)
        self._buttonremovejson.clicked.connect(self.remove_json)
        jsonbuttons.layout.addWidget(self._buttonopenjson)
        jsonbuttons.layout.addSpacing(10)
        jsonbuttons.layout.addWidget(self._buttonremovejson)
        visualizationwidget.layout.addWidget(jsonbuttons)
        # Save Buttons
        savebuttons = QWidget()
        savebuttons.layout = QHBoxLayout(savebuttons)
        self._buttonsavejson = QPushButton("Save JSON result")
        self._buttonsavejson.setEnabled(False)
        self._buttonsavejson.clicked.connect(self.save_json)
        self._buttonsavevideo = QPushButton("Save video")
        self._buttonsavevideo.setEnabled(False)
        self._buttonsavevideo.clicked.connect(self.save_video)
        savebuttons.layout.addWidget(self._buttonsavejson)
        savebuttons.layout.addSpacing(10)
        savebuttons.layout.addWidget(self._buttonsavevideo)
        visualizationwidget.layout.addWidget(savebuttons)

        # Create tab widget
        self.tabwidget = QTabWidget()
        self.tabwidget.addTab(inferencewidget, 'Run inference')
        self.tabwidget.addTab(visualizationwidget, 'Visualize results')

        rightside = QWidget()
        rightside.layout = QHBoxLayout(rightside)
        rightside.layout.addWidget(self.tabwidget)

        videowidget = QWidget()
        self._playbutton = QPushButton()
        self._playbutton.setEnabled(False)
        self._playbutton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self._playbutton.clicked.connect(self.play)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.sliderMoved.connect(self.set_position)
        self._slider.setEnabled(False)
        self._durationlabel = QLabel()
        self._durationlabel.setText("-- / --")
        self._durationlabel.setFixedHeight(50)

        controlwidget = QWidget()
        controlwidget.layout = QHBoxLayout()
        controlwidget.layout.addWidget(self._playbutton)
        controlwidget.layout.addWidget(self._slider)
        controlwidget.layout.addWidget(self._durationlabel)
        controlwidget.setFixedHeight(50)

        videowidget.layout = QVBoxLayout(videowidget)
        videowidget.layout.addWidget(self.frame_label)
        videowidget.layout.addLayout(controlwidget.layout)


        # FINAL WIDGET CREATION
        widget.layout = QHBoxLayout(widget)
        widget.layout.addWidget(videowidget)
        widget.layout.addWidget(rightside)
        self.setCentralWidget(widget)

        available_geometry = self.screen().availableGeometry()
        self.setFixedSize(round(available_geometry.width() / 2), round(available_geometry.height() / 2))
        self.video_size = QSize(round(available_geometry.width() / 2) - 400,
                                round(available_geometry.height() / 2) - 60)
        image = np.zeros((self.video_size.height(), self.video_size.width(), 3), np.uint8)
        self.frame_label.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(image)))

        self.threadpool = QThreadPool()

    def set_position(self, position):
        if self._show.isChecked():
            self._show.setChecked(False)
        if not position == 0:
            self.frameindex = position - 1
        else:
            self.frameindex = -1
        self.display_video_stream()

    def progress_fn(self, res):
        frame_id = res["frame_id"]
        image = res["image"]
        self.video[1][frame_id] = image
        self._progressbar.setValue(frame_id)
        if self._show.isChecked():
            self.frameindex = frame_id - 1
            self.display_video_stream()

    def handle_result(self, s):
        self.inferenceresults["results"] = s

    def thread_complete(self):
        self._progressbar.setValue(self._progressbar.value() + 1)
        self._buttonstop.setEnabled(False)
        self._buttonstart.setEnabled(True)
        self._resultlist.item(1).setHidden(False)
        self.running = False
        self._resultlist.setCurrentRow(1)
        self.change_json()

    def about(self):
        QMessageBox.about(self, 'About', "                Application made by Bc. Antonín Čech\n"
                                         "as a part of a master's thesis on human pose estimation and tracking   \n"
                                         "               at FNSPE CTU in Prague, Czech Republic")

    def bboxsliderchanged(self):
        self._bboxline.setText(str(self._bboxslider.value() / 100))

    def kptsliderchanged(self):
        self._kptline.setText(str(self._kptslider.value() / 100))

    def enable_tracking(self):
        if self._trackcombo.isEnabled():
            self._trackcombo.setEnabled(False)
            self._kptslider.setEnabled(False)
        else:
            self._trackcombo.setEnabled(True)
            self._kptslider.setEnabled(True)

    def start_inference(self):
        self._buttonstart.setEnabled(False)
        globals.kill_thread = False
        tracking = self._trackcombo.currentText() if self._track.isChecked() else "Tracking disabled"
        self.inferenceresults["pose"] = self._posecombo.currentText()
        self.inferenceresults["track"] = tracking
        self.inferenceresults["video_path"] = self.url
        self.inferenceresults["bbox_thr"] = self._bboxslider.value() / 100
        self.inferenceresults["kpt_thr"] = self._kptslider.value() / 100
        self.loaded_jsons[1]['pose'] = self.inferenceresults["pose"]
        self.loaded_jsons[1]['track'] = self.inferenceresults["track"]
        self.loaded_jsons[1]['bbox_thr'] = self.inferenceresults["bbox_thr"]
        self.loaded_jsons[1]['kpt_thr'] = self.inferenceresults["kpt_thr"]
        self.video[1] = self.video[0].copy()
        self.worker = Worker(self.url, tracking, self._posecombo.currentText(),
                        self._bboxslider.value() / 100, self._kptslider.value() / 100, self._kptradius.value(),
                        self._linethickness.value())
        self.worker.signals.result.connect(self.handle_result)
        self.worker.signals.finished.connect(self.thread_complete)
        self.worker.signals.progress.connect(self.progress_fn)
        self._buttonstop.setEnabled(True)
        self.running = True

        # Execute
        self.threadpool.start(self.worker)

    def stop_inference(self):
        globals.kill_thread = True

    def save_json(self):
        videoname, _ = os.path.splitext(os.path.basename(self.url))
        fileName, _ = QFileDialog.getSaveFileName(self, "Save JSON", os.path.dirname(self.url) + '/' + videoname
                                                  + "_result.json",
                                                  "JSON Files (*.json);;All Files")

        if fileName != '':
            with open(fileName, "w") as outfile:
                json.dump(self.inferenceresults, outfile)

    def save_video(self):
        videoname, _ = os.path.splitext(os.path.basename(self.url))
        fileName, _ = QFileDialog.getSaveFileName(self, "Save video", os.path.dirname(self.url) + '/' + videoname
                                                  + "_result.avi",
                                                  "Video Files (*.avi);;All Files")
        if fileName != '':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter(
                fileName, fourcc, self.fps, (self.originalsize.width(), self.originalsize.height()))
            item = self._resultlist.currentRow()
            self.progress = QProgressDialog("Saving video...", "Cancel", 0, self.length, self)
            self.progress.setWindowModality(Qt.WindowModal)
            self.progress.show()
            for i in range(self.length):
                self.progress.setValue(i + 1)
                if self.progress.wasCanceled():
                    break
                videoWriter.write(self.video[item][i])
            videoWriter.release()

    def open_file(self):
        self.ensure_stopped()
        self.video = [[], []]
        self.fps = 0
        self.length = 0
        self.frameindex = -1
        self._resultlist.clear()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open video", QDir.homePath())

        if fileName != '':
            video = mmcv.VideoReader(fileName)
            self.progress = QProgressDialog("Loading video...", "Cancel", 0, video.frame_cnt, self)
            self.progress.setWindowModality(Qt.WindowModal)
            self.progress.show()
            for i in range(video.frame_cnt):
                self.progress.setValue(i + 1)
                if self.progress.wasCanceled():
                    break
                self.video[0].append(video[i])
            self.fps = video.fps
            self.length = len(self.video[0])
            self.url = fileName
            self.originalsize = QSize(video.width, video.height)
            self._progressbar.setRange(0, self.length)
            self._progressbar.setTextVisible(True)
            self._playbutton.setEnabled(True)
            self._posecombo.setEnabled(True)
            self._track.setEnabled(True)
            self._bboxslider.setEnabled(True)
            self._kptradius.setEnabled(True)
            self._linethickness.setEnabled(True)
            self._kptradius1.setEnabled(True)
            self._linethickness1.setEnabled(True)
            self._show.setEnabled(True)
            self._buttonstart.setEnabled(True)
            self._buttonopenjson.setEnabled(True)
            self._resultlist.addItem("Original video")
            self._resultlist.addItem("Inference result")
            self._resultlist.item(1).setHidden(True)
            self._resultlist.setCurrentRow(0)
            self._slider.setEnabled(True)
            self._slider.setRange(0, self.length - 1)
            self._progressbar.setValue(0)
            self.frameindex = -1
            self.display_video_stream()

    def open_json(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open JSON result", QDir.homePath())

        if fileName != '':
            with open(fileName, "r") as infile:
                file_contents = json.loads(infile.read())
            self.tabwidget.setCurrentIndex(1)
            videoname, _ = os.path.splitext(os.path.basename(file_contents["video_path"]))
            self._resultlist.addItem(videoname)
            num = self._resultlist.count()
            file_contents["num"] = num
            self.loaded_jsons.append(file_contents)
            if file_contents["pose"] == "Hourglass":
                pose_config = './configs/hourglass.py'
                pose_checkpoint = './checkpoints/hourglass.pth'
            elif file_contents["pose"] == 'HRNet':
                pose_config = './configs/hrnet.py'
                pose_checkpoint = './checkpoints/hrnet.pth'
            elif file_contents["pose"] == "DEKR":
                pose_config = './configs/dekr.py'
                pose_checkpoint = './checkpoints/dekr.pth'
            elif file_contents["pose"] == "HigherHRNet":
                pose_config = './configs/higher_hrnet.py'
                pose_checkpoint = './checkpoints/higher_hrnet.pth'

            self.progress = QProgressDialog("Visualizing results...", "Cancel", 0, len(file_contents["results"]), self)
            self.progress.setWindowModality(Qt.WindowModal)
            self.progress.show()

            pose_model = init_pose_model(
                pose_config, pose_checkpoint, device='cuda:0')

            dataset = pose_model.cfg.data['test']['type']
            dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
            dataset_info = DatasetInfo(dataset_info)

            vid = self.video[0].copy()
            self.video.append(vid)

            for i in range(len(file_contents["results"])):
                self.progress.setValue(i + 1)
                if self.progress.wasCanceled():
                    break
                self.video[num - 1][i] = vis_pose_tracking_result(
                                                     pose_model,
                                                     self.video[0][i],
                                                     file_contents["results"][i],
                                                     radius=self._kptradius1.value(),
                                                     thickness=self._linethickness1.value(),
                                                     dataset=dataset,
                                                     dataset_info=dataset_info,
                                                     kpt_score_thr=0.3,
                                                     show=False)
            self._resultlist.setCurrentRow(num - 1)
            self._infolabel2.setText(f'Pose method: {file_contents["pose"]}\n'
                                     f'Tracking method: {file_contents["track"]}\n'
                                     f'Pose bounding box threshold: {file_contents["bbox_thr"]}\n'
                                     f'Tracking keypoint threshold: {file_contents["kpt_thr"]}')
            self.frameindex = self.frameindex - 1
            self.display_video_stream()

    def remove_json(self):
        item = self._resultlist.currentRow()
        self._resultlist.takeItem(item)
        del self.loaded_jsons[item]
        del self.video[item]
        self._resultlist.setCurrentRow(0)
        self._buttonremovejson.setEnabled(False)
        self._infolabel2.setText('Pose method: --\n'
                                 'Tracking method: --\n'
                                 'Pose bounding box threshold: --\n'
                                 'Tracking keypoint threshold: --')

    def change_json(self):
        item = self._resultlist.currentRow()
        self._infolabel2.setText(f'Pose method: {self.loaded_jsons[item]["pose"]}\n'
                                 f'Tracking method: {self.loaded_jsons[item]["track"]}\n'
                                 f'Pose bounding box threshold: {self.loaded_jsons[item]["bbox_thr"]}\n'
                                 f'Tracking keypoint threshold: {self.loaded_jsons[item]["kpt_thr"]}')
        self.frameindex = self.frameindex - 1
        self.display_video_stream()
        self._linethickness1.setValue(self._linethickness.value())
        self._kptradius1.setValue(self._kptradius.value())
        if item == 0:
            self._buttonremovejson.setEnabled(False)
            self._buttonsavejson.setEnabled(False)
            self._buttonsavevideo.setEnabled(False)
        elif item == 1:
            self._buttonremovejson.setEnabled(False)
            self._buttonsavejson.setEnabled(True)
            self._buttonsavevideo.setEnabled(True)
        else:
            self._buttonremovejson.setEnabled(True)
            self._buttonsavejson.setEnabled(False)
            self._buttonsavevideo.setEnabled(True)

    def exit_call(self):
        self.ensure_stopped()
        self.close()

    def play(self):
        if self._show.isChecked():
            self._show.setChecked(False)
        if self.frameindex == self.length - 1:
            self.frameindex = -1
            self.display_video_stream()

        if not self.pause:
            self.frame_timer.stop()
            self._playbutton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.frame_timer.start(int(1000 // self.fps))
            self._playbutton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause))

        self.pause = not self.pause

    def display_video_stream(self):
        if self.frameindex == self.length - 1:
            self.frame_timer.stop()
            self._playbutton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))
            self.pause = True
        else:
            self.frameindex += 1
            self._slider.setValue(self.frameindex)
            self._durationlabel.setText(str(self.frameindex + 1) + " / " + str(self.length))

            if self.running:
                frame = cv2.cvtColor(self.video[1][self.frameindex], cv2.COLOR_BGR2RGB)
            else:
                item = self._resultlist.currentRow()
                frame = cv2.cvtColor(self.video[item][self.frameindex], cv2.COLOR_BGR2RGB)

            h, w = frame.shape[:2]
            sw, sh = self.video_size.width(), self.video_size.height()
            padColor = 0

            # interpolation method
            if h > sh or w > sw:  # shrinking image
                interp = cv2.INTER_AREA
            else:  # stretching image
                interp = cv2.INTER_CUBIC

            # aspect ratio of image
            aspect = w / h  # if on Python 2, you might need to cast as a float: float(w)/h

            # compute scaling and pad sizing
            if aspect > 1:  # horizontal image
                new_w = sw
                new_h = np.round(new_w / aspect).astype(int)
                pad_vert = (sh - new_h) / 2
                pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
                pad_left, pad_right = 0, 0
            elif aspect < 1:  # vertical image
                new_h = sh
                new_w = np.round(new_h * aspect).astype(int)
                pad_horz = (sw - new_w) / 2
                pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
                pad_top, pad_bot = 0, 0
            else:  # square image
                new_h, new_w = sh, sw
                pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

            # set pad color
            if len(frame.shape) == 3 and not isinstance(padColor, (
            list, tuple, np.ndarray)):  # color image but only one color provided
                padColor = [padColor] * 3

            # scale and pad
            scaled_img = cv2.resize(frame, (new_w, new_h), interpolation=interp)
            scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                            borderType=cv2.BORDER_CONSTANT, value=padColor)

            image = qimage2ndarray.array2qimage(scaled_img)
            self.frame_label.setPixmap(QPixmap.fromImage(image))

    def ensure_stopped(self):
        if not self.pause:
            self.frame_timer.stop()
            self.play_pause_button.setText("Play")
            self._playbutton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))


if __name__ == '__main__':
    globals.initialize()
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
