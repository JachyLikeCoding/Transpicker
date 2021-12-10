import argparse
import os
import sys
import time
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtG
import matplotlib
import matplotlib.backends.backend_qt5agg as plt_qtbackend
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QFontMetrics
import boxmanagertoolbar
import coord_io
import read_image
from matplotlib.patches import Rectangle
# from denoise import filter_single_image


class MyRectangle:
    def __init__(self, xy, width, height, angle=0.0, est_size=None, **kwargs):
        self.confidence = None
        self.est_size = est_size
        self.xy = xy
        self.width = width
        self.height = height
        self.angle = angle
        self.rectInstance = None
        self.kwargs = kwargs
        # super(MyRectangle, self).__init__()

    def getRect(self):
        if self.rectInstance is None:
            self.rectInstance = Rectangle(self.xy, self.width, self.height, self.angle, **self.kwargs)
        return self.rectInstance

    def set_confidence(self, confidence):
        self.confidence = confidence


argparser = argparse.ArgumentParser(
    description="Train and validate TransPicker on any dataset.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

argparser.add_argument("-i", "--image_dir", help="Path to image directory.")

argparser.add_argument("-b", "--box_dir", help="Path to box directory.")

# argparser.add_argument("--wildcard", help="Wildcard for filtering specific image formats. e.g. *_new_*.mrc")

argparser.add_argument("-s", "--box_size", type=int, help="Box size to display.", default=200)

argparser.add_argument("-p", "--prob_threshold", help="The display probability threshold of particles.", default=0)


class MainWindow(QtG.QMainWindow):
    def __init__(self, font, images_path=None, boxes_path=None, box_size=200, prob_threshold=0, parent=None):
        super(MainWindow, self).__init__(parent)
        # SETUP QT
        self.font = font
        self.setWindowTitle("Box manager")
        central_widget = QtG.QWidget(self)

        self.setCentralWidget(central_widget)

        # Center on screen
        resolution = QtG.QDesktopWidget().screenGeometry()
        self.move(
            (resolution.width() / 2) - (self.frameSize().width() / 2),
            (resolution.height() / 2) - (self.frameSize().height() / 2),
        )

        # Setup Menu
        close_action = QtG.QAction("Close", self)
        close_action.setShortcut("Ctrl+Q")
        close_action.setStatusTip("Leave the app.")
        close_action.triggered.connect(self.close_boxmanager)

        open_image_folder = QtG.QAction("Open image folder", self)
        open_image_folder.triggered.connect(self.open_image_folder)

        import_box_folder = QtG.QAction("Import box files", self)
        import_box_folder.triggered.connect(self.load_box_files)

        write_box_files = QtG.QAction("BOX", self)
        write_box_files.triggered.connect(self.write_box_files)

        write_star_files = QtG.QAction("STAR", self)
        write_star_files.triggered.connect(self.write_star_files)

        self.show_confidence_histogram_action = QtG.QAction("Confidence histogram", self)
        self.show_confidence_histogram_action.triggered.connect(self.show_confidence_histogram)
        self.show_confidence_histogram_action.setEnabled(False)

        self.show_size_distribution_action = QtG.QAction("Size distribution", self)
        self.show_size_distribution_action.triggered.connect(self.show_size_distribution)
        self.show_size_distribution_action.setEnabled(False)

        self.mainMenu = self.menuBar()
        self.fileMenu = self.mainMenu.addMenu("&File")
        self.fileMenu.addAction(open_image_folder)
        self.fileMenu.addAction(import_box_folder)

        write_menu = self.fileMenu.addMenu("&Write")
        write_menu.addAction(write_box_files)
        write_menu.addAction(write_star_files)

        self.fileMenu.addAction(close_action)
        self.image_folder = ""

        self.plotMenu = self.mainMenu.addMenu("&Plot")
        self.plotMenu.addAction(self.show_confidence_histogram_action)
        self.plotMenu.addAction(self.show_size_distribution_action)

        # Setup tree
        self.layout = QtG.QGridLayout(central_widget)
        self.setMenuBar(self.mainMenu)

        self.tree = QtG.QTreeWidget(self)
        self.tree.setHeaderHidden(True)
        self.layout.addWidget(self.tree, 0, 0, 1, 3)
        self.tree.currentItemChanged.connect(self._event_image_changed)
        line_counter = 1

        # Box size setup
        self.boxsize = int(box_size)
        self.boxsize_label = QtG.QLabel()
        self.boxsize_label.setText("Box size: ")
        self.layout.addWidget(self.boxsize_label, line_counter, 0)

        self.boxsize_line = QtG.QLineEdit()
        self.boxsize_line.setText(str(self.boxsize))
        self.boxsize_line.returnPressed.connect(self.box_size_changed)
        self.layout.addWidget(self.boxsize_line, line_counter, 1)

        self.button_set_box_size = QtG.QPushButton("Set")
        self.button_set_box_size.clicked.connect(self.box_size_changed)
        self.layout.addWidget(self.button_set_box_size, line_counter, 2)
        line_counter = line_counter + 1

        # Show estimated size
        self.use_estimated_size_label = QtG.QLabel()
        self.use_estimated_size_label.setText("Use estimated size:")
        self.use_estimated_size_label.setEnabled(False)
        self.layout.addWidget(self.use_estimated_size_label, line_counter, 0)

        self.use_estimated_size_checkbox = QtG.QCheckBox()
        self.layout.addWidget(self.use_estimated_size_checkbox, line_counter, 1)
        self.use_estimated_size_checkbox.stateChanged.connect(self.use_estimated_size_changed)
        self.use_estimated_size_checkbox.setEnabled(False)
        line_counter = line_counter + 1

        # Lower size
        self.lower_size_thresh = 0
        self.lower_size_thresh_label = QtG.QLabel()
        self.lower_size_thresh_label.setText("Minimum size: ")
        self.layout.addWidget(self.lower_size_thresh_label, line_counter, 0)
        self.lower_size_thresh_label.setEnabled(False)
        self.lower_size_thresh_slide = QtG.QSlider(QtCore.Qt.Horizontal)
        self.lower_size_thresh_slide.setMinimum(0)
        self.lower_size_thresh_slide.setMaximum(500)
        self.lower_size_thresh_slide.setValue(0)
        self.lower_size_thresh_slide.valueChanged.connect(self.lower_size_thresh_changed)

        self.lower_size_thresh_slide.setTickPosition(QtG.QSlider.TicksBelow)
        self.lower_size_thresh_slide.setTickInterval(1)
        self.lower_size_thresh_slide.setEnabled(False)
        self.layout.addWidget(self.lower_size_thresh_slide, line_counter, 1)

        self.lower_size_thresh_line = QtG.QLineEdit()
        self.lower_size_thresh_line.setText("0")
        self.lower_size_thresh_line.textChanged.connect(self.lower_size_label_changed)
        self.lower_size_thresh_line.setEnabled(False)
        self.layout.addWidget(self.lower_size_thresh_line, line_counter, 2)
        line_counter = line_counter + 1

        # Upper size threshold
        self.upper_size_thresh = 99999
        self.upper_size_thresh_label = QtG.QLabel()
        self.upper_size_thresh_label.setText("Maximum size: ")
        self.layout.addWidget(self.upper_size_thresh_label, line_counter, 0)
        self.upper_size_thresh_label.setEnabled(False)
        self.upper_size_thresh_slide = QtG.QSlider(QtCore.Qt.Horizontal)
        self.upper_size_thresh_slide.setMinimum(0)
        self.upper_size_thresh_slide.setMaximum(99999)
        self.upper_size_thresh_slide.setValue(99999)
        self.upper_size_thresh_slide.valueChanged.connect(self.upper_size_thresh_changed)

        self.upper_size_thresh_slide.setTickPosition(QtG.QSlider.TicksBelow)
        self.upper_size_thresh_slide.setTickInterval(1)
        self.upper_size_thresh_slide.setEnabled(False)
        self.layout.addWidget(self.upper_size_thresh_slide, line_counter, 1)

        self.upper_size_thresh_line = QtG.QLineEdit()
        self.upper_size_thresh_line.setText("99999")
        self.upper_size_thresh_line.textChanged.connect(self.upper_size_label_changed)
        self.upper_size_thresh_line.setEnabled(False)
        self.layout.addWidget(self.upper_size_thresh_line, line_counter, 2)
        line_counter = line_counter + 1

        # Confidence threshold setup
        self.current_conf_thresh = 0.3
        self.conf_thresh_label = QtG.QLabel()
        self.conf_thresh_label.setText("Confidence threshold:")
        self.layout.addWidget(self.conf_thresh_label, line_counter, 0)
        self.conf_thresh_label.setEnabled(False)
        self.conf_thresh_slide = QtG.QSlider(QtCore.Qt.Horizontal)
        self.conf_thresh_slide.setMinimum(0)
        self.conf_thresh_slide.setMaximum(100)
        self.conf_thresh_slide.setValue(30)
        self.conf_thresh_slide.valueChanged.connect(self.conf_thresh_changed)
        self.conf_thresh_slide.setTickPosition(QtG.QSlider.TicksBelow)
        self.conf_thresh_slide.setTickInterval(1)
        self.conf_thresh_slide.setEnabled(False)
        self.layout.addWidget(self.conf_thresh_slide, line_counter, 1)

        self.conf_thresh_line = QtG.QLineEdit()
        self.conf_thresh_line.setText("0.3")
        self.conf_thresh_line.textChanged.connect(self.conf_thresh_label_changed)
        self.conf_thresh_line.setEnabled(False)
        self.layout.addWidget(self.conf_thresh_line, line_counter, 2)
        line_counter = line_counter + 1

        # Low pass filter setup
        self.filter_freq = 0.1
        self.filter_label = QtG.QLabel()
        self.filter_label.setText("Low pass filter cut-off:")
        self.layout.addWidget(self.filter_label, line_counter, 0)

        self.filter_line = QtG.QLineEdit()
        self.filter_line.setText(str(self.filter_freq))
        self.layout.addWidget(self.filter_line, line_counter, 1)

        self.button_apply_filter = QtG.QPushButton("Apply")
        self.button_apply_filter.clicked.connect(self.apply_filter)
        self.button_apply_filter.setEnabled(False)
        self.layout.addWidget(self.button_apply_filter, line_counter, 2)

        # Show image selection
        self.show()
        self.box_dictionary = {}

        self.plot = None
        self.fig = None
        self.ax = None
        self.moving_box = None
        self.zoom_update = False
        self.doresizing = False
        self.current_image_path = None
        self.background_current = None
        self.unsaved_changes = False
        self.is_cbox = False
        self.toggle = False
        self.use_estimated_size = False
        self.prob_threshold = prob_threshold
        # self.wildcard = wildcard

        if images_path:
            img_loaded = self._open_image_folder(images_path)
            if img_loaded:
                self.button_apply_filter.setEnabled(True)
            if boxes_path:
                self._import_boxes(box_dir=boxes_path, keep=False)

    def close_boxmanager(self):
        if self.unsaved_changes:
            msg = "All loaded boxes are discarded. Are you sure?"
            reply = QtG.QMessageBox.question(
                self, "Message", msg, QtG.QMessageBox.Yes, QtG.QMessageBox.Cancel
            )
            if reply == QtG.QMessageBox.Cancel:
                return
        self.close()

    def _event_image_changed(self, root_tree_item):
        if (
                root_tree_item is not None
                and root_tree_item.childCount() == 0
                and self.current_image_path is not None
        ):
            self.current_tree_item = root_tree_item
            filename = root_tree_item.text(0)
            pure_filename = os.path.splitext(os.path.basename(self.current_image_path))[0]

            if pure_filename in self.box_dictionary:
                self.rectangles = self.box_dictionary[pure_filename]
                self.delete_all_patches(self.rectangles)
            else:
                self.rectangles = []
            prev_size = read_image.read_width_height(self.current_image_path)
            self.current_image_path = os.path.join(self.image_folder, str(filename))
            self.fig.canvas.set_window_title(os.path.basename(self.current_image_path))
            img = self.read_image(self.current_image_path)
            prev_size = prev_size[::-1]
            if prev_size == img.shape:
                self.im.set_data(img)
            else:
                self.im = self.ax.imshow(
                    img, origin="lower", cmap="gray", interpolation="Hanning"
                )

            self.plot.setWindowTitle(os.path.basename(self.current_image_path))
            self.fig.canvas.draw()
            self.background_current = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.background_orig = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.update_boxes_on_current_image()

    def lower_size_label_changed(self):
        try:
            new_value = int(float(self.lower_size_thresh_line.text()))
            upper_value = self.upper_size_thresh_slide.value()
            if new_value >= upper_value:
                return
        except ValueError:
            return
        self.lower_size_thresh_slide.setValue(new_value)

    def upper_size_label_changed(self):
        try:
            new_value = int(float(self.upper_size_thresh_line.text()))
            lower_value = self.lower_size_thresh_slide.value()
            if new_value <= lower_value:
                return
        except ValueError:
            return
        self.upper_size_thresh_slide.setValue(new_value)

    @pyqtSlot()
    def conf_thresh_label_changed(self):
        try:
            new_value = float(self.conf_thresh_line.text())
            if new_value > 1.0 or new_value < 0:
                return
        except ValueError:
            return
        self.current_conf_thresh = new_value
        self.conf_thresh_slide.setValue(new_value * 100)

    @pyqtSlot()
    def conf_thresh_changed(self):
        try:
            self.current_conf_thresh = float(self.conf_thresh_slide.value()) / 100
        except ValueError:
            return
        try:
            if (
                    np.abs(float(self.conf_thresh_line.text()) - self.current_conf_thresh)
                    >= 0.01
            ):
                self.conf_thresh_line.setText("" + str(self.current_conf_thresh))
        except ValueError:
            self.conf_thresh_line.setText("" + str(self.current_conf_thresh))

        self.update_boxes_on_current_image()
        self.fig.canvas.restore_region(self.background_current)
        self._draw_all_boxes()
        self.unsaved_changes = True
        self.update_tree_boxsizes()

    def upper_size_thresh_changed(self):
        self.upper_size_thresh = int(float(self.upper_size_thresh_slide.value()))
        self.upper_size_thresh_line.setText("" + str(self.upper_size_thresh))
        if self.upper_size_thresh <= self.lower_size_thresh:
            self.lower_size_thresh_slide.setValue(self.upper_size_thresh - 1)
        self.update_boxes_on_current_image()
        self.fig.canvas.restore_region(self.background_current)
        self._draw_all_boxes()
        self.unsaved_changes = True
        self.update_tree_boxsizes()

    def lower_size_thresh_changed(self):
        self.lower_size_thresh = int(float(self.lower_size_thresh_slide.value()))
        self.lower_size_thresh_line.setText("" + str(self.lower_size_thresh))
        if self.lower_size_thresh >= self.upper_size_thresh:
            self.upper_size_thresh_slide.setValue(self.lower_size_thresh + 1)
        self.update_boxes_on_current_image()
        self.fig.canvas.restore_region(self.background_current)
        self._draw_all_boxes()
        self.unsaved_changes = True
        self.update_tree_boxsizes()

    def box_size_changed(self):
        try:
            self.boxsize = int(float(self.boxsize_line.text()))
        except ValueError:
            return
        if self.boxsize >= 0:
            # for _, rectangles in self.box_dictionary.items():
            #    QtCore.QCoreApplication.instance().processEvents()
            # for rect in self.rectangles:
            #    if self.use_estimated_size:
            #        self.resize_box(rect, rect.est_size)
            #    else:
            #        self.resize_box(rect, self.boxsize)
            if self.background_current:
                self.update_boxes_on_current_image()
                self.fig.canvas.restore_region(self.background_current)
                self._draw_all_boxes()
                self.unsaved_changes = True

    def resize_box(self, rect, new_size):
        if rect.get_width() != new_size or rect.get_height() != new_size:
            height_diff = new_size - rect.get_height()
            width_diff = new_size - rect.get_width()
            newy = rect.get_y() - height_diff / 2
            newx = rect.get_x() - width_diff / 2
            rect.set_height(new_size)
            rect.set_width(new_size)
            rect.set_xy((newx, newy))

    @pyqtSlot()
    def use_estimated_size_changed(self):
        self.use_estimated_size = self.use_estimated_size_checkbox.isChecked()
        self.box_size_changed()
        self.button_set_box_size.setEnabled(not self.use_estimated_size)
        self.boxsize_line.setEnabled(not self.use_estimated_size)
        self.boxsize_label.setEnabled(not self.use_estimated_size)

        self.upper_size_thresh_line.setEnabled(self.use_estimated_size)
        self.upper_size_thresh_slide.setEnabled(self.use_estimated_size)

        self.lower_size_thresh_line.setEnabled(self.use_estimated_size)
        self.lower_size_thresh_slide.setEnabled(self.use_estimated_size)

    def apply_filter(self):
        try:
            self.filter_freq = float(self.filter_line.text())
        except ValueError:
            return
        if 0.5 > self.filter_freq >= 0:
            img = filter_single_image(self.current_image_path, self.filter_freq)
            im_type = self.get_file_type(self.current_image_path)
            img = self.normalize_and_flip(img, im_type)

            self.delete_all_patches(self.rectangles)
            self.im.set_data(img)
            self.fig.canvas.draw()
            self.background_current = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.background_orig = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.update_boxes_on_current_image()
        else:
            msg = "Frequency has to be between 0 and 0.5."
            QtG.QMessageBox.information(self, "Message", msg)

    def show_confidence_histogram(self):

        confidence = []
        for box in self.rectangles:
            confidence.append(box.confidence)
        fig = plt.figure()
        # mpl.rcParams["figure.dpi"] = 200
        # mpl.rcParams.update({"font.size": 7})
        width = max(10, int((np.max(confidence) - np.min(confidence)) / 0.05))
        plt.hist(confidence, bins=width)
        plt.title("Confidence distribution")
        bin_size_str = "{0:.2f}".format(
            ((np.max(confidence) - np.min(confidence)) / width)
        )
        plt.xlabel("Confidence (Bin size: " + bin_size_str + ")")
        plt.ylabel("Count")

        plot = QtG.QDialog(self)
        plot.canvas = plt_qtbackend.FigureCanvasQTAgg(fig)
        layout = QtG.QVBoxLayout()
        layout.addWidget(plot.canvas)
        plot.setLayout(layout)
        plot.setWindowTitle("Size distribution")
        plot.canvas.draw()
        plot.show()

    def show_size_distribution(self):

        estimated_size = []
        for ident in self.box_dictionary:
            for box in self.box_dictionary[ident]:
                estimated_size.append(box.est_size)
        fig = plt.figure()
        # mpl.rcParams["figure.dpi"] = 200
        # mpl.rcParams.update({"font.size": 7})
        width = max(10, int((np.max(estimated_size) - np.min(estimated_size)) / 10))
        plt.hist(estimated_size, bins=width)
        plt.title("Particle diameter distribution")
        plt.xlabel("Partilce diameter [px] (Bin size: " + str(width) + "px )")
        plt.ylabel("Count")

        plot = QtG.QDialog(self)
        plot.canvas = plt_qtbackend.FigureCanvasQTAgg(fig)
        layout = QtG.QVBoxLayout()
        layout.addWidget(plot.canvas)
        plot.setLayout(layout)
        plot.setWindowTitle("Size distribution")
        plot.canvas.draw()
        plot.show()

    def load_box_files(self):
        self.is_cbox = False
        keep = False
        if self.unsaved_changes:
            msg = "There are unsaved changes. Are you sure?"
            reply = QtG.QMessageBox.question(
                self, "Message", msg, QtG.QMessageBox.Yes, QtG.QMessageBox.Cancel
            )

            if reply == QtG.QMessageBox.Cancel:
                return

        if len(self.box_dictionary) > 0:
            msg = "Keep old boxes loaded and show the new ones in a different color?"
            reply = QtG.QMessageBox.question(
                self, "Message", msg, QtG.QMessageBox.Yes, QtG.QMessageBox.No
            )

            if reply == QtG.QMessageBox.Yes:
                keep = True

        if not keep:
            self.delete_all_boxes()

        if self.plot is not None:
            box_dir = str(
                QtG.QFileDialog.getExistingDirectory(self, "Select Box Directory")
            )

            if box_dir == "":
                return

            self._import_boxes(box_dir, keep)
        else:
            errmsg = QtG.QErrorMessage(self)
            errmsg.showMessage("Please open an image folder first")

    def _import_boxes(self, box_dir, keep=False):
        import time as t

        t_start = t.time()
        self.setWindowTitle("Box manager (Showing: " + box_dir + ")")
        box_imported = 0
        all_image_filenames = self.get_all_loaded_filesnames()

        print("box_dir: ", box_dir)
        onlyfiles = [
            f for f in os.listdir(box_dir)
            if os.path.isfile(os.path.join(box_dir, f))
               and not f.startswith(".")
               and os.path.splitext(f)[0] in all_image_filenames
               and (
                       f.endswith(".box")
                       or f.endswith(".txt")
                       or f.endswith(".star")
                       or f.endswith(".cbox")
               )
               and os.stat(os.path.join(box_dir, f)).st_size != 0
        ]

        print("onlyfiles: ", onlyfiles)
        colors = ["b", "r", "c", "m", "y", "k", "w"]
        if keep == False:
            rand_color = "r"
        else:
            import random
            rand_color = random.choice(colors)
            while rand_color == "r":
                rand_color = random.choice(colors)
        star_dialog_was_shown = False

        self.conf_thresh_line.setEnabled(False)
        self.conf_thresh_slide.setEnabled(False)
        self.conf_thresh_label.setEnabled(False)
        self.use_estimated_size_label.setEnabled(False)
        self.use_estimated_size_checkbox.setEnabled(False)
        self.show_confidence_histogram_action.setEnabled(False)
        self.show_size_distribution_action.setEnabled(False)

        if len(onlyfiles) > 0:
            pd = QtG.QProgressDialog("Load box files...", "Cancel", 0, 100, self)
            pd.show()

        for file_index, file in enumerate(onlyfiles):
            if pd.wasCanceled():
                break
            else:
                pd.show()
                pd.setValue(int((file_index + 1) * 100 / len(onlyfiles)))
            QtCore.QCoreApplication.instance().processEvents()

            path = os.path.join(box_dir, file)

            self.is_cbox = False
            if path.endswith(".box"):
                print("path endwith .box")
                boxes = coord_io.read_eman_boxfile(path)
            if path.endswith(".star"):
                print("path endwith .star")
                boxes = coord_io.read_star_file(path, 200)
            if path.endswith(".cbox"):
                print("path endwith .cbox")
                boxes = coord_io.read_cbox_boxfile(path)
                self.is_cbox = True

            dict_entry_name = os.path.splitext(file)[0]
            # print("boxes:", boxes)
            rects = [self.box_to_rectangle(box, rand_color) for box in boxes]
            box_imported = box_imported + len(rects)

            if dict_entry_name in self.box_dictionary:
                self.box_dictionary[dict_entry_name].extend(rects)
            else:
                self.box_dictionary[dict_entry_name] = rects

        if self.is_cbox:
            self.conf_thresh_line.setEnabled(True)
            self.conf_thresh_slide.setEnabled(True)
            self.conf_thresh_label.setEnabled(True)
            self.use_estimated_size_label.setEnabled(True)
            self.use_estimated_size_checkbox.setEnabled(True)
            self.show_confidence_histogram_action.setEnabled(True)
            self.show_size_distribution_action.setEnabled(True)

        self.update_boxes_on_current_image()
        self.boxsize_line.setText(str(self.boxsize))

        # In case of cbox files, set the minimum and maximum
        if self.is_cbox:
            min_size = 99999
            max_size = -99999
            for _, rectangles in self.box_dictionary.items():
                for rect in rectangles:
                    if rect.est_size > max_size:
                        max_size = rect.est_size
                    if rect.est_size < min_size:
                        min_size = rect.est_size
            self.upper_size_thresh_slide.setMaximum(max_size)
            self.upper_size_thresh_slide.setMinimum(min_size)
            self.upper_size_thresh_slide.setValue(max_size)
            self.upper_size_thresh_line.setText("" + str(max_size))
            self.lower_size_thresh_slide.setMaximum(max_size)
            self.lower_size_thresh_slide.setMinimum(min_size)
            self.lower_size_thresh_slide.setValue(min_size)
            self.lower_size_thresh_line.setText("" + str(min_size))

        # Update particle numbers
        self.update_tree_boxsizes()
        print("Total time", t.time() - t_start)
        print("Total imported particles: ", box_imported)

    def box_to_rectangle(self, box, color):
        # if np.random.rand()>0.8:
        x_ll = int(box.x)
        y_ll = int(box.y)
        # if x_ll > (197 - 10) and x_ll < (197 + 10) and y_ll > (
        #        225 - 10) and y_ll < (225 + 10):
        width = int(box.w)
        height = int(box.h)
        avg_size = (width + height) // 2
        self.boxsize = avg_size
        est_size = avg_size
        if "est_box_size" in box.meta:
            est_size = (box.meta["est_box_size"][0] + box.meta["est_box_size"][1]) // 2
            # est_size = min(box.meta["est_size"][0],box.meta["est_size"][1])
            if self.use_estimated_size:
                width = est_size
                height = est_size
                #self.boxsize = est_size

        rect = MyRectangle(
            (x_ll, y_ll),
            width,
            height,
            est_size=est_size,
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )
        if box.c:
            rect.set_confidence(box.c)
        else:
            rect.set_confidence(1)
        return rect

    def check_if_should_be_visible(self, box):
        return (
                box.confidence > self.current_conf_thresh
                and self.upper_size_thresh >= box.est_size >= self.lower_size_thresh
        )

    def get_all_loaded_filesnames(self):
        root = self.tree.invisibleRootItem().child(0)
        child_count = root.childCount()
        filenames = []
        for i in range(child_count):
            item = root.child(i)
            filename = os.path.splitext(item.text(0))[0]
            filenames.append(filename)
        print("filenames:", filenames)
        return filenames

    def update_tree_boxsizes(self, update_current=False):
        state = self.get_filter_state()

        def update(boxes, item):

            res = [self.check_if_should_be_visible(box) for box in boxes]

            num_box_vis = int(np.sum(res))
            item.setText(1, "{0:> 4d}  / {1:> 4d}".format(num_box_vis, len(res)))

        if update_current:
            item = self.tree.currentItem()
            filename = os.path.splitext(item.text(0))[0]
            update(self.box_dictionary[filename], item)

        else:
            start = time.time()
            root = self.tree.invisibleRootItem().child(0)
            child_count = root.childCount()

            for i in range(child_count):
                QtCore.QCoreApplication.instance().processEvents()
                if not self.filter_tuple_is_equal(self.get_filter_state(), state):
                    break
                item = root.child(i)
                filename = os.path.splitext(item.text(0))[0]
                if filename in self.box_dictionary:
                    # num_box = sum(map(lambda x: self.check_if_should_be_visible(x), self.box_dictionary[filename]))
                    # item.setText(1, str(num_box))
                    update(self.box_dictionary[filename], item)

    def filter_tuple_is_equal(self, a, b):
        return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]

    def write_coordinates(self, type):
        """
              Write all files
              :return: None
              """
        box_dir = str(
            QtG.QFileDialog.getExistingDirectory(self, "Select Box Directory")
        )
        if box_dir == "":
            return

        # Remove untitled from path if untitled not exists
        if box_dir[-8] == "untitled" and os.path.isdir(box_dir):
            box_dir = box_dir[:-8]

        if box_dir == "":
            return
        num_writtin_part = 0

        pd = QtG.QProgressDialog("Write box files to " + box_dir, "Cancel", 0, 100, self)
        pd.show()
        counter = 0

        if type == "BOX":
            file_ext = ".box"
            write_coords_ = coord_io.write_box_file
        elif type == "STAR":
            file_ext = ".star"
            write_coords_ = coord_io.write_star_file

        for filename, rectangles in self.box_dictionary.items():

            if pd.wasCanceled():
                break
            else:
                pd.show()
                val = int((counter + 1) * 100 / len(self.box_dictionary.items()))
                pd.setValue(val)
            QtCore.QCoreApplication.instance().processEvents()

            box_filename = filename + file_ext
            box_file_path = os.path.join(box_dir, box_filename)
            if self.is_cbox:
                rectangles = [
                    box for box in rectangles if self.check_if_should_be_visible(box)
                ]
                # if self.use_estimated_size_checkbox.isChecked():
            real_rects = [rect.getRect() for rect in rectangles]
            for rect in real_rects:
                self.resize_box(rect, self.boxsize)
            num_writtin_part = num_writtin_part + len(real_rects)

            boxes = []
            from utils import BoundBox
            for rect in real_rects:
                x_lowerleft = int(rect.get_x())
                y_lowerleft = int(rect.get_y())
                boxize = int(rect.get_width())
                box = BoundBox(x=x_lowerleft, y=y_lowerleft, w=boxize, h=boxize)
                boxes.append(box)

            write_coords_(box_file_path, boxes)

            self.unsaved_changes = False
            counter = counter + 1
        print(num_writtin_part, "particles written")

    def write_star_files(self):
        self.write_coordinates(type="STAR")

    def write_box_files(self):
        self.write_coordinates(type="BOX")

    def delete_all_boxes(self):
        for _, rectangles in self.box_dictionary.items():
            self.delete_all_patches(rectangles)

        self.rectangles = []
        self.box_dictionary = {}
        self.update_boxes_on_current_image()
        if self.background_current is not None:
            self.fig.canvas.restore_region(self.background_current)

    def get_filter_state(self):
        lowersize = int(float(self.lower_size_thresh_line.text()))
        uppersize = int(float(self.lower_size_thresh_line.text()))
        conf = float(self.conf_thresh_slide.value())
        return (lowersize, uppersize, conf)

    def draw_all_patches(self, rects):
        state = self.get_filter_state()
        visible_rects = [box.getRect() for box in rects if self.check_if_should_be_visible(box)]
        for rect in visible_rects:
            if rect.get_visible() == False:
                rect.set_visible(True)
            if rect not in self.ax.patches:
                self.ax.add_patch(rect)
            if not self.filter_tuple_is_equal(self.get_filter_state(), state):
                break

    def open_image_folder(self):
        """
        Let the user choose the image folder and adds it to the ImageFolder-Tree
        :return: none
        """
        selected_folder = str(
            QtG.QFileDialog.getExistingDirectory(self, "Select Image Directory")
        )

        if selected_folder == "":
            return

        if self.unsaved_changes:
            msg = "All loaded boxes are discarded. Are you sure?"
            reply = QtG.QMessageBox.question(
                self, "Message", msg, QtG.QMessageBox.Yes, QtG.QMessageBox.Cancel
            )

            if reply == QtG.QMessageBox.Cancel:
                return

        self.current_image_path = None
        img_loaded = self._open_image_folder(selected_folder)
        if img_loaded:
            self.button_apply_filter.setEnabled(True)

    def _open_image_folder(self, path):
        """
        Reads the image folder, setup the folder daemon and signals
        :param path: Path to image folder
        """
        self.image_folder = path
        if path != "":
            title = str(path)
            # if self.wildcard:
            #     title = os.path.join(str(path), self.wildcard)
            root = QtG.QTreeWidgetItem([title])
            self.tree.clear()
            self.tree.setColumnCount(2)
            self.tree.setHeaderHidden(False)
            self.tree.setHeaderLabels(["Filename", "Number of boxes"])
            if self.plot is not None:
                self.plot.close()
            self.rectangles = []
            self.box_dictionary = {}
            self.tree.addTopLevelItem(root)
            fm = QFontMetrics(self.font)
            w = fm.width(path)
            self.tree.setMinimumWidth(w + 150)
            self.tree.setColumnWidth(0, 300)

            onlyfiles = [f for f in sorted(os.listdir(path))
                         if os.path.isfile(os.path.join(path, f))
                         ]

            onlyfiles = [i for i in onlyfiles
                         if not i.startswith(".") and i.endswith((".jpg", ".jpeg", ".png", ".mrc", ".tif", ".tiff"))
                         ]
            all_items = [QtG.QTreeWidgetItem([file]) for file in onlyfiles]

            pd = None
            if len(all_items) > 0:
                pd = QtG.QProgressDialog("Load images", "Cancel", 0, 100, self)
                pd.show()
            for item_index, item in enumerate(all_items):
                pd.show()
                QtCore.QCoreApplication.instance().processEvents()
                pd.setValue(int((item_index + 1) * 100 / len(all_items)))
                root.addChild(item)
            # root.addChildren(all_items)

            if onlyfiles:
                root.setExpanded(True)
                # Show first image
                self.current_image_path = os.path.join(
                    self.image_folder, str(root.child(0).text(0))
                )
                self.current_tree_item = root.child(0)
                im = self.read_image(self.current_image_path)

                self.rectangles = []
                # Create figure and axes
                self.fig, self.ax = plt.subplots(1)
                self.ax.xaxis.set_visible(False)
                self.ax.yaxis.set_visible(False)

                self.fig.tight_layout()
                self.fig.canvas.set_window_title(
                    os.path.basename(self.current_image_path)
                )
                # Display the image
                self.im = self.ax.imshow(im, origin="lower", cmap="gray", interpolation="Hanning")
                self.plot = QtG.QDialog(self)
                self.plot.canvas = plt_qtbackend.FigureCanvasQTAgg(self.fig)
                self.plot.canvas.mpl_connect("button_press_event", self.onclick)
                self.plot.canvas.mpl_connect("key_press_event", self.myKeyPressEvent)
                self.plot.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
                self.plot.canvas.setFocus()
                self.plot.canvas.mpl_connect("button_release_event", self.onrelease)
                self.plot.canvas.mpl_connect("motion_notify_event", self.onmove)
                self.plot.canvas.mpl_connect("resize_event", self.onresize)
                self.plot.canvas.mpl_connect("draw_event", self.ondraw)
                self.plot.toolbar = boxmanagertoolbar.BoxmanagerToolbar(
                    self.plot.canvas, self.plot, self.fig, self.ax, self
                )  # plt_qtbackend.NavigationToolbar2QT(self.plot.canvas, self.plot)
                layout = QtG.QVBoxLayout()
                layout.addWidget(self.plot.toolbar)
                layout.addWidget(self.plot.canvas)
                self.plot.setLayout(layout)
                self.plot.canvas.draw()
                self.plot.show()
                self.background_current = self.fig.canvas.copy_from_bbox(self.ax.bbox)
                self.background_orig = self.fig.canvas.copy_from_bbox(self.ax.bbox)

                self.tree.setCurrentItem(
                    self.tree.invisibleRootItem().child(0).child(0)
                )
                self.plot.setWindowTitle(os.path.basename(self.current_image_path))
                return True
            return False

    def myKeyPressEvent(self, event):
        if event.name == "key_press_event" and event.key == "h":
            pure_filename = os.path.basename(self.current_image_path)[:-4]
            if pure_filename in self.box_dictionary:
                rects = self.box_dictionary[pure_filename]
                if self.toggle:
                    self.draw_all_patches(rects)
                    self.fig.canvas.draw()
                    self.toggle = False
                else:
                    self.delete_all_patches(rects)
                    self.fig.canvas.draw()
                    self.toggle = True

    def ondraw(self, event):
        if self.zoom_update:
            self.zoom_update = False
            self.background_current = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.draw_all_patches(self.rectangles)
            self._draw_all_boxes()

    def onresize(self, event):
        self.delete_all_patches(self.rectangles)
        self.fig.canvas.draw()
        self.background_current = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.background_orig = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.draw_all_patches(self.rectangles)

    def onmove(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            if self.moving_box is not None:
                x = event.xdata - self.moving_box.getRect().get_width() / 2
                y = event.ydata - self.moving_box.getRect().get_width() / 2
                self.boxsize = self.moving_box.getRect().get_width()  # Update the current boxsize
                self.moving_box.getRect().set_x(x)
                self.moving_box.getRect().set_y(y)

                self.fig.canvas.restore_region(self.background_current)
                ## draw all boxes again
                self._draw_all_boxes()

    def get_file_type(self, path):
        if path.endswith(("jpg", "jpeg", "png")):
            im_type = 0
        if path.endswith(("tif", "tiff")):
            im_type = 1
        if path.endswith(("mrc", "mrcs")):
            im_type = 2

        return im_type

    def read_image(self, path):
        im_type = self.get_file_type(path)
        img = read_image.image_read(path)
        img = self.normalize_and_flip(img, im_type)
        return img

    def normalize_and_flip(self, img, file_type):
        if file_type == 0:
            # JPG PNG
            img = np.flip(img, 0)
        if file_type == 1 or file_type == 2:
            # tif or mrc
            if not np.issubdtype(img.dtype, np.float32):
                img = img.astype(np.float32)
            img = np.flip(img, 0)
            mean = np.mean(img)
            sd = np.std(img)
            img = (img - mean) / sd
            img[img > 3] = 3
            img[img < -3] = -3

        return img

    def update_boxes_on_current_image(self):
        if self.current_image_path is None:
            return
        pure_filename = os.path.splitext(os.path.basename(self.current_image_path))[0]
        if pure_filename in self.box_dictionary:
            self.rectangles = self.box_dictionary[pure_filename]
            self.delete_all_patches(self.rectangles, update=True)
            self.draw_all_patches(self.rectangles)
            self._draw_all_boxes()

    def delete_all_patches(self, rects, update=False):
        state = self.get_filter_state()
        for box in rects:
            if self.check_if_should_be_visible(box) == False or update == False:
                rect = box.getRect()
                rect.set_visible(False)
                if rect.pickable():
                    rect.remove()
            if not self.filter_tuple_is_equal(self.get_filter_state(), state):
                break

    def onrelease(self, event):
        self.moving_box = None

    def onclick(self, event):
        # if self.plot.toolbar._active is not None:
        #     return

        modifiers = QtG.QApplication.keyboardModifiers()

        if event.xdata is None or event.ydata is None or event.xdata < 0 or event.ydata < 0:
            return
        pure_filename = os.path.splitext(os.path.basename(self.current_image_path))[0]

        if pure_filename in self.box_dictionary:
            self.rectangles = self.box_dictionary[pure_filename]
        else:
            self.rectangles = []
            self.box_dictionary[pure_filename] = self.rectangles

        if (
                modifiers == QtCore.Qt.ControlModifier
                or modifiers == QtCore.Qt.MetaModifier
        ):
            # Delete box
            box = self.get_corresponding_box(
                event.xdata - self.boxsize / 2,
                event.ydata - self.boxsize / 2,
                self.rectangles,
            )

            if box is not None:
                self.delete_box(box)
        else:
            self.moving_box = self.get_corresponding_box(
                event.xdata - self.boxsize / 2,
                event.ydata - self.boxsize / 2,
                self.rectangles,
            )
            if self.moving_box is None:

                # Delete lower confidence box if available
                box = self.get_corresponding_box(
                    event.xdata - self.boxsize / 2,
                    event.ydata - self.boxsize / 2,
                    self.rectangles,
                    get_low=True,
                )

                if box is not None:
                    self.rectangles.remove(box)

                # Create new box
                xy = (event.xdata - self.boxsize / 2, event.ydata - self.boxsize / 2)
                rect = MyRectangle(
                    xy,
                    self.boxsize,
                    self.boxsize,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                rect.est_size = self.boxsize
                rect.set_confidence(1)
                self.moving_box = rect
                self.rectangles.append(rect)
                # Add the patch to the Axes
                self.ax.add_patch(rect.getRect())
                self.ax.draw_artist(rect.getRect())

                self.fig.canvas.blit(self.ax.bbox)
                self.unsaved_changes = True
            self.update_tree_boxsizes(update_current=True)
            # self.fig.canvas.draw()

    def delete_box(self, box):
        box.getRect().remove()
        del self.rectangles[self.rectangles.index(box)]
        self.fig.canvas.restore_region(self.background_current)
        self._draw_all_boxes()
        self.unsaved_changes = True
        self.update_tree_boxsizes(update_current=True)

    def _draw_all_boxes(self):
        state = self.get_filter_state()
        for box in self.rectangles:
            rect = box.getRect()
            if self.use_estimated_size:
                self.resize_box(rect, box.est_size)
            else:
                self.resize_box(rect, self.boxsize)
            self.ax.draw_artist(rect)
            if not self.filter_tuple_is_equal(self.get_filter_state(), state):
                break
        self.fig.canvas.blit(self.ax.bbox)

    def get_corresponding_box(self, x, y, rectangles, get_low=False):
        a = np.array([x, y])

        for box in rectangles:
            b = np.array(box.getRect().xy)
            dist = np.linalg.norm(a - b)
            if get_low:
                if (
                        dist < self.boxsize / 2
                        and box.confidence < self.current_conf_thresh
                ):
                    return box
            else:
                if (
                        dist < self.boxsize / 2
                        and box.confidence > self.current_conf_thresh
                ):
                    return box
        return None

    def get_number_visible_boxes(self, rectangles):
        i = 0
        for box in rectangles:
            if box.is_figure_set():
                i = i + 1
        return i


def start_boxmanager(image_dir, box_dir, box_size, prob_threshold):
    app = QtG.QApplication(sys.argv)
    gui = MainWindow(app.font(), images_path=image_dir, boxes_path=box_dir, box_size=box_size, prob_threshold=prob_threshold)
    sys.exit(app.exec_())


def run(args=None):
    args = argparser.parse_args()
    image_dir = args.image_dir
    box_dir = args.box_dir
    # wildcard = args.wildcard
    box_size = args.box_size
    prob_threshold = args.prob_threshold
    start_boxmanager(image_dir, box_dir, box_size, prob_threshold)


if __name__ == "__main__":
    run()
