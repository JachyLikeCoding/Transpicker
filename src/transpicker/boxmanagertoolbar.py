from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)


class BoxmanagerToolbar(NavigationToolbar):
    """
    Tool for Boxmanager.
    """

    def __init__(self, canvas_, parent_, fig, ax, boxmanager):
        self.fig = fig
        self.axes = ax
        self.dozoom = False
        self.boxmanager = boxmanager
        NavigationToolbar.__init__(self, canvas_, parent_)

    def press_zoom(self, event):
        super(BoxmanagerToolbar, self).press_zoom(event)

    def zoom(self, *args):
        super(BoxmanagerToolbar, self).zoom(args)

    def home(self, *args):
        self.boxmanager.delete_all_patches(self.boxmanager.rectangles)
        self.boxmanager.fig.canvas.restore_region(self.boxmanager.background_orig)
        self.boxmanager.backfround_current = self.fig.canvas.copy_from_bbox(
            self.axes.bbox
        )
        self.boxmanager.draw_all_patches(self.boxmanager.rectangles)
        super(BoxmanagerToolbar, self).home(args)

    def release_zoom(self, event):
        if not self._xypress:
            return
        self.dozoom = False
        for cur_xypress in self._xypress:
            x_pos, y_pos = event.x, event.y
            lastx = cur_xypress[0]
            lasty = cur_xypress[1]
            # ignore singular clicks - 5 pixels is a threshold
            if not (abs(x_pos - lastx) < 5 or abs(y_pos - lasty) < 5):
                self.dozoom = True
                self.boxmanager.delete_all_patches(self.boxmanager.rectangles)
                self.boxmanager.fig.canvas.restore_region(
                    self.boxmanager.background_orig
                )
                self.boxmanager.zoom_update = True

        super(BoxmanagerToolbar, self).release_zoom(event)

    def pan(self, *args):
        super(BoxmanagerToolbar, self).pan(args)

    def drag_pan(self, event):
        print("drag pan")
        super(BoxmanagerToolbar, self).drag_pan(event)
        self.boxmanager.delete_all_patches(self.boxmanager.rectangles)
        self.fig.canvas.restore_region(self.boxmanager.background_current)
        self.boxmanager.background_current = self.fig.canvas.copy_from_bbox(self.axes.bbox)
        self.boxmanager.zoom_update = True
