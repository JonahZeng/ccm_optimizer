'''preview widget'''
from typing import Optional
from PySide6.QtWidgets import QWidget, QStyleOption, QStyle
from PySide6.QtGui import QPaintEvent, QPainter, QImage, QMouseEvent, QPen, QColor, QResizeEvent
from PySide6.QtCore import Slot, QRect, QPoint
import numpy as np


class RawPreviewWidget(QWidget):
    """raw preview widget

    Args:
        QWidget (_type_): parent widget
    """
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.bgr_data: Optional[np.ndarray] = None
        self.bgr_data_preview: Optional[np.ndarray] = None
        self.begin_paint = False
        self.start_pt = QPoint(0, 0)
        self.end_pt = QPoint(0, 0)
        self.start_x_percent = 0.0
        self.start_y_percent = 0.0
        self.end_x_percent = 0.0
        self.end_y_percent = 0.0
        self.patch_size = 0.8
        self.rgain = 1.0
        self.bgain = 1.0
        self.wb_is_done = False

    @Slot() # type: ignore
    def paintEvent(self, event: QPaintEvent) -> None:
        """_summary_

        Args:
            event (QPaintEvent): _description_
        """
        opt = QStyleOption()
        opt.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PrimitiveElement.PE_Widget, opt, p, self)
        if self.bgr_data is None:
            pass
        elif isinstance(self.bgr_data, np.ndarray) and isinstance(self.bgr_data_preview, np.ndarray):
            widget_rect = self.geometry()
            widget_paint_rect = QRect(0, 0, widget_rect.width(), widget_rect.height())
            qimg = QImage(self.bgr_data_preview.tobytes(), self.bgr_data.shape[1], self.bgr_data.shape[0],
                          self.bgr_data.shape[1] * 3, QImage.Format.Format_BGR888)
            p.drawImage(widget_paint_rect, qimg)
            if self.start_pt != self.end_pt:
                pen = QPen()
                pen.setColor(QColor(200, 80, 80))
                pen.setWidth(2)
                p.setPen(pen)
                p.drawRect(QRect(self.start_pt, self.end_pt))
                x_step = (self.end_pt.x() - self.start_pt.x()) / 6.0
                y_step = (self.end_pt.y() - self.start_pt.y()) / 4.0
                gap = (1.0 - self.patch_size) / 2.0
                for row in range(1, 5):
                    for col in range(1, 7):
                        p0 = self.start_pt + QPoint(
                            int((col - 1) * x_step + x_step * gap),
                            int((row - 1) * y_step + y_step * gap)
                        )
                        p1 = p0 + QPoint(int(x_step * self.patch_size), int(y_step * self.patch_size))
                        p.drawRect(QRect(p0, p1))
        super().paintEvent(event)

    @Slot() # type: ignore
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """_summary_

        Args:
            event (QMouseEvent): _description_
        """
        if self.bgr_data is None:
            return
        self.start_pt = QPoint(0, 0)
        self.end_pt = QPoint(0, 0)
        self.repaint()
        self.begin_paint = True
        self.start_pt = event.pos()
        self.end_pt = event.pos()

        widget_rect = self.geometry()
        self.start_x_percent = self.start_pt.x() / widget_rect.width()
        self.start_y_percent = self.start_pt.y() / widget_rect.height()
        self.end_x_percent = self.end_pt.x() / widget_rect.width()
        self.end_y_percent = self.end_pt.y() / widget_rect.height()

    @Slot() # type: ignore
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """_summary_

        Args:
            event (QMouseEvent): _description_
        """
        if self.bgr_data is None:
            return
        if self.begin_paint:
            self.begin_paint = False
            self.end_pt = event.pos()
            widget_rect = self.geometry()
            self.end_x_percent = self.end_pt.x() / widget_rect.width()
            self.end_y_percent = self.end_pt.y() / widget_rect.height()
            self.update()

    @Slot() # type: ignore
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """_summary_

        Args:
            event (QMouseEvent): _description_
        """
        if self.bgr_data is None:
            return
        if self.begin_paint:
            self.end_pt = event.pos()
            widget_rect = self.geometry()
            self.end_x_percent = self.end_pt.x() / widget_rect.width()
            self.end_y_percent = self.end_pt.y() / widget_rect.height()
            self.update()

    @Slot() # type: ignore
    def resizeEvent(self, event: QResizeEvent) -> None:
        """_summary_

        Args:
            event (QResizeEvent): _description_
        """
        if self.bgr_data is None:
            return
        if self.start_pt == self.end_pt:
            return
        self.start_pt = QPoint(int(event.size().width() * self.start_x_percent),
                              int(event.size().height() * self.start_y_percent))
        self.end_pt = QPoint(int(event.size().width() * self.end_x_percent),
                             int(event.size().height() * self.end_y_percent))

    @Slot()
    def adj_roi_outertop(self):
        """_summary_
        """
        if self.bgr_data is None:
            return
        if self.start_pt == self.end_pt:
            return
        self.start_pt = self.start_pt + QPoint(0, -2)
        widget_rect = self.geometry()
        self.start_y_percent = self.start_pt.y() / widget_rect.height()
        self.update()

    @Slot()
    def adj_roi_innertop(self):
        """_summary_
        """
        if self.bgr_data is None:
            return
        if self.start_pt == self.end_pt:
            return
        self.start_pt = self.start_pt + QPoint(0, 2)
        widget_rect = self.geometry()
        self.start_y_percent = self.start_pt.y() / widget_rect.height()
        self.update()

    @Slot()
    def adj_roi_outerbottom(self):
        """_summary_
        """
        if self.bgr_data is None:
            return
        if self.start_pt == self.end_pt:
            return
        self.end_pt = self.end_pt + QPoint(0, 2)
        widget_rect = self.geometry()
        self.end_y_percent = self.end_pt.y() / widget_rect.height()
        self.update()

    @Slot()
    def adj_roi_innerbottom(self):
        """_summary_
        """
        if self.bgr_data is None:
            return
        if self.start_pt == self.end_pt:
            return
        self.end_pt = self.end_pt + QPoint(0, -2)
        widget_rect = self.geometry()
        self.end_y_percent = self.end_pt.y() / widget_rect.height()
        self.update()

    @Slot()
    def adj_roi_outerleft(self):
        """_summary_
        """
        if self.bgr_data is None:
            return
        if self.start_pt == self.end_pt:
            return
        self.start_pt = self.start_pt + QPoint(-2, 0)
        widget_rect = self.geometry()
        self.start_x_percent = self.start_pt.x() / widget_rect.width()
        self.update()

    @Slot()
    def adj_roi_innerleft(self):
        """_summary_
        """
        if self.bgr_data is None:
            return
        if self.start_pt == self.end_pt:
            return
        self.start_pt = self.start_pt + QPoint(2, 0)
        widget_rect = self.geometry()
        self.start_x_percent = self.start_pt.x() / widget_rect.width()
        self.update()

    @Slot()
    def adj_roi_outerright(self):
        """_summary_
        """
        if self.bgr_data is None:
            return
        if self.start_pt == self.end_pt:
            return
        self.end_pt = self.end_pt + QPoint(2, 0)
        widget_rect = self.geometry()
        self.end_x_percent = self.end_pt.x() / widget_rect.width()
        self.update()

    @Slot()
    def adj_roi_innerright(self):
        """_summary_
        """
        if self.bgr_data is None:
            return
        if self.start_pt == self.end_pt:
            return
        self.end_pt = self.end_pt + QPoint(-2, 0)
        widget_rect = self.geometry()
        self.end_x_percent = self.end_pt.x() / widget_rect.width()
        self.update()

    @Slot() # type: ignore
    def preview_brigh_adj(self, val: int):
        """_summary_

        Args:
            val (int): _description_
        """
        if self.bgr_data is None:
            return
        if isinstance(self.bgr_data, np.ndarray):
            self.bgr_data_preview = np.clip(self.bgr_data * (val / 10.0), 0, 255).astype(np.uint8)
            self.bgr_data_preview[:, :, 0] = np.clip(self.bgr_data_preview[:, :, 0] *
                                                     self.bgain, 0, 255).astype(np.uint8)
            self.bgr_data_preview[:, :, 2] = np.clip(self.bgr_data_preview[:, :, 2] *
                                                     self.rgain, 0, 255).astype(np.uint8)
            self.update()

    @Slot() # type: ignore
    def patch_size_change(self, val: int):
        """_summary_

        Args:
            val (int): _description_
        """
        if self.bgr_data is None:
            self.patch_size = val / 100
            return
        if self.start_pt == self.end_pt:
            return
        if isinstance(self.bgr_data, np.ndarray):
            self.patch_size = val / 100
            self.update()

    @Slot() # type: ignore
    def apply_wb(self, bgain: float, rgain: float):
        """_summary_

        Args:
            bgain (float): _description_
            rgain (float): _description_
        """
        if self.bgr_data is None or self.bgr_data_preview is None:
            return
        if self.wb_is_done:
            return
        if isinstance(self.bgr_data_preview, np.ndarray):
            self.rgain = rgain
            self.bgain = bgain
            self.wb_is_done = True
            self.bgr_data_preview[:, :, 0] = np.clip(self.bgr_data_preview[:, :, 0] * bgain, 0, 255).astype(np.uint8)
            self.bgr_data_preview[:, :, 2] = np.clip(self.bgr_data_preview[:, :, 2] * rgain, 0, 255).astype(np.uint8)
            self.update()
