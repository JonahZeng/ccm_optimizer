from PySide6.QtWidgets import QWidget, QStyleOption, QStyle
from PySide6.QtGui import QPaintEvent, QPainter, QImage, QMouseEvent, QPen, QColor, QResizeEvent
from PySide6.QtCore import Slot, QRect, QPoint
import numpy as np


class rawPreviewWidget(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.bgrData = None
        self.bgrDataPreview = None
        self.beginPaint = False
        self.startPt = QPoint(0, 0)
        self.endPt = QPoint(0, 0)
        self.startXPercent = 0.0
        self.startYPercent = 0.0
        self.endXPercent = 0.0
        self.endYPercent = 0.0
        self.patchSize = 0.8
        self.rgain = 1.0
        self.bgain = 1.0
        self.hadDoWb = False

    @Slot()
    def paintEvent(self, event: QPaintEvent) -> None:
        opt = QStyleOption()
        opt.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, p, self)
        if self.bgrData is None:
            pass
        elif type(self.bgrData) == np.ndarray and type(self.bgrDataPreview) == np.ndarray:
            widgetRect = self.geometry()
            widgetPaintRect = QRect(0, 0, widgetRect.width(), widgetRect.height())
            qimg = QImage(self.bgrDataPreview.tobytes(), self.bgrData.shape[1], self.bgrData.shape[0],
                          self.bgrData.shape[1] * 3, QImage.Format_BGR888)
            p.drawImage(widgetPaintRect, qimg)
            if self.startPt != self.endPt:
                pen = QPen()
                pen.setColor(QColor(200, 80, 80))
                pen.setWidth(2)
                p.setPen(pen)
                p.drawRect(QRect(self.startPt, self.endPt))
                x_step = (self.endPt.x() - self.startPt.x()) / 6.0
                y_step = (self.endPt.y() - self.startPt.y()) / 4.0
                gap = (1.0 - self.patchSize) / 2.0
                for row in range(1, 5):
                    for col in range(1, 7):
                        p0 = self.startPt + QPoint((col - 1) * x_step + x_step * gap, (row - 1) * y_step + y_step * gap)
                        p1 = p0 + QPoint(x_step * self.patchSize, y_step * self.patchSize)
                        p.drawRect(QRect(p0, p1))
        super().paintEvent(event)

    @Slot()
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.bgrData is None:
            return
        self.startPt = QPoint(0, 0)
        self.endPt = QPoint(0, 0)
        self.repaint()
        self.beginPaint = True
        self.startPt = event.pos()
        self.endPt = event.pos()

        widgetRect = self.geometry()
        self.startXPercent = self.startPt.x() / widgetRect.width()
        self.startYPercent = self.startPt.y() / widgetRect.height()
        self.endXPercent = self.endPt.x() / widgetRect.width()
        self.endYPercent = self.endPt.y() / widgetRect.height()

    @Slot()
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.bgrData is None:
            return
        if self.beginPaint:
            self.beginPaint = False
            self.endPt = event.pos()
            widgetRect = self.geometry()
            self.endXPercent = self.endPt.x() / widgetRect.width()
            self.endYPercent = self.endPt.y() / widgetRect.height()
            self.update()

    @Slot()
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.bgrData is None:
            return
        if self.beginPaint:
            self.endPt = event.pos()
            widgetRect = self.geometry()
            self.endXPercent = self.endPt.x() / widgetRect.width()
            self.endYPercent = self.endPt.y() / widgetRect.height()
            self.update()

    @Slot()
    def resizeEvent(self, event: QResizeEvent) -> None:
        if self.bgrData is None:
            return
        if self.startPt == self.endPt:
            return
        self.startPt = QPoint(event.size().width() * self.startXPercent, event.size().height() * self.startYPercent)
        self.endPt = QPoint(event.size().width() * self.endXPercent, event.size().height() * self.endYPercent)

    @Slot()
    def adjRoiOuterTop(self):
        if self.bgrData is None:
            return
        if self.startPt == self.endPt:
            return
        self.startPt = self.startPt + QPoint(0, -2)
        widgetRect = self.geometry()
        self.startYPercent = self.startPt.y() / widgetRect.height()
        self.update()

    @Slot()
    def adjRoiInnerTop(self):
        if self.bgrData is None:
            return
        if self.startPt == self.endPt:
            return
        self.startPt = self.startPt + QPoint(0, 2)
        widgetRect = self.geometry()
        self.startYPercent = self.startPt.y() / widgetRect.height()
        self.update()

    @Slot()
    def adjRoiOuterBottom(self):
        if self.bgrData is None:
            return
        if self.startPt == self.endPt:
            return
        self.endPt = self.endPt + QPoint(0, 2)
        widgetRect = self.geometry()
        self.endYPercent = self.endPt.y() / widgetRect.height()
        self.update()

    @Slot()
    def adjRoiInnerBottom(self):
        if self.bgrData is None:
            return
        if self.startPt == self.endPt:
            return
        self.endPt = self.endPt + QPoint(0, -2)
        widgetRect = self.geometry()
        self.endYPercent = self.endPt.y() / widgetRect.height()
        self.update()

    @Slot()
    def adjRoiOuterLeft(self):
        if self.bgrData is None:
            return
        if self.startPt == self.endPt:
            return
        self.startPt = self.startPt + QPoint(-2, 0)
        widgetRect = self.geometry()
        self.startXPercent = self.startPt.x() / widgetRect.width()
        self.update()

    @Slot()
    def adjRoiInnerLeft(self):
        if self.bgrData is None:
            return
        if self.startPt == self.endPt:
            return
        self.startPt = self.startPt + QPoint(2, 0)
        widgetRect = self.geometry()
        self.startXPercent = self.startPt.x() / widgetRect.width()
        self.update()

    @Slot()
    def adjRoiOuterRight(self):
        if self.bgrData is None:
            return
        if self.startPt == self.endPt:
            return
        self.endPt = self.endPt + QPoint(2, 0)
        widgetRect = self.geometry()
        self.endXPercent = self.endPt.x() / widgetRect.width()
        self.update()

    @Slot()
    def adjRoiInnerRight(self):
        if self.bgrData is None:
            return
        if self.startPt == self.endPt:
            return
        self.endPt = self.endPt + QPoint(-2, 0)
        widgetRect = self.geometry()
        self.endXPercent = self.endPt.x() / widgetRect.width()
        self.update()

    @Slot()
    def previewBrightAdj(self, val: int):
        if self.bgrData is None:
            return
        if type(self.bgrData) == np.ndarray:
            self.bgrDataPreview = np.clip(self.bgrData * (val / 10.0), 0, 255).astype(np.uint8)
            self.bgrDataPreview[:, :, 0] = np.clip(self.bgrDataPreview[:, :, 0] * self.bgain, 0, 255).astype(np.uint8)
            self.bgrDataPreview[:, :, 2] = np.clip(self.bgrDataPreview[:, :, 2] * self.rgain, 0, 255).astype(np.uint8)
            self.update()

    @Slot()
    def patchSizeChange(self, val: int):
        if self.bgrData is None:
            self.patchSize = val / 100
            return
        if self.startPt == self.endPt:
            return
        if type(self.bgrData) == np.ndarray:
            self.patchSize = val / 100
            self.update()

    @Slot()
    def doWb(self, bgain: float, rgain: float):
        if self.bgrData is None or self.bgrDataPreview is None:
            return
        if self.hadDoWb:
            return
        if type(self.bgrDataPreview) == np.ndarray:
            self.rgain = rgain
            self.bgain = bgain
            self.hadDoWb = True
            self.bgrDataPreview[:, :, 0] = np.clip(self.bgrDataPreview[:, :, 0] * bgain, 0, 255).astype(np.uint8)
            self.bgrDataPreview[:, :, 2] = np.clip(self.bgrDataPreview[:, :, 2] * rgain, 0, 255).astype(np.uint8)
            self.update()
