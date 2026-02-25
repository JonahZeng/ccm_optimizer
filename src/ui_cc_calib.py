# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'cc_calib.ui'
##
## Created by: Qt User Interface Compiler version 6.4.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QSlider, QSpacerItem,
    QVBoxLayout, QWidget)

from .raw_preview_widget import RawPreviewWidget

class Ui_eCCM(object):
    def setupUi(self, eCCM):
        if not eCCM.objectName():
            eCCM.setObjectName(u"eCCM")
        eCCM.resize(1262, 778)
        eCCM.setStyleSheet(u"")
        self.horizontalLayout = QHBoxLayout(eCCM)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(4, 4, 4, 4)
        self.rawPreviewWidget = RawPreviewWidget(eCCM)
        self.rawPreviewWidget.setObjectName(u"rawPreviewWidget")
        self.rawPreviewWidget.setStyleSheet(u"background-color: rgb(100, 100, 100);")

        self.horizontalLayout.addWidget(self.rawPreviewWidget)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.importRawBtn = QPushButton(eCCM)
        self.importRawBtn.setObjectName(u"importRawBtn")
        self.importRawBtn.setStyleSheet(u"background-color: rgb(170, 170, 255);\n"
"")

        self.verticalLayout.addWidget(self.importRawBtn)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)

        self.label = QLabel(eCCM)
        self.label.setObjectName(u"label")

        self.verticalLayout.addWidget(self.label)

        self.patchSizeSlider = QSlider(eCCM)
        self.patchSizeSlider.setObjectName(u"patchSizeSlider")
        self.patchSizeSlider.setMaximum(80)
        self.patchSizeSlider.setPageStep(5)
        self.patchSizeSlider.setValue(80)
        self.patchSizeSlider.setOrientation(Qt.Horizontal)

        self.verticalLayout.addWidget(self.patchSizeSlider)

        self.groupBox = QGroupBox(eCCM)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setMinimumSize(QSize(166, 209))
        self.adjTopOuterBtn = QPushButton(self.groupBox)
        self.adjTopOuterBtn.setObjectName(u"adjTopOuterBtn")
        self.adjTopOuterBtn.setGeometry(QRect(40, 20, 75, 24))
        self.adjTopOuterBtn.setStyleSheet(u"background-color: rgb(128, 128, 128);")
        self.adjLeftOuterBtn = QPushButton(self.groupBox)
        self.adjLeftOuterBtn.setObjectName(u"adjLeftOuterBtn")
        self.adjLeftOuterBtn.setGeometry(QRect(0, 80, 21, 61))
        self.adjLeftOuterBtn.setStyleSheet(u"background-color: rgb(128, 128, 128);")
        self.adjRightOuterBtn = QPushButton(self.groupBox)
        self.adjRightOuterBtn.setObjectName(u"adjRightOuterBtn")
        self.adjRightOuterBtn.setGeometry(QRect(140, 80, 21, 61))
        self.adjRightOuterBtn.setStyleSheet(u"background-color: rgb(128, 128, 128);")
        self.adjBottomOuterBtn = QPushButton(self.groupBox)
        self.adjBottomOuterBtn.setObjectName(u"adjBottomOuterBtn")
        self.adjBottomOuterBtn.setGeometry(QRect(40, 180, 75, 24))
        self.adjBottomOuterBtn.setStyleSheet(u"background-color: rgb(128, 128, 128);")
        self.adjTopInnerBtn = QPushButton(self.groupBox)
        self.adjTopInnerBtn.setObjectName(u"adjTopInnerBtn")
        self.adjTopInnerBtn.setGeometry(QRect(40, 50, 75, 23))
        self.adjTopInnerBtn.setStyleSheet(u"background-color: rgb(128, 128, 128);")
        self.pushButton = QPushButton(self.groupBox)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(40, 150, 75, 23))
        self.pushButton.setStyleSheet(u"background-color: rgb(128, 128, 128);")
        self.pushButton_2 = QPushButton(self.groupBox)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(30, 80, 21, 61))
        self.pushButton_2.setStyleSheet(u"background-color: rgb(128, 128, 128);")
        self.pushButton_3 = QPushButton(self.groupBox)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(111, 80, 20, 61))
        self.pushButton_3.setStyleSheet(u"background-color: rgb(128, 128, 128);")

        self.verticalLayout.addWidget(self.groupBox)

        self.previewBrightLabel = QLabel(eCCM)
        self.previewBrightLabel.setObjectName(u"previewBrightLabel")

        self.verticalLayout.addWidget(self.previewBrightLabel)

        self.previewBrightSlider = QSlider(eCCM)
        self.previewBrightSlider.setObjectName(u"previewBrightSlider")
        self.previewBrightSlider.setMaximum(40)
        self.previewBrightSlider.setSingleStep(1)
        self.previewBrightSlider.setPageStep(2)
        self.previewBrightSlider.setValue(10)
        self.previewBrightSlider.setOrientation(Qt.Horizontal)

        self.verticalLayout.addWidget(self.previewBrightSlider)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_4)

        self.importTargetBtn = QPushButton(eCCM)
        self.importTargetBtn.setObjectName(u"importTargetBtn")
        self.importTargetBtn.setStyleSheet(u"background-color: rgb(128, 128, 128);")

        self.verticalLayout.addWidget(self.importTargetBtn)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_3)

        self.calcuateCcmBtn = QPushButton(eCCM)
        self.calcuateCcmBtn.setObjectName(u"calcuateCcmBtn")
        self.calcuateCcmBtn.setStyleSheet(u"background-color: rgb(85, 170, 127);")

        self.verticalLayout.addWidget(self.calcuateCcmBtn)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_2)

        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(4, 1)
        self.verticalLayout.setStretch(9, 1)
        self.verticalLayout.setStretch(11, 1)

        self.horizontalLayout.addLayout(self.verticalLayout)

        self.horizontalLayout.setStretch(0, 1)

        self.retranslateUi(eCCM)
        self.importRawBtn.clicked.connect(eCCM.import_raw_file)
        self.adjTopOuterBtn.clicked.connect(self.rawPreviewWidget.adj_roi_outertop)
        self.adjBottomOuterBtn.clicked.connect(self.rawPreviewWidget.adj_roi_outerbottom)
        self.adjLeftOuterBtn.clicked.connect(self.rawPreviewWidget.adj_roi_outerleft)
        self.adjRightOuterBtn.clicked.connect(self.rawPreviewWidget.adj_roi_outerright)
        self.adjTopInnerBtn.clicked.connect(self.rawPreviewWidget.adj_roi_innertop)
        self.pushButton_2.clicked.connect(self.rawPreviewWidget.adj_roi_innerleft)
        self.pushButton.clicked.connect(self.rawPreviewWidget.adj_roi_innerbottom)
        self.pushButton_3.clicked.connect(self.rawPreviewWidget.adj_roi_innerright)
        self.previewBrightSlider.valueChanged.connect(self.rawPreviewWidget.preview_brigh_adj)
        self.previewBrightSlider.valueChanged.connect(eCCM.show_preview_bright)
        self.patchSizeSlider.valueChanged.connect(self.rawPreviewWidget.patch_size_change)
        self.calcuateCcmBtn.clicked.connect(eCCM.calculate_ccm)
        self.importTargetBtn.clicked.connect(eCCM.import_target_file)

        QMetaObject.connectSlotsByName(eCCM)
    # setupUi

    def retranslateUi(self, eCCM):
        eCCM.setWindowTitle(QCoreApplication.translate("eCCM", u"eCCM", None))
        self.importRawBtn.setText(QCoreApplication.translate("eCCM", u"import raw", None))
        self.label.setText(QCoreApplication.translate("eCCM", u"patch size:", None))
        self.groupBox.setTitle(QCoreApplication.translate("eCCM", u"adjust ROI", None))
        self.adjTopOuterBtn.setText(QCoreApplication.translate("eCCM", u"^", None))
        self.adjLeftOuterBtn.setText(QCoreApplication.translate("eCCM", u"<", None))
        self.adjRightOuterBtn.setText(QCoreApplication.translate("eCCM", u">", None))
        self.adjBottomOuterBtn.setText(QCoreApplication.translate("eCCM", u"v", None))
        self.adjTopInnerBtn.setText(QCoreApplication.translate("eCCM", u"v", None))
        self.pushButton.setText(QCoreApplication.translate("eCCM", u"^", None))
        self.pushButton_2.setText(QCoreApplication.translate("eCCM", u">", None))
        self.pushButton_3.setText(QCoreApplication.translate("eCCM", u"<", None))
        self.previewBrightLabel.setText(QCoreApplication.translate("eCCM", u"preview brightness:1.0", None))
        self.importTargetBtn.setText(QCoreApplication.translate("eCCM", u"import target + gamma", None))
        self.calcuateCcmBtn.setText(QCoreApplication.translate("eCCM", u"calculate ccm", None))
    # retranslateUi

