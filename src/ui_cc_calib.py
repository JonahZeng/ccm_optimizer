# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'cc_calib.ui'
##
## Created by: Qt User Interface Compiler version 6.4.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QMetaObject, QRect,
    QSize, Qt)
from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QSizePolicy, QSlider, QSpacerItem,
    QVBoxLayout)

from .raw_preview_widget import RawPreviewWidget

class Ui_eCCM(object):
    def setupUi(self, eCCM):
        if not eCCM.objectName():
            eCCM.setObjectName("eCCM")
        eCCM.resize(1262, 778)
        eCCM.setStyleSheet("")
        self.horizontalLayout = QHBoxLayout(eCCM)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout.setContentsMargins(4, 4, 4, 4)
        self.rawPreviewWidget = RawPreviewWidget(eCCM)
        self.rawPreviewWidget.setObjectName("rawPreviewWidget")
        self.rawPreviewWidget.setStyleSheet("background-color: rgb(100, 100, 100);")

        self.horizontalLayout.addWidget(self.rawPreviewWidget)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.importRawBtn = QPushButton(eCCM)
        self.importRawBtn.setObjectName("importRawBtn")
        self.importRawBtn.setStyleSheet("background-color: rgb(170, 170, 255);\n"
"")

        self.verticalLayout.addWidget(self.importRawBtn) # 0

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer) # 1

        self.label = QLabel(eCCM)
        self.label.setObjectName("label")

        self.verticalLayout.addWidget(self.label) # 2

        self.patchSizeSlider = QSlider(eCCM)
        self.patchSizeSlider.setObjectName("patchSizeSlider")
        self.patchSizeSlider.setMaximum(80)
        self.patchSizeSlider.setPageStep(5)
        self.patchSizeSlider.setValue(80)
        self.patchSizeSlider.setOrientation(Qt.Orientation.Horizontal)

        self.verticalLayout.addWidget(self.patchSizeSlider) # 3

        self.groupBox = QGroupBox(eCCM)
        self.groupBox.setObjectName("groupBox")
        self.groupBox.setMinimumSize(QSize(166, 209))
        self.adjTopOuterBtn = QPushButton(self.groupBox)
        self.adjTopOuterBtn.setObjectName("adjTopOuterBtn")
        self.adjTopOuterBtn.setGeometry(QRect(40, 20, 75, 24))
        self.adjTopOuterBtn.setStyleSheet("background-color: rgb(128, 128, 128);")
        self.adjLeftOuterBtn = QPushButton(self.groupBox)
        self.adjLeftOuterBtn.setObjectName("adjLeftOuterBtn")
        self.adjLeftOuterBtn.setGeometry(QRect(0, 80, 21, 61))
        self.adjLeftOuterBtn.setStyleSheet("background-color: rgb(128, 128, 128);")
        self.adjRightOuterBtn = QPushButton(self.groupBox)
        self.adjRightOuterBtn.setObjectName("adjRightOuterBtn")
        self.adjRightOuterBtn.setGeometry(QRect(140, 80, 21, 61))
        self.adjRightOuterBtn.setStyleSheet("background-color: rgb(128, 128, 128);")
        self.adjBottomOuterBtn = QPushButton(self.groupBox)
        self.adjBottomOuterBtn.setObjectName("adjBottomOuterBtn")
        self.adjBottomOuterBtn.setGeometry(QRect(40, 180, 75, 24))
        self.adjBottomOuterBtn.setStyleSheet("background-color: rgb(128, 128, 128);")
        self.adjTopInnerBtn = QPushButton(self.groupBox)
        self.adjTopInnerBtn.setObjectName("adjTopInnerBtn")
        self.adjTopInnerBtn.setGeometry(QRect(40, 50, 75, 23))
        self.adjTopInnerBtn.setStyleSheet("background-color: rgb(128, 128, 128);")
        self.adjBottomInnerBtn = QPushButton(self.groupBox)
        self.adjBottomInnerBtn.setObjectName("adjBottomInnerBtn")
        self.adjBottomInnerBtn.setGeometry(QRect(40, 150, 75, 23))
        self.adjBottomInnerBtn.setStyleSheet("background-color: rgb(128, 128, 128);")
        self.adjLeftInnerBtn = QPushButton(self.groupBox)
        self.adjLeftInnerBtn.setObjectName("adjLeftInnerBtn")
        self.adjLeftInnerBtn.setGeometry(QRect(30, 80, 21, 61))
        self.adjLeftInnerBtn.setStyleSheet("background-color: rgb(128, 128, 128);")
        self.adjRightInnerBtn = QPushButton(self.groupBox)
        self.adjRightInnerBtn.setObjectName("adjRightInnerBtn")
        self.adjRightInnerBtn.setGeometry(QRect(111, 80, 20, 61))
        self.adjRightInnerBtn.setStyleSheet("background-color: rgb(128, 128, 128);")

        self.verticalLayout.addWidget(self.groupBox) # 4

        self.previewBrightLabel = QLabel(eCCM)
        self.previewBrightLabel.setObjectName("previewBrightLabel")

        self.verticalLayout.addWidget(self.previewBrightLabel) # 5

        self.previewBrightSlider = QSlider(eCCM)
        self.previewBrightSlider.setObjectName("previewBrightSlider")
        self.previewBrightSlider.setMaximum(40)
        self.previewBrightSlider.setSingleStep(1)
        self.previewBrightSlider.setPageStep(2)
        self.previewBrightSlider.setValue(10)
        self.previewBrightSlider.setOrientation(Qt.Orientation.Horizontal)

        self.verticalLayout.addWidget(self.previewBrightSlider) # 6

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_4) # 7

        self.importTargetBtn = QPushButton(eCCM)
        self.importTargetBtn.setObjectName("importTargetBtn")
        self.importTargetBtn.setStyleSheet("background-color: rgb(128, 128, 128);")

        self.verticalLayout.addWidget(self.importTargetBtn) # 8

        self.optimizeModeComboBox = QComboBox(eCCM)
        self.optimizeModeComboBox.setObjectName("optimizeModeComboBox")
        self.optimizeModeComboBox.addItem("gd with bp")
        self.optimizeModeComboBox.addItem("gd without bp")
        optimizeModeLabel = QLabel("optimization method:", parent=eCCM)
        optimizeModeLayout = QHBoxLayout()
        optimizeModeLayout.addWidget(optimizeModeLabel)
        optimizeModeLayout.addWidget(self.optimizeModeComboBox)
        self.verticalLayout.addLayout(optimizeModeLayout) # 9

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_3)  # 10

        self.calcuateCcmBtn = QPushButton(eCCM)
        self.calcuateCcmBtn.setObjectName("calcuateCcmBtn")
        self.calcuateCcmBtn.setStyleSheet("background-color: rgb(85, 170, 127);")

        self.verticalLayout.addWidget(self.calcuateCcmBtn) # 11

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer_2) # 12

        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(4, 1)
        self.verticalLayout.setStretch(10, 1)
        self.verticalLayout.setStretch(12, 1)

        self.horizontalLayout.addLayout(self.verticalLayout)

        self.horizontalLayout.setStretch(0, 1)

        self.retranslateUi(eCCM)
        self.importRawBtn.clicked.connect(eCCM.import_raw_file)
        self.adjTopOuterBtn.clicked.connect(self.rawPreviewWidget.adj_roi_outertop)
        self.adjBottomOuterBtn.clicked.connect(self.rawPreviewWidget.adj_roi_outerbottom)
        self.adjLeftOuterBtn.clicked.connect(self.rawPreviewWidget.adj_roi_outerleft)
        self.adjRightOuterBtn.clicked.connect(self.rawPreviewWidget.adj_roi_outerright)
        self.adjTopInnerBtn.clicked.connect(self.rawPreviewWidget.adj_roi_innertop)
        self.adjLeftInnerBtn.clicked.connect(self.rawPreviewWidget.adj_roi_innerleft)
        self.adjBottomInnerBtn.clicked.connect(self.rawPreviewWidget.adj_roi_innerbottom)
        self.adjRightInnerBtn.clicked.connect(self.rawPreviewWidget.adj_roi_innerright)
        self.previewBrightSlider.valueChanged.connect(self.rawPreviewWidget.preview_brigh_adj)
        self.previewBrightSlider.valueChanged.connect(eCCM.show_preview_bright)
        self.patchSizeSlider.valueChanged.connect(self.rawPreviewWidget.patch_size_change)
        self.calcuateCcmBtn.clicked.connect(eCCM.calculate_ccm)
        self.importTargetBtn.clicked.connect(eCCM.import_target_file)

        QMetaObject.connectSlotsByName(eCCM)
    # setupUi

    def retranslateUi(self, eCCM):
        eCCM.setWindowTitle(QCoreApplication.translate("eCCM", "eCCM", None))
        self.importRawBtn.setText(QCoreApplication.translate("eCCM", "import raw", None))
        self.label.setText(QCoreApplication.translate("eCCM", "patch size:", None))
        self.groupBox.setTitle(QCoreApplication.translate("eCCM", "adjust ROI", None))
        self.adjTopOuterBtn.setText(QCoreApplication.translate("eCCM", "^", None))
        self.adjLeftOuterBtn.setText(QCoreApplication.translate("eCCM", "<", None))
        self.adjRightOuterBtn.setText(QCoreApplication.translate("eCCM", ">", None))
        self.adjBottomOuterBtn.setText(QCoreApplication.translate("eCCM", "v", None))
        self.adjTopInnerBtn.setText(QCoreApplication.translate("eCCM", "v", None))
        self.adjBottomInnerBtn.setText(QCoreApplication.translate("eCCM", "^", None))
        self.adjLeftInnerBtn.setText(QCoreApplication.translate("eCCM", ">", None))
        self.adjRightInnerBtn.setText(QCoreApplication.translate("eCCM", "<", None))
        self.previewBrightLabel.setText(QCoreApplication.translate("eCCM", "preview brightness:1.0", None))
        self.importTargetBtn.setText(QCoreApplication.translate("eCCM", "import target + gamma", None))
        self.calcuateCcmBtn.setText(QCoreApplication.translate("eCCM", "calculate ccm", None))
    # retranslateUi

