# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'rawinfo.ui'
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
from PySide6.QtWidgets import (QAbstractButton, QApplication, QComboBox, QDialog,
    QDialogButtonBox, QGridLayout, QLabel, QLineEdit,
    QSizePolicy, QVBoxLayout, QWidget)

class Ui_rawInfoDlg(object):
    def setupUi(self, rawInfoDlg):
        if not rawInfoDlg.objectName():
            rawInfoDlg.setObjectName(u"rawInfoDlg")
        rawInfoDlg.resize(374, 195)
        self.verticalLayout = QVBoxLayout(rawInfoDlg)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_3 = QLabel(rawInfoDlg)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)

        self.bayerComboBox = QComboBox(rawInfoDlg)
        self.bayerComboBox.addItem("")
        self.bayerComboBox.addItem("")
        self.bayerComboBox.addItem("")
        self.bayerComboBox.addItem("")
        self.bayerComboBox.setObjectName(u"bayerComboBox")

        self.gridLayout.addWidget(self.bayerComboBox, 1, 1, 1, 1)

        self.label_4 = QLabel(rawInfoDlg)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)

        self.label = QLabel(rawInfoDlg)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.rawWidthLine = QLineEdit(rawInfoDlg)
        self.rawWidthLine.setObjectName(u"rawWidthLine")

        self.gridLayout.addWidget(self.rawWidthLine, 2, 1, 1, 1)

        self.rawBitComboBox = QComboBox(rawInfoDlg)
        self.rawBitComboBox.addItem("")
        self.rawBitComboBox.addItem("")
        self.rawBitComboBox.addItem("")
        self.rawBitComboBox.addItem("")
        self.rawBitComboBox.addItem("")
        self.rawBitComboBox.addItem("")
        self.rawBitComboBox.addItem("")
        self.rawBitComboBox.addItem("")
        self.rawBitComboBox.addItem("")
        self.rawBitComboBox.setObjectName(u"rawBitComboBox")
        self.rawBitComboBox.setEditable(False)
        self.rawBitComboBox.setMaxCount(32)
        self.rawBitComboBox.setMinimumContentsLength(2)

        self.gridLayout.addWidget(self.rawBitComboBox, 0, 1, 1, 1)

        self.rawHeightLine = QLineEdit(rawInfoDlg)
        self.rawHeightLine.setObjectName(u"rawHeightLine")

        self.gridLayout.addWidget(self.rawHeightLine, 3, 1, 1, 1)

        self.label_2 = QLabel(rawInfoDlg)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.label_5 = QLabel(rawInfoDlg)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)

        self.blcLineEdit = QLineEdit(rawInfoDlg)
        self.blcLineEdit.setObjectName(u"blcLineEdit")

        self.gridLayout.addWidget(self.blcLineEdit, 4, 1, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)

        self.buttonBox = QDialogButtonBox(rawInfoDlg)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.verticalLayout.addWidget(self.buttonBox)


        self.retranslateUi(rawInfoDlg)
        self.buttonBox.accepted.connect(rawInfoDlg.accept)
        self.buttonBox.rejected.connect(rawInfoDlg.reject)

        self.rawBitComboBox.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(rawInfoDlg)
    # setupUi

    def retranslateUi(self, rawInfoDlg):
        rawInfoDlg.setWindowTitle(QCoreApplication.translate("rawInfoDlg", u"raw info", None))
        self.label_3.setText(QCoreApplication.translate("rawInfoDlg", u"width:", None))
        self.bayerComboBox.setItemText(0, QCoreApplication.translate("rawInfoDlg", u"RGGB", None))
        self.bayerComboBox.setItemText(1, QCoreApplication.translate("rawInfoDlg", u"GRBG", None))
        self.bayerComboBox.setItemText(2, QCoreApplication.translate("rawInfoDlg", u"GBRG", None))
        self.bayerComboBox.setItemText(3, QCoreApplication.translate("rawInfoDlg", u"BGGR", None))

        self.label_4.setText(QCoreApplication.translate("rawInfoDlg", u"height:", None))
        self.label.setText(QCoreApplication.translate("rawInfoDlg", u"raw bits:", None))
        self.rawWidthLine.setInputMask(QCoreApplication.translate("rawInfoDlg", u"9000", None))
        self.rawBitComboBox.setItemText(0, QCoreApplication.translate("rawInfoDlg", u"8", None))
        self.rawBitComboBox.setItemText(1, QCoreApplication.translate("rawInfoDlg", u"10", None))
        self.rawBitComboBox.setItemText(2, QCoreApplication.translate("rawInfoDlg", u"12", None))
        self.rawBitComboBox.setItemText(3, QCoreApplication.translate("rawInfoDlg", u"14", None))
        self.rawBitComboBox.setItemText(4, QCoreApplication.translate("rawInfoDlg", u"16", None))
        self.rawBitComboBox.setItemText(5, QCoreApplication.translate("rawInfoDlg", u"18", None))
        self.rawBitComboBox.setItemText(6, QCoreApplication.translate("rawInfoDlg", u"20", None))
        self.rawBitComboBox.setItemText(7, QCoreApplication.translate("rawInfoDlg", u"22", None))
        self.rawBitComboBox.setItemText(8, QCoreApplication.translate("rawInfoDlg", u"24", None))

        self.rawBitComboBox.setCurrentText(QCoreApplication.translate("rawInfoDlg", u"8", None))
        self.rawHeightLine.setInputMask(QCoreApplication.translate("rawInfoDlg", u"9000", None))
        self.label_2.setText(QCoreApplication.translate("rawInfoDlg", u"bayer:", None))
        self.label_5.setText(QCoreApplication.translate("rawInfoDlg", u"blc:", None))
        self.blcLineEdit.setInputMask(QCoreApplication.translate("rawInfoDlg", u"9000", None))
    # retranslateUi

