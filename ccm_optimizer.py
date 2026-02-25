"""start main process"""
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from src.form_main import ECcmFrom

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    QApplication.setApplicationName("eCCM")
    QApplication.setOrganizationName("vISP")
    QApplication.setOrganizationDomain("vISP.dev")
    window = ECcmFrom()
    window.show()
    sys.exit(app.exec())
