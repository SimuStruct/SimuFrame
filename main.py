# Built-in libraries
import sys
import os

# Third-party libraries
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

# Author's libraries
from SimuFrame.GUI.main_window import MainApplicationWindow

# Absolute path to the icon
icon_path = os.path.join(
    os.path.dirname(__file__),
    "SimuFrame",
    "GUI",
    "assets",
    "icon.ico"
)

# Define the app ID for Windows
if os.name == 'nt':
    import ctypes
    myappid = 'SimuFrame.0.3.1'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

# Initialize the UI
if __name__ == "__main__":
    # App initialization
    app = QApplication.instance() or QApplication(sys.argv)

    # UI initialization
    window = MainApplicationWindow()
    window.setWindowIcon(QIcon(icon_path))

    # App execution
    window.show()
    sys.exit(app.exec())
