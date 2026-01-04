# Built-in libraries
import time
from enum import Enum
from typing import Any, List

# Third-party libraries
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


class AnalysisType(Enum):
    NEWTON_RAPHSON = "Newton-Raphson (Load Control)"
    ARC_LENGTH = "Arc-Length (Kadapa)"

class SolverVisualizer:
    """
    Visualizer for both Newton-Raphson and Arc-Length methods.
    Shows the equilibrium path in real-time with status indicators.
    """

    def __init__(self, analysis_type: AnalysisType, show_window: bool = True):
        """
        Args:
            show_window (bool): If False, does not create a window (headless mode)
        """
        self.show_window = show_window
        self.analysis_type = analysis_type

        if not show_window:
            self.app = None
            return

        # Create Qt application
        self.app = pg.mkQApp(f"Nonlinea Analysis - {analysis_type.value}")

        # Create main window
        self.win = QtWidgets.QDialog()
        self.win.setWindowTitle(f"Equilibrium Path - {analysis_type.value}")
        self.win.resize(900, 700)

        # Create main layout
        main_layout = QtWidgets.QVBoxLayout(self.win)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Window title
        title_label = QtWidgets.QLabel(f"Nonlinear Analysis - {analysis_type.value}")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(title_label)

        # Graph layout
        plot_layout = QtWidgets.QHBoxLayout()

        # Create plot widget
        plot_widget: Any = pg.GraphicsLayoutWidget()
        self.plot = plot_widget.addPlot(title="Equilibrium Path")
        self.plot.setLabel(
            "bottom", "Maximum displacement (m)", **{"font-size": "12pt"}
        )
        self.plot.setLabel("left", "Load factor (λ)", **{"font-size": "12pt"})
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        # Accepted equilibrium path (main curve)
        self.main_curve = self.plot.plot(
            pen=pg.mkPen(color="#3498db", width=3, style=QtCore.Qt.PenStyle.SolidLine),
            symbol="o",
            symbolSize=8,
            symbolBrush=pg.mkBrush("#3498db"),
            symbolPen=pg.mkPen(color="#2980b9", width=2),
            antialias=True,
        )

        # Shadow curve
        self.shadow_curve = self.plot.plot(
            pen=pg.mkPen(color="#bdc3c7", width=5), antialias=True
        )

        # Rejected equilibrium points
        self.rejected_points = self.plot.plot(
            pen=None,
            symbol="x",
            symbolSize=12,
            symbolBrush="#e74c3c",
            symbolPen=pg.mkPen(color="#e74c3c", width=2),
        )

        # Reference lines
        self.vertical_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen("#95a5a6", style=QtCore.Qt.PenStyle.DashLine, width=2),
        )
        self.horizontal_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen("#95a5a6", style=QtCore.Qt.PenStyle.DashLine, width=2),
        )

        # Information label of the current step point
        self.point_label = pg.TextItem(
            anchor=(0, 1),
            color="#2c3e50",
            fill=pg.mkBrush(255, 255, 255, 240),
            border=pg.mkPen("#3498db", width=2),
        )
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setFamily("Arial")
        self.point_label.setFont(font)

        # Adicionar elements ao gráfico
        self.plot.addItem(self.vertical_line)
        self.plot.addItem(self.horizontal_line)
        self.plot.addItem(self.point_label)
        plot_layout.addWidget(plot_widget)

        # Status bar
        self.status_bar = self._create_status_bar()
        plot_layout.addWidget(self.status_bar)
        main_layout.addLayout(plot_layout)

        # Info panel
        info_panel = self._create_info_panel()
        main_layout.addWidget(info_panel)

        # Show window
        if self.app:
            self.win.show()
            self.app.processEvents()

    def _create_status_bar(self) -> QtWidgets.QWidget:
        """Create colored status bar."""
        status_bar = QtWidgets.QWidget()
        status_bar.setAutoFillBackground(True)
        status_bar.setFixedWidth(30)
        status_bar.setMinimumHeight(400)

        # Tooltip
        status_bar.setToolTip("Green: Converging\nOrange: Rejected\nRed: Waiting")

        # Set initial color as red (Waiting)
        self.set_status_color(status_bar, "#e74c3c")
        return status_bar

    def set_status_color(self, status_bar: QtWidgets.QWidget, color: str):
        """Set the color of the status bar."""
        # Animation smooth transition
        animation = QtCore.QPropertyAnimation(status_bar, b"styleSheet")
        animation.setDuration(300)  # 300ms
        animation.setStartValue(
            f"background-color: {status_bar.palette().color(QtGui.QPalette.ColorRole.Window).name()};"
        )
        animation.setEndValue(f"background-color: {color}; border-radius: 5px;")
        animation.start()

        palette = status_bar.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(color))
        status_bar.setPalette(palette)

        # Store reference to prevent destruction
        if not hasattr(self, "_animations"):
            self._animations = []
        self._animations.append(animation)

    def _create_info_panel(self) -> QtWidgets.QWidget:
        """Create real-time information panel."""
        panel = QtWidgets.QGroupBox("Step Data")
        panel.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        layout = QtWidgets.QGridLayout(panel)

        # Informations labels
        self.info_labels = {}

        arc_length_items = [
            ("Step:", "step"),
            ("Iterations:", "iterations"),
            ("Lambda (λ):", "lambda"),
            ("Δs (current):", "delta_s"),
            ("||R||:", "residue"),
            ("Status:", "status"),
        ]

        newton_raphson_items = [
            ("Step:", "step"),
            ("Iterations:", "iterations"),
            ("Load Factor (λ):", "lambda"),
            ("||R||:", "residue"),
            ("Status:", "status"),
        ]

        # Select items based on analysis type
        items = (
            arc_length_items
            if self.analysis_type == AnalysisType.ARC_LENGTH
            else newton_raphson_items
        )

        for row, (label_text, key) in enumerate(items):
            label = QtWidgets.QLabel(label_text)
            label.setStyleSheet("font-weight: bold; font-size: 12px;")

            value = QtWidgets.QLabel("--")
            value.setStyleSheet("font-size: 12px; color: #34495e;")

            layout.addWidget(label, row, 0)
            layout.addWidget(value, row, 1)

            self.info_labels[key] = value

        # Add progress bar
        progress_label = QtWidgets.QLabel("Progress:")
        progress_label.setStyleSheet("font-weight: bold; font-size: 12px;")

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2ecc71);
                border-radius: 3px;
            }
        """)

        layout.addWidget(progress_label, len(items), 0)
        layout.addWidget(self.progress_bar, len(items), 1)

        return panel

    def update(
        self,
        max_displ_history: List[float],
        lambda_history: List[float],
        step: int,
        iter: int,
        λ: float,
        max_displ: float,
        delta_s: float = 0.0,
        residue: float = 0.0,
        allow_lambda_exceed: bool = False,
        max_lambda: float = 1.0,
    ) -> None:
        """
        Update visualization with new data.

        Args:
            max_displ_history (list): Displacement history.
            lambda_history (list): Lambda (load factor) history.
            step (int): Current step number.
            iter (int): Current iteration number.
            λ (float): Current lambda (load factor) value.
            max_displ (float): Current maximum displacement.
            delta_s (float): Current arc length.
            residue (float): Current residual norm.
        """
        if not self.show_window:
            return

        # Update main curve
        self.main_curve.setData(max_displ_history, lambda_history)

        # Update reference lines
        self.vertical_line.setValue(max_displ)
        self.horizontal_line.setValue(float(λ))

        # Update point label
        if self.analysis_type == AnalysisType.NEWTON_RAPHSON:
            label = f"Step: {step}\nIter: {iter}\nλ = {λ:.6f}"
        else:
            label = f"Step: {step}\nIter: {iter}\nλ = {λ:.6f}\nΔs = {delta_s:.6e}"
        self.point_label.setText(label)

        # Update position of point label
        x_pos = max_displ * 0.8
        y_pos = float(λ) * 0.8
        self.point_label.setPos(x_pos, y_pos)

        # Update info labels
        self.info_labels["step"].setText(f"{step}")
        self.info_labels["iterations"].setText(f"{iter}")
        self.info_labels["lambda"].setText(f"{λ:.6f}")
        if self.analysis_type == AnalysisType.ARC_LENGTH:
            self.info_labels["delta_s"].setText(f"{delta_s:.6e}")
        self.info_labels["residue"].setText(f"{residue:.6e}")
        self.info_labels["status"].setText("✓ Convergindo")
        self.info_labels["status"].setStyleSheet(
            "font-size: 12px; color: #27ae60; font-weight: bold;"
        )

        # Change status to converging (green)
        self.set_status_color(self.status_bar, "#27ae60")

        # Update progress bar based on the value of λ
        if self.analysis_type == AnalysisType.NEWTON_RAPHSON:
            progress_percent = int(λ * 100)
        else:
            if allow_lambda_exceed:
                progress_percent = int(min((λ / max_lambda) * 100, 100))
            else:
                progress_percent = int(λ * 100)

        self.progress_bar.setValue(progress_percent)
        self.progress_bar.setFormat(f"{progress_percent}% (λ = {λ:.4f})")

        # Process events
        if self.app:
            self.app.processEvents()

    def show_failure(
        self,
        max_displ: float = 0.0,
        λ: float = 0.0,
        rejected_displ: List[float] = [],
        rejected_lambda: List[float] = [],
    ):
        """
        Update the visualizer to show a failed step and/or iteration.

        Args:
            max_displ (float): Maximum displacement of the rejected point.
            λ (float): Lambda of the rejected point.
            rejected_displ (list): List of rejected displacements.
            rejected_lambda (list): List of rejected lambdas.
        """
        if not self.show_window:
            return

        # Update rejected points
        if max_displ > 0 and λ > 0:
            if not rejected_displ:
                rejected_displ = [max_displ]
                rejected_lambda = [λ]
            else:
                rejected_displ.append(max_displ)
                rejected_lambda.append(λ)

        # Update rejected points
        if rejected_displ and rejected_lambda:
            self.rejected_points.setData(rejected_displ, rejected_lambda)

        # Set color to orange (rejected)
        self.set_status_color(self.status_bar, "#e67e22")

        # Update info labels
        self.info_labels["status"].setText("✗ Rejected")
        self.info_labels["status"].setStyleSheet(
            "font-size: 12px; color: #e74c3c; font-weight: bold;"
        )

        if self.app:
            self.app.processEvents()

    def finalize(
        self,
        converged: bool,
        final_lambda: float,
        total_increments: int,
        accepted_increments: int,
    ):
        """
        End the visualization process and show the final results.

        Args:
            converged (bool): Check if the analysis converged.
            final_lambda (float): Final lambda value.
            total_increments (int): Total number of increments attempted.
            accepted_increments (int): Accepted increments.
        """
        if not self.show_window:
            return

        # Final color of the status bar
        if converged:
            self.set_status_color(self.status_bar, "#27ae60")  # Green
            status_text = "✓ Done"
            status_color = "#27ae60"
        else:
            self.set_status_color(self.status_bar, "#e74c3c")  # Red
            status_text = "✗ Failed"
            status_color = "#e74c3c"

        # Update info labels
        self.info_labels["status"].setText(status_text)
        self.info_labels["status"].setStyleSheet(
            f"font-size: 12px; color: {status_color}; font-weight: bold;"
        )

        # Add final text to the main curve
        final_text = (
            f"ANALYSIS COMPLETED\n"
            f"λ_final = {final_lambda:.6f}\n"
            f"Steps: {accepted_increments}/{total_increments}"
        )

        self.point_label.setText(final_text)

        if self.app:
            self.app.processEvents()

    def wait_and_close(self, delay_sec: float):
        """
        Wait for delay_sec seconds and close the window.

        Args:
            delay_sec (float): Time in seconds to wait before closing the window.
        """
        if not self.app:
            return

        end_time = time.time() + delay_sec
        while time.time() < end_time:
            self.app.processEvents()
            time.sleep(0.01)

        # Clean up resources and close the window
        if hasattr(self, 'main_curve'):
            self.main_curve.clear()

        if hasattr(self, 'rejected_points'):
            self.rejected_points.clear()
        self.win.close()
        self.app.processEvents()
