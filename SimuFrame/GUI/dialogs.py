# Built-in libraries
import dataclasses
from typing import List, Dict, Tuple, Optional, cast

# Third-party libraries
import numpy as np
import numpy.typing as npt
import matplotlib
matplotlib.use('QtAgg')
import pyqtgraph as pg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QWidget, QLabel, QDialogButtonBox,
    QPushButton, QHeaderView, QGroupBox, QComboBox, QFormLayout,
    QListWidget, QSplitter, QFileDialog, QAbstractItemView
)

# Local libraries
from SimuFrame.core.model import Structure
from .interactive.InteractiveCursor import InteractiveCursor


class ConvergenceDialog(QDialog):
    """Dialog to visualize the convergence data of both the Arc-Length and Newton-Raphson methods."""
    def __init__(self, convergence_data: dict, parent=None):
        super().__init__(parent)

        # Convert object to dict to maintain compatibility with .get() calls
        if not isinstance(convergence_data, dict):
            try:
                # If it's a @dataclass
                self.convergence_data = dataclasses.asdict(convergence_data)
            except TypeError:
                # If it's a standard class
                self.convergence_data = vars(convergence_data)
        else:
            self.convergence_data = convergence_data

        self.setWindowTitle("Convergence Overview")
        self.resize(1200, 800)
        self.setup_ui()
        self.populate_data()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Create main tab widget
        self.tab_widget = QTabWidget()

        # Tab 1: Main (General information)
        self.tab_widget.addTab(self._create_main_tab(), "Overview")

        # Tab 2: Table (Table of increments)
        self.tab_widget.addTab(self._create_table_tab(), "Increments")

        # Tab 3: Diagram (Load factor x displacement graph)
        self.tab_widget.addTab(self._create_diagram_tab(), "Diagram")

        # Tab 4: Convergence Table (Detailed convergence table)
        self.tab_widget.addTab(self._create_convergence_tab(), "Convergence details")

        layout.addWidget(self.tab_widget)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        export_btn = QPushButton("Export...")
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        button_layout.addWidget(export_btn)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)

    def _create_main_tab(self):
        """Creates the Overview tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Title
        title = QLabel("Nonlinear Analysis")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        self.info_labels = {}
        info_keys = [
            ("Analysis:", "analysis_type"),
            ("Method:", "method"),
            ("Total Increments:", "total_increments"),
            ("Accepted Increments:", "accepted_increments"),
            ("Rejected Increments:", "rejected_increments"),
            ("Final Load Factor (λ):", "final_lambda"),
            ("Maximum Displacement:", "max_displacement"),
            ("Status:", "status")
        ]

        for label_text, key in info_keys:
            h_layout = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(200)
            value_lbl = QLabel("--")
            value_lbl.setStyleSheet("font-weight: bold;")

            h_layout.addWidget(lbl)
            h_layout.addWidget(value_lbl)
            h_layout.addStretch()
            layout.addLayout(h_layout)
            self.info_labels[key] = value_lbl

        layout.addStretch()
        return widget

    def _create_table_tab(self):
        """Creates the Increments table tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Description
        layout.addWidget(QLabel("<b>Load Increments Summary</b>"))

        # Table
        self.increment_table = QTableWidget()
        headers = ["Step", "λ", "Δλ", "Δs", "Iter.",
                   "||R|| (final)", "||d|| (final)", "||Π|| (final)",
                   "||A|| (final)", "||R + A|| (final)"
                   "Max Displ.", "Status", "Notes"]
        self._setup_table(self.increment_table, headers)
        layout.addWidget(self.increment_table)
        return widget

    def _create_diagram_tab(self):
        """Creates the load factor x displacement diagram."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Description
        layout.addWidget(QLabel("<b>Equilibrium Path - Load Factor × Displacement (norm)</b>"))

        # Graph
        self.plot_widget = pg.PlotWidget(background='w')
        self.plot_widget.setLabel('left', "Load Factor (λ)", color='k')
        self.plot_widget.setLabel('bottom', "Maximum Displacement (m)", color='k')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()

        # Style axes
        for axis in ['left', 'bottom']:
                    self.plot_widget.getAxis(axis).setPen(pg.mkPen(color='k', width=1))
                    self.plot_widget.getAxis(axis).setTextPen(pg.mkPen(color='k'))

        layout.addWidget(self.plot_widget)
        return widget

    def _create_convergence_tab(self):
        """Creates the detailed convergence table tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Description
        layout.addWidget(QLabel("<b>Convergence Details per Iteration</b>"))

        # Detailed convergence table
        self.convergence_table = QTableWidget()
        headers = ["Step", "Iter.", "λ", "Δλ", "δλ", "||R||", "||d||", "||Π||",
                   "||δd||", "||A||", "||R + A||",
                   "Arc-Length Crit.", "Force Crit.", "Displacement Crit.", "Energy Crit.", "Converged"]
        self._setup_table(self.convergence_table, headers)
        layout.addWidget(self.convergence_table)
        return widget

    def _setup_table(self, table: QTableWidget, headers: List[str]):
        """Helper to configure table styling."""
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        table.setAlternatingRowColors(True)

    def populate_data(self):
        """Populate all tabs with the corresponding data."""
        if not self.convergence_data:
            return

        # Initial data
        data = self.convergence_data
        method = data.get('method', "Newton-Raphson (Load Control)")
        is_newton = 'newton' in method.lower()

        # 1. Overview Tab
        self.info_labels['analysis_type'].setText(str(data.get('analysis', "Linear Elastic")))
        self.info_labels['method'].setText(str(method))
        self.info_labels['total_increments'].setText(str(data.get('total_increments', 0)))
        self.info_labels['accepted_increments'].setText(str(data.get('accepted_increments', 0)))
        self.info_labels['rejected_increments'].setText(str(data.get('rejected_increments', 0)))
        self.info_labels['final_lambda'].setText(f"{data.get('final_lambda', 0.0):.3f}")
        self.info_labels['max_displacement'].setText(f"{data.get('max_displacement', 0.0):.3f} m")

        converged = data.get('converged', False)
        status_text = "✓ Converged" if converged else "✗ Failed"
        status_color = "green" if converged else "red"
        self.info_labels['status'].setText(status_text)
        self.info_labels['status'].setStyleSheet(f"font-weight: bold; color: {status_color};")

        # 2. Increments Table
        increments = data.get('increments', [])
        self.increment_table.setRowCount(len(increments))

        for r, inc in enumerate(increments):
            accepted = inc.get('accepted', True)
            status_item = QTableWidgetItem("Accepted" if accepted else "Rejected")
            if not accepted:
                status_item.setForeground(QColor("red"))

            items = [
                str(inc.get('step', '')),
                f"{inc.get('lambda', 0.0):.6f}",
                f"{inc.get('delta_lambda', 0.0):.6e}",
                f"{inc.get('arc_length', 0.0):.6e}",
                str(inc.get('iterations', 0)),
                f"{inc.get('norm_R', 0.0):.2e}",
                f"{inc.get('norm_d', 0.0):.2e}",
                f"{inc.get('norm_E', 0.0):.2e}",
                f"{inc.get('norm_A', 0.0):.2e}",
                f"{inc.get('total_norm', 0.0):.2e}",
                f"{inc.get('max_displacement', 0.0):.6e}",
                status_item,
                inc.get('observations', '')
            ]

            for c, item in enumerate(items):
                if not isinstance(item, QTableWidgetItem):
                    item = QTableWidgetItem(item)
                self.increment_table.setItem(r, c, item)

        # 3. Diagram
        l_hist = data.get('lambda_history', [])
        d_hist = data.get('max_displ_history', [])

        if l_hist and d_hist:
            self.plot_widget.plot(d_hist, l_hist, pen=pg.mkPen('#0096FF', width=3),
                                symbol='o', symbolBrush='#0096FF', name='Equilibrium Path')

            rejected = data.get('rejected_points', {'displ': [], 'lambda': []})
            if rejected['displ']:
                self.plot_widget.plot(rejected['displ'], rejected['lambda'], pen=None,
                                    symbol='x', symbolPen='r', symbolSize=10, name='Rejected')

        # 4. Convergence Table
        details = data.get('iteration_details', [])
        self.convergence_table.setRowCount(len(details))

        for r, d in enumerate(details):
            def create_crit_item(val):
                item = QTableWidgetItem("✓" if val else "✗")
                item.setForeground(QColor("green") if val else QColor("red"))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                return item

            items = [
                str(d.get('step', '')),
                str(d.get('iteration', '')),
                f"{d.get('lambda', 0.0):.6f}",
                f"{d.get('delta_lambda', 0.0):.6e}",
                f"{d.get('delta_lambda_iter', 0.0):.6e}",
                f"{d.get('norm_R', 0.0):.2e}",
                f"{d.get('norm_d', 0.0):.2e}",
                f"{d.get('norm_E', 0.0):.2e}",
                f"{d.get('norm_delta_d', 0.0):.2e}",
                f"{d.get('norm_A', 0.0):.2e}",
                f"{d.get('total_norm', 0.0):.2e}",
                create_crit_item(d.get('arc_length_criterion', False)),
                create_crit_item(d.get('force_criterion', False)),
                create_crit_item(d.get('displ_criterion', False)),
                create_crit_item(d.get('energy_criterion', False)),
            ]

            # Converged Column
            is_conv = d.get('converged', False)
            conv_item = QTableWidgetItem("YES" if is_conv else "NO")
            conv_item.setForeground(QColor("green") if is_conv else QColor("red"))
            conv_item.setFont(QFont("Arial", weight=QFont.Weight.Bold))
            items.append(conv_item)

            for c, item in enumerate(items):
                if not isinstance(item, QTableWidgetItem):
                    item = QTableWidgetItem(item)
                self.convergence_table.setItem(r, c, item)

        # Hide columns specific for one method or vice-versa
        if is_newton:
            self.increment_table.setColumnHidden(3, True)
            self.increment_table.setColumnHidden(8, True)
            self.increment_table.setColumnHidden(9, True)
            self.convergence_table.setColumnHidden(4, True)
            self.convergence_table.setColumnHidden(8, True)
            self.convergence_table.setColumnHidden(9, True)
            self.convergence_table.setColumnHidden(10, True)
            self.convergence_table.setColumnHidden(11, True)
        else:
            self.increment_table.setColumnHidden(6, True)
            self.increment_table.setColumnHidden(7, True)
            self.convergence_table.setColumnHidden(6, True)
            self.convergence_table.setColumnHidden(7, True)

class MplCanvas(FigureCanvas):
    """Create a widget for a Matplotlib plot."""
    def __init__(self, parent=None, width=5, height=4, dpi=120):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.fig.tight_layout()
        super(MplCanvas, self).__init__(self.fig)


class LoadDisplacementDialog(QDialog):
    """Dialog for visualizing load-displacement plots."""

    def __init__(
        self,
        history: List[Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]],
        structure: Structure,
        analysis: str = "Nonlinear",
        parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Initial data
        self.history = history
        self.structure = structure
        self.analysis = analysis
        self.num_nodes = len(structure.original_nodes)
        self.cursors = []

        # Extract data from history
        self.lambda_values = np.array([item[0] for item in history])
        self.displacement_history = np.hstack([item[2] for item in history]).T

        # Axis configuration
        self.dof_names = ['|U|', 'Ux', 'Uy', 'Uz', 'Rx', 'Ry', 'Rz']
        self.dof_units = ['m', 'm', 'm', 'm', 'rad', 'rad', 'rad']
        self.dof_desc = ['Magnitude', 'Disp X', 'Disp Y', 'Disp Z', 'Rot X', 'Rot Y', 'Rot Z']

        # State
        self.selections = {
            'h': {'result': 1, 'dof': 'Ux', 'nodes': [1]},
            'v': {'result': 0, 'dof': None, 'nodes': []}
        }

        self.setWindowTitle("Load vs Displacement")
        self.resize(1400, 900)
        self.setup_ui()
        self.update_plot()

    def setup_ui(self):
        """Setup user interface for the equilibrium path dialog."""
        layout = QVBoxLayout(self)

        # Main widget
        self.tab_widget = QTabWidget()

        # Tab 1: Interactive Diagram
        self.tab_widget.addTab(self._create_interactive_tab(), "Diagram")

        # Tab 2: Data Table
        self.tab_widget.addTab(self._create_table_tab(), "Table")

        # Tab 3: Full Diagram (Full plot)
        self.full_diagram_tab = self.create_full_plot_tab()
        self.tab_widget.addTab(self.full_diagram_tab, "Full plot")

        layout.addWidget(self.tab_widget)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        export_btn = QPushButton("Export...")
        export_btn.clicked.connect(self.export_image)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(export_btn)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)

        # Connect tab change signal to update plots
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Initial State Trigger
        self.ui_h['type'].setCurrentIndex(1) # Default X-Axis: Displacement
        self.ui_v['type'].setCurrentIndex(0) # Default Y-Axis: Load Factor

    def _create_interactive_tab(self) -> QWidget:
        """Create the diagram tab with selection on the left and plot on the right."""
        widget = QWidget()
        main_layout = QHBoxLayout(widget)

        # Create right panel (graph visualization)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Create graph visualization widget
        self.preview_plot = self.create_plot_widget()
        right_layout.addWidget(self.preview_plot)

        # Create left panel (Controls)
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # Create analysis group
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_label = QLabel(self.analysis)
        analysis_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #0066cc;")
        analysis_layout.addWidget(analysis_label)
        left_layout.addWidget(analysis_group)

        # Axis Controls
        self.ui_h = self._create_axis_control_group("Horizontal Axis", 'h', left_layout)
        self.ui_v = self._create_axis_control_group("Vertical Axis", 'v', left_layout)

        left_layout.addStretch()

        # Size adjustment
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)
        return widget

    def _create_axis_control_group(self, title: str, axis_key: str, parent_layout: QVBoxLayout) -> Dict:
        """Helper to create identical control groups for H and V axes."""
        group = QGroupBox(title)
        layout = QVBoxLayout(group)

        # Result Type
        h_res = QHBoxLayout()
        h_res.addWidget(QLabel("Type:"))
        combo_type = QComboBox()
        combo_type.addItems(["Load Factor (λ)", "Global Displacements"])
        h_res.addWidget(combo_type)
        layout.addLayout(h_res)

        # Data Selection (Hidden by default if Load Factor)
        layout_data = QHBoxLayout()
        lbl_data = QLabel("DOF:")
        layout_data.addWidget(lbl_data)
        combo_dof = QComboBox()
        for name, desc in zip(self.dof_names, self.dof_desc):
            combo_dof.addItem(f"{name} - {desc}", name)
        layout_data.addWidget(combo_dof, 1)
        layout.addLayout(layout_data)

        # Node Selection
        lbl_nodes = QLabel("Nodes:")
        list_nodes = QListWidget()
        list_nodes.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        list_nodes.setMaximumHeight(150)
        for i in range(1, self.num_nodes + 1):
            list_nodes.addItem(f"Node {i}")

        if axis_key == 'h':
            list_nodes.item(0).setSelected(True)

        layout.addWidget(lbl_nodes)
        layout.addWidget(list_nodes)

        parent_layout.addWidget(group)

        # Logic / Connections
        def update_visibility():
            is_displ = (combo_type.currentIndex() == 1)

            # Show/Hide widgets
            lbl_data.setVisible(is_displ)
            combo_dof.setVisible(is_displ)
            lbl_nodes.setVisible(is_displ)
            list_nodes.setVisible(is_displ)

            # Update internal state dictionary
            self.selections[axis_key]['result'] = combo_type.currentIndex()
            if is_displ:
                self.selections[axis_key]['dof'] = combo_dof.currentData()
                self.selections[axis_key]['nodes'] = [
                    int(item.text().split()[1]) for item in list_nodes.selectedItems()
                ]
            else:
                self.selections[axis_key]['dof'] = None
                self.selections[axis_key]['nodes'] = []

            # Refresh plots/tables
            self.update_plot()
            self.update_table()

        # Connect signals and force initial update
        combo_type.currentIndexChanged.connect(update_visibility)
        combo_dof.currentIndexChanged.connect(update_visibility)
        list_nodes.itemSelectionChanged.connect(update_visibility)
        update_visibility()

        # Return handles to widgets
        return {
            'type': combo_type,
            'dof_lbl': lbl_data,
            'dof_combo': combo_dof,
            'nodes_lbl': lbl_nodes,
            'nodes_list': list_nodes
        }

    def _create_table_tab(self):
        """Create table tab with graph data."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Description
        desc_label = QLabel("Graph Data Points")
        desc_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(desc_label)

        # Table
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        layout.addWidget(self.data_table)

        return widget

    def create_full_plot_tab(self):
        """Create full plot tab with the selected data in tab 1."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create full plot widget
        self.full_plot = self.create_plot_widget()
        layout.addWidget(self.full_plot)

        return widget

    def create_plot_widget(self):
        """Creates a Matplotlib canvas widget."""
        return MplCanvas(self, width=5, height=4, dpi=100)

    def update_plot(self):
        """Triggers a redraw of the preview plot."""
        self._draw_plot(self.preview_plot.axes)

    def update_full_plot(self):
        """Triggers a redraw of the full-screen plot."""
        self._draw_plot(self.full_plot.axes, is_full_plot=True)

    def on_tab_changed(self, index):
        """Updates the full plot when the user switches to that tab."""
        if index == 2:  # Full Plot Tab
            self.update_full_plot()

    def _draw_plot(self, ax, is_full_plot: bool = False):
        """
        Main plotting logic.
        Draws the Load Factor vs Displacement/Rotation graph on the provided axes.
        """
        # Cleanup of old cursors
        self._clear_cursor(ax)
        ax.clear()

        # Fetch Data
        x_series, x_axis_label = self._get_axis_data('h')
        y_series, y_axis_label = self._get_axis_data('v')

        if not x_series or not y_series:
            ax.figure.canvas.draw()
            return

        # Plotting Loop
        colors = matplotlib.pyplot.get_cmap('tab10').colors # type: ignore
        color_idx = 0

        all_x_points = []
        all_y_points = []
        cursor_lines = []

        # Cartesian product: Plot every selected X against every selected Y
        for x_item in x_series:
            for y_item in y_series:
                # Avoid plotting a variable against itself (identity check)
                if x_item['label'] == y_item['label']:
                    continue

                # Construct Legend
                # If X is just Load Factor (common case), we only label based on Y
                if "Load Factor" in x_item['label']:
                    legend_label = f"SimuFrame - {y_item['label']}"
                else:
                    legend_label = f"SimuFrame - {y_item['label']} vs {x_item['label']}"

                current_color = colors[color_idx % len(colors)]

                # Plot
                ax.plot(
                    x_item['data'],
                    y_item['data'],
                    marker='o',
                    markersize=4,
                    linestyle='-',
                    label=legend_label,
                    color=current_color
                )

                # Store data for interactive cursor and limits
                cursor_lines.append({
                    'x': x_item['data'],
                    'y': y_item['data'],
                    'label': legend_label,
                    'color': current_color
                })

                all_x_points.append(x_item['data'])
                all_y_points.append(y_item['data'])
                color_idx += 1

        # Axis Formatting
        # Auto-center if variation is negligible (e.g., zero displacement)
        tol = 1e-6
        if all_x_points:
            x_concat = np.concatenate(all_x_points)
            if np.ptp(x_concat) < tol:
                mean_x = np.mean(x_concat)
                ax.set_xlim(mean_x - tol, mean_x + tol)

        if all_y_points:
            y_concat = np.concatenate(all_y_points)
            if np.ptp(y_concat) < tol:
                mean_y = np.mean(y_concat)
                ax.set_ylim(mean_y - tol, mean_y + tol)

        ax.set_xlabel(x_axis_label, fontsize=11, weight='bold')
        ax.set_ylabel(y_axis_label, fontsize=11, weight='bold')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

        # Only create a legend if there are actually labeled items plotted
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.margins(x=0.05, y=0.05)
        ax.figure.tight_layout()

        # Interactive Cursor
        if cursor_lines:
            try:
                # Assuming InteractiveCursor is imported/defined elsewhere
                precision = self._determine_precision(all_x_points, all_y_points)
                cursor = InteractiveCursor(
                    ax,
                    cursor_lines,
                    x_label=x_axis_label,
                    y_label=y_axis_label,
                    precision=precision
                )
                self.cursors.append((ax, cursor))
            except NameError:
                pass

        ax.figure.canvas.draw()

    def _get_axis_data(self, axis_key: str) -> Tuple[List[dict], str]:
        """
        Helper to extract data series based on current UI selections.
        Returns: (List of dicts with 'data' and 'label', Axis Title String)
        """
        sel = self.selections[axis_key]
        series_list = []
        axis_title = ""

        # Case 1: Load Factor
        if sel['result'] == 0:
            series_list.append({
                'data': self.lambda_values,
                'label': "Load Factor (λ)"
            })
            axis_title = "Load Factor (λ)"

        # Case 2: Displacements
        elif sel['result'] == 1 and sel['dof']:
            dof_idx = self.dof_names.index(str(sel['dof']))
            unit = self.dof_units[dof_idx]

            # Initialize selected nodes
            selected_nodes = cast(List[int], sel['nodes'])

            for node_id in selected_nodes:
                # Calculate column index
                col_idx = (node_id - 1) * 6 + (dof_idx - 1)
                if sel['dof'] == '|U|':
                    # Get translational displacement indices
                    ux_idx = (node_id - 1) * 6 + 0
                    uy_idx = (node_id - 1) * 6 + 1
                    uz_idx = (node_id - 1) * 6 + 2

                    # Translational displacements values
                    ux = self.displacement_history[:, ux_idx]
                    uy = self.displacement_history[:, uy_idx]
                    uz = self.displacement_history[:, uz_idx]

                    # Calculate magnitude of displacement vector
                    node_data = np.sqrt(ux**2 + uy**2 + uz**2)
                else:
                    # Direct DOF access
                    if col_idx < self.displacement_history.shape[1]:
                        node_data = self.displacement_history[:, col_idx]
                    else:
                        node_data = np.zeros_like(self.lambda_values)

                series_list.append({
                    'data': node_data,
                    'label': f"Node {node_id} - {sel['dof']}"
                })

            axis_title = f"{sel['dof']} [{unit}]"

        return series_list, axis_title

    def _determine_precision(self, x_arrays, y_arrays) -> int:
        """Determines ideal decimal precision based on data magnitude."""
        if not x_arrays or not y_arrays:
            return 4

        # Concatenate all data arrays
        all_values = np.concatenate(x_arrays + y_arrays)

        # Calculate the maximum absolute value
        max_val = np.max(np.abs(all_values))
        if max_val == 0:
            return 4

        # Calculate magnitude using log10
        magnitude = np.floor(np.log10(max_val))

        if magnitude >= 3:
            return 3    # >= 1000
        if magnitude >= 0:
           return 4     # 1 to 999
        if magnitude >= -3:
            return 5    # 0.001 to 1
        return 6        # < 0.001

    def _clear_cursor(self, ax):
        """Removes the interactive cursor from the specific axes."""
        # Filter self.cursors, disconnecting the one that matches 'ax'
        remaining_cursors = []
        for stored_ax, cursor in self.cursors:
            if stored_ax is ax:
                try:
                    cursor.disconnect()
                except AttributeError:
                    pass
            else:
                remaining_cursors.append((stored_ax, cursor))

        self.cursors = remaining_cursors

    def update_table(self):
        """Refreshes the data table based on the plotted series."""
        if not hasattr(self, 'data_table'):
            return

        # Re-fetch data to ensure table matches plot exactly
        x_series, _ = self._get_axis_data('h')
        y_series, _ = self._get_axis_data('v')

        if not x_series or not y_series:
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(0)
            return

        # Prepare headers and columns
        headers = ["Step"]
        columns = [np.arange(len(self.lambda_values))]
        seen_labels = set()

        for series in x_series + y_series:
            lbl = series['label']
            if lbl not in seen_labels:
                headers.append(lbl)
                columns.append(series['data'])
                seen_labels.add(lbl)

        # Update Table Widget
        num_rows = len(self.lambda_values)
        self.data_table.setRowCount(num_rows)
        self.data_table.setColumnCount(len(headers))
        self.data_table.setHorizontalHeaderLabels(headers)

        # Batch populate
        for col_idx, col_data in enumerate(columns):
            for row_idx, value in enumerate(col_data):
                fmt = f"{value:.6e}" if isinstance(value, (float, np.floating)) else str(value)
                self.data_table.setItem(row_idx, col_idx, QTableWidgetItem(fmt))

        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def export_image(self):
        """Exports the full plot to a file."""
        if not self.full_plot:
            return

        self.update_full_plot()

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Chart",
            "",
            "Images (*.png *.jpg);;Vector Graphics (*.svg *.pdf)"
        )

        if file_path:
            try:
                self.full_plot.fig.savefig(file_path, bbox_inches='tight', dpi=300)
            except Exception as e:
                print(f"Error saving file: {e}")

    def closeEvent(self, event):
        """Ensure clean shutdown of cursors to prevent memory leaks."""
        for _, cursor in self.cursors:
            try:
                cursor.disconnect()
            except Exception:
                pass
        self.cursors.clear()
        super().closeEvent(event)

class AnalysisParametersDialog(QDialog):
    """Dialog for editing structural analysis parameters."""

    def __init__(self, current_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Analysis Parameters")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(450)
        self.setMinimumHeight(550)

        # Define dictionaries
        self.current_params = current_params
        self.config = current_params.get('config', {})

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)

        self._create_mesh_section(main_layout)
        self._create_analysis_section(main_layout)
        self._create_button_box(main_layout)

        # Initialize visibility states
        self._update_visibility()

        self._apply_stylesheet()

    def _create_mesh_section(self, parent_layout):
        """Create mesh parameters section."""
        mesh_group = QGroupBox("Mesh")
        mesh_layout = QFormLayout()

        self.mesh_param_spin = QDoubleSpinBox()
        self.mesh_param_spin.setRange(0.01, 512.0)
        self.mesh_param_spin.setDecimals(2)
        self.mesh_param_spin.setValue(self.current_params.get('mesh_parameter', 1.0))
        mesh_layout.addRow("Mesh parameter:", self.mesh_param_spin)

        mesh_group.setLayout(mesh_layout)
        parent_layout.addWidget(mesh_group)

    def _create_analysis_section(self, parent_layout):
        """Create analysis parameters section."""
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout()

        # Analysis type selector
        type_layout = QFormLayout()
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems(['Linear', 'Nonlinear', 'Buckling'])
        self.analysis_combo.setCurrentText(
            self.current_params.get('analysis_type', 'Linear').capitalize()
        )
        self.analysis_combo.currentTextChanged.connect(self._update_visibility)
        type_layout.addRow("Type:", self.analysis_combo)
        analysis_layout.addLayout(type_layout)

        # Create all parameter widgets
        self._create_nonlinear_params(analysis_layout)
        self._create_buckling_params(analysis_layout)

        analysis_group.setLayout(analysis_layout)
        parent_layout.addWidget(analysis_group)
        parent_layout.addStretch()  # Push everything to top

    def _create_nonlinear_params(self, parent_layout):
        """Create nonlinear analysis parameters."""
        self.nonlinear_widget = QWidget()
        nonlinear_layout = QVBoxLayout(self.nonlinear_widget)
        nonlinear_layout.setContentsMargins(0, 10, 0, 0)
        nonlinear_layout.setSpacing(10)

        # Method selector
        method_layout = QFormLayout()
        self.method_combo = QComboBox()
        self.method_combo.addItems(['Newton-Raphson', 'Arc-Length'])
        self.method_combo.setCurrentText(self.config.get('method', 'Newton-Raphson'))
        self.method_combo.currentTextChanged.connect(self._update_arc_length_visibility)
        method_layout.addRow("Method:", self.method_combo)
        nonlinear_layout.addLayout(method_layout)

        # Common parameters
        common_layout = QFormLayout()

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 1000)
        self.steps_spin.setValue(self.config.get('initial_steps', 1))
        common_layout.addRow("Initial steps:", self.steps_spin)

        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 1000)
        self.max_iter_spin.setValue(self.config.get('max_iterations', 50))
        common_layout.addRow("Max iterations:", self.max_iter_spin)

        nonlinear_layout.addLayout(common_layout)

        # Arc-Length specific parameters
        self._create_arc_length_params(nonlinear_layout)

        parent_layout.addWidget(self.nonlinear_widget)

    def _create_arc_length_params(self, parent_layout):
        """Create Arc-Length specific parameters."""
        self.arc_length_widget = QWidget()
        arc_layout = QFormLayout()
        arc_layout.setSpacing(8)

        self.load_factor_spin = QDoubleSpinBox()
        self.load_factor_spin.setRange(0.001, 1000.0)
        self.load_factor_spin.setDecimals(3)
        self.load_factor_spin.setValue(self.config.get('max_load_factor', 1.0))
        arc_layout.addRow("Max load factor:", self.load_factor_spin)

        self.arc_type_combo = QComboBox()
        self.arc_type_combo.addItems(['Spherical', 'Cylindrical', 'Custom'])
        self.arc_type_combo.setCurrentText(
            self.config.get('arc_type', 'Spherical').capitalize()
        )
        self.arc_type_combo.currentTextChanged.connect(self._update_psi_visibility)
        arc_layout.addRow("Arc type:", self.arc_type_combo)

        self.psi_spin = QDoubleSpinBox()
        self.psi_spin.setRange(0.0, 1.0)
        self.psi_spin.setDecimals(3)
        self.psi_spin.setSingleStep(0.1)
        self.psi_spin.setValue(self.config.get('phi', 1.0))
        self.psi_label = QLabel("Psi:")
        arc_layout.addRow(self.psi_label, self.psi_spin)

        self.arc_length_widget.setLayout(arc_layout)
        parent_layout.addWidget(self.arc_length_widget)

    def _create_buckling_params(self, parent_layout):
        """Create buckling analysis parameters."""
        self.buckling_widget = QWidget()
        buckling_layout = QFormLayout()

        self.buckling_modes_spin = QSpinBox()
        self.buckling_modes_spin.setRange(1, 100)
        self.buckling_modes_spin.setValue(self.config.get('buckling_modes', 5))
        buckling_layout.addRow("Buckling modes:", self.buckling_modes_spin)

        self.buckling_widget.setLayout(buckling_layout)
        parent_layout.addWidget(self.buckling_widget)

    def _create_button_box(self, parent_layout):
        """Create dialog button box."""
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        parent_layout.addWidget(button_box)

    def _update_visibility(self):
        """Update widget visibility based on analysis type."""
        analysis_type = self.analysis_combo.currentText()

        is_nonlinear = analysis_type == 'Nonlinear'
        is_buckling = analysis_type == 'Buckling'

        self.nonlinear_widget.setVisible(is_nonlinear)
        self.buckling_widget.setVisible(is_buckling)

        if is_nonlinear:
            self._update_arc_length_visibility()

    def _update_arc_length_visibility(self):
        """Update Arc-Length specific parameters visibility."""
        is_arc_length = self.method_combo.currentText() == 'Arc-Length'
        self.arc_length_widget.setVisible(is_arc_length)

        if is_arc_length:
            self._update_psi_visibility()

    def _update_psi_visibility(self):
        """Update Psi parameter visibility based on arc type."""
        is_custom = self.arc_type_combo.currentText() == 'Custom'
        self.psi_label.setVisible(is_custom)
        self.psi_spin.setVisible(is_custom)

    def get_parameters(self):
        """Extract and return current parameter values."""
        analysis_type = self.analysis_combo.currentText().lower()

        params = {
            'analysis_type': analysis_type,
            'mesh_parameter': self.mesh_param_spin.value(),
            'config': {}
        }

        if analysis_type == 'nonlinear':
            params['config'] = {
                'method': self.method_combo.currentText(),
                'initial_steps': self.steps_spin.value(),
                'max_iterations': self.max_iter_spin.value(),
            }

            if self.method_combo.currentText() == 'Arc-Length':
                params['config'].update({
                    'max_load_factor': self.load_factor_spin.value(),
                    'arc_type': self.arc_type_combo.currentText().lower(),
                })

                if self.arc_type_combo.currentText() == 'Custom':
                    params['config']['phi'] = self.psi_spin.value()

        elif analysis_type == 'buckling':
            params['config'] = {
                'buckling_modes': self.buckling_modes_spin.value()
            }

        return params

    def _apply_stylesheet(self):
        """Apply dialog stylesheet."""
        self.setStyleSheet("""
            QDialog {
                background-color: #f8fafc;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
                color: #1e293b;
            }
            QGroupBox {
                font-weight: 600;
                font-size: 14px;
                color: #1e293b;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 16px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 8px;
                background-color: #ffffff;
            }
            QLabel {
                font-weight: 500;
                color: #475569;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 6px 10px;
                min-width: 150px;
                color: #1e293b;
                selection-background-color: #3b82f6;
            }
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border: 1px solid #94a3b8;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #3b82f6;
                padding: 5px 9px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 0px;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            QPushButton {
                font-weight: 600;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                min-width: 80px;
            }
            QPushButton[text="OK"], QPushButton[text="&OK"] {
                background-color: #3b82f6;
                color: white;
                border: 1px solid #2563eb;
            }
            QPushButton[text="OK"]:hover, QPushButton[text="&OK"]:hover {
                background-color: #2563eb;
            }
            QPushButton[text="OK"]:pressed, QPushButton[text="&OK"]:pressed {
                background-color: #1d4ed8;
            }
            QPushButton[text="Cancel"], QPushButton[text="&Cancel"] {
                background-color: #ffffff;
                color: #475569;
                border: 1px solid #cbd5e1;
            }
            QPushButton[text="Cancel"]:hover, QPushButton[text="&Cancel"]:hover {
                background-color: #f1f5f9;
                border-color: #94a3b8;
                color: #1e293b;
            }
        """)

    def on_analysis_type_changed(self, text):
        self.nonlinear_widget.setVisible(text == 'Nonlinear')
        self.buckling_widget.setVisible(text == 'Buckling')

    def on_method_changed(self, text):
        self.arc_length_widget.setVisible(text == 'Arc-Length')

    def on_arc_type_changed(self, text):
        if text == 'Spherical':
            self.psi_spin.setValue(1.0)
            self.psi_spin.setEnabled(False)
        elif text == 'Cylindrical':
            self.psi_spin.setValue(0.0)
            self.psi_spin.setEnabled(False)
        else:
            self.psi_spin.setEnabled(True)

    def get_parameters(self):
        params = {
            'mesh_parameter': self.mesh_param_spin.value(),
            'analysis_type': self.analysis_combo.currentText()
        }

        if self.analysis_combo.currentText() == 'Nonlinear':
            params['method'] = self.method_combo.currentText()
            params['initial_steps'] = self.steps_spin.value()
            params['max_iterations'] = self.max_iter_spin.value()

            if self.method_combo.currentText() == 'Arc-Length':
                params['max_load_factor'] = self.load_factor_spin.value()
                params['arc_type'] = self.arc_type_combo.currentText()
                params['phi'] = self.psi_spin.value()

        if self.analysis_combo.currentText() == 'Buckling':
            params['buckling_modes'] = self.buckling_modes_spin.value()

        return params
