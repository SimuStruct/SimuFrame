# Built-in libraries
from typing import cast

# Third-party libraries
import qtawesome as qta
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QAction, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QFileDialog, QLabel, QMessageBox,
    QTreeWidget, QApplication, QStatusBar, QDockWidget, QTableWidget,
    QTabWidget, QHeaderView, QDialog, QTreeWidgetItem, QDoubleSpinBox,
    QRadioButton, QCheckBox, QComboBox
)

# Local libraries
from SimuFrame.post_processing.visualization import plot_structure
from .controller import SimuController
from .dialogs import (
    ConvergenceDialog, LoadDisplacementDialog, AnalysisParametersDialog
)
from .plot_manager import PlotManager
from .results_tree_manager import ResultsTreeManager
from .state_manager import StateManager
from .ui_populators import (
    DataTreePopulator, TablesPopulator, ResultsTreePopulator, MeshLoader
)
from .widgets import (
    _create_table_tab, _create_sections_tab, create_scale_panel,
    create_results_tree, create_options_panel, setup_view_toolbar,
    setup_buckling_toolbar
)


class MainApplicationWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # Initialize state manager
        self.state = StateManager()

        # Setup UI
        self._setup_window()
        self._create_central_widget()
        self._setup_docks()
        self._setup_toolbars()

        # Create managers
        self.plot_manager = PlotManager(self.plotter, self.state)
        self.controller = SimuController(self)
        self.results_tree_manager = ResultsTreeManager(self.results_tree)

        # Setup menus and statusbar
        self._setup_menus()
        self._setup_statusbar()

        # Connect signals
        self._connect_signals()

        # Set initial state
        self._set_initial_state()

        # Create widgets (type checking)
        self.scale_spinbox: QDoubleSpinBox
        self.radio_diagram: QRadioButton
        self.show_section_checkbox: QCheckBox
        self.mode_combobox: QComboBox

    def _setup_window(self):
        """Configure main window properties and styling."""
        self.setWindowTitle("SimuFrame - Structural Analysis")
        try:
            self.setWindowIcon(QIcon("SimuFrame/GUI/assets/icon.ico"))
        except FileNotFoundError:
            print("Warning: Icon 'SimuFrame/GUI/assets/icon.ico' not found.")

        self.resize(1600, 900)
        self.setMinimumSize(1200, 700)

        # Global style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fa;
            }
            QDockWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
                color: #1e293b;
                background-color: #f8fafc;
            }
            QDockWidget::title {
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-bottom: none;
                padding: 8px 10px;
                font-weight: bold;
                color: #1e293b;
                text-align: left;
            }
            QDockWidget::close-button, QDockWidget::float-button {
                background-color: transparent;
                border: none;
                padding: 2px;
            }
            QDockWidget::close-button:hover, QDockWidget::float-button:hover {
                background-color: #e2e8f0;
                border-radius: 3px;
            }
        """)

    def _create_central_widget(self):
        """Create the central PyVista plotter widget."""
        self.plotter = QtInteractor()
        self.dark_mode = False
        self.setCentralWidget(cast(QWidget, self.plotter))

    def _set_initial_state(self):
        """Set initial UI state on startup."""
        self.nav_tabs.widget(1).setEnabled(False)
        self.nav_tabs.setCurrentIndex(0)
        self.right_dock.setVisible(False)
        self.run_analysis_action.setEnabled(False)

    def _setup_docks(self):
        """Create all dock widgets."""
        self._create_left_dock()
        self._create_right_dock()
        self._create_bottom_dock()

    def _create_left_dock(self):
        """Create left navigator dock with data and results tabs."""
        self.left_dock = QDockWidget("Navigator", self)
        self.left_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        self.nav_tabs = QTabWidget()
        self.nav_tabs.setStyleSheet(self._get_tab_stylesheet())

        # Data tree tab
        self.data_tree = QTreeWidget()
        self.data_tree.setHeaderLabels(["Properties"])
        self.data_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.data_tree.setStyleSheet(self._get_tree_stylesheet())
        self.nav_tabs.addTab(self.data_tree, "Data")

        # Results tree tab
        self.results_tree = create_results_tree(self)
        self.nav_tabs.addTab(self.results_tree, "Results")

        self.left_dock.setWidget(self.nav_tabs)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.left_dock)

    def _create_right_dock(self):
        """Create right control panel dock."""
        self.right_dock = QDockWidget("Control Panel", self)
        self.right_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(12)

        self.scale_panel = create_scale_panel(self)
        self.options_panel = create_options_panel(self)

        control_layout.addWidget(self.scale_panel)
        control_layout.addWidget(self.options_panel)
        control_layout.addStretch()

        self.right_dock.setWidget(control_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.right_dock)

    def _create_bottom_dock(self):
        """Create bottom data table dock."""
        self.bottom_dock = QDockWidget("Data Table", self)
        self.bottom_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )

        self.table_tabs = QTabWidget()
        self.table_tabs.setStyleSheet(self._get_tab_stylesheet())

        # Create tables
        self.nodes_tab = _create_table_tab(
            ["ID", "X (m)", "Y (m)", "Z (m)", "Boundary Conditions"]
        )
        self.elements_tab = _create_table_tab(
            ["ID", "Initial Node", "Final Node", "Section", "Material", "Length (m)"]
        )
        self.sections_tab_widget = _create_sections_tab(self)

        self.table_tabs.addTab(self.nodes_tab, "Nodes")
        self.table_tabs.addTab(self.elements_tab, "Elements")
        self.table_tabs.addTab(self.sections_tab_widget, "Sections")

        self.bottom_dock.setWidget(self.table_tabs)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.bottom_dock)

    def _setup_menus(self):
        """Create menu bar with all menus."""
        self.main_menu_bar = self.menuBar()
        self.main_menu_bar.setStyleSheet(self._get_menubar_stylesheet())
        icon_color = '#475569'

        # Create menus
        self._create_file_menu(icon_color)
        self._create_analysis_menu(icon_color)
        self._create_results_menu()
        self._create_view_menu()
        self._create_help_menu(icon_color)

    def _create_file_menu(self, icon_color):
        """Create file menu with open, save, and export actions."""
        file_menu = self.main_menu_bar.addMenu("&File")

        open_action = QAction(qta.icon('fa5s.folder-open', color=icon_color), "Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.controller.load_project_from_yaml)

        new_action = QAction(qta.icon('fa5s.plus-square', color=icon_color), "New model...", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.open_structure_builder)

        save_action = QAction(qta.icon('fa5s.save', color=icon_color), "Save", self)
        save_action.setShortcut("Ctrl+S")

        export_action = QAction(qta.icon('fa5s.image', color=icon_color), "Export...", self)
        export_action.triggered.connect(self.take_screenshot)

        exit_action = QAction(qta.icon('fa5s.sign-out-alt', color=icon_color), "Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

        file_menu.addActions([open_action, save_action])
        file_menu.addSeparator()
        file_menu.addAction(new_action)
        file_menu.addSeparator()
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

    def _create_analysis_menu(self, icon_color):
        """Create analysis menu with run and parameter actions."""
        analysis_menu = self.main_menu_bar.addMenu("&Analysis")

        self.run_analysis_action = QAction(qta.icon('fa5s.play', color=icon_color), "Run Analysis", self)
        self.run_analysis_action.setShortcut("F5")
        self.run_analysis_action.triggered.connect(self.on_run_analysis_clicked)

        self.edit_params_action = QAction(qta.icon('fa5s.edit', color=icon_color), "Edit Parameters...", self)
        self.edit_params_action.setShortcut("Ctrl+E")
        self.edit_params_action.triggered.connect(self.open_analysis_parameters_dialog)

        analysis_menu.addActions([self.run_analysis_action, self.edit_params_action])

    def _create_results_menu(self):
        """Create results menu for post-processing."""
        self.results_menu = self.main_menu_bar.addMenu("&Results")

        history_action = QAction("Load history...", self)
        history_action.triggered.connect(self.open_load_displacement_dialog)

        convergence_overview = QAction("Convergence Overview...", self)
        convergence_overview.triggered.connect(self.open_convergence_dialog)

        self.results_menu.addActions([history_action, convergence_overview])
        self.results_menu.menuAction().setVisible(False)

    def _create_view_menu(self):
        """Create view menu for panel visibility."""
        self.view_menu = self.main_menu_bar.addMenu("View")
        self.view_menu.addSection("Panels")
        self.view_menu.addAction(self.left_dock.toggleViewAction())
        self.view_menu.addAction(self.right_dock.toggleViewAction())
        self.view_menu.addAction(self.bottom_dock.toggleViewAction())

    def _create_help_menu(self, icon_color):
        """Create help menu with about dialog."""
        help_menu = self.main_menu_bar.addMenu("&Help")

        about_action = QAction(qta.icon('fa5s.info-circle', color=icon_color), "About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _setup_toolbars(self):
        """Create application toolbars."""
        self.view_toolbar = setup_view_toolbar(self)
        self.addToolBar(self.view_toolbar)
        self.buckling_toolbar = None

    def _setup_statusbar(self):
        """Create status bar with processing label."""
        self.status_bar = QStatusBar(self)
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: white;
                border-top: 1px solid #e2e8f0;
            }
        """)
        self.setStatusBar(self.status_bar)

        self.processing_label = QLabel("Done.")
        self.processing_label.setStyleSheet("""
            QLabel {
                padding: 0 10px;
                color: #475569;
            }
        """)
        self.status_bar.addPermanentWidget(self.processing_label)

    def _connect_signals(self):
        """Connect all signal-slot pairs."""
        # Controller connections
        self.controller.status_message.connect(self.show_processing_message)
        self.controller.structure_loaded.connect(self.on_structure_loaded)
        self.controller.analysis_completed.connect(self.on_analysis_completed)

        # Results tree connections
        self.results_tree_manager.selection_changed.connect(self.on_results_selection_changed)

        # Visualization connections
        self.scale_spinbox.valueChanged.connect(self.on_visualization_option_changed)
        self.radio_diagram.toggled.connect(self.on_visualization_option_changed)
        self.show_section_checkbox.toggled.connect(self.on_section_visibility_changed)

    def on_structure_loaded(self):
        """Handle structure loaded event."""
        self.populate_data_ui()
        self.plot_undeformed_structure()
        self.bottom_dock.setVisible(True)
        self.run_analysis_action.setEnabled(True)

    def populate_data_ui(self):
        """Populate UI components with structure metadata."""
        if not self.state.has_structure():
            return

        self.show_processing_message("Loading model...")

        # Populate metadata tree
        DataTreePopulator.populate(self.data_tree, self.state.structure)

        # Populate tables
        yaml_data = self.state.structure.metadata
        nodes_table = self.nodes_tab.findChild(QTableWidget)
        elements_table = self.elements_tab.findChild(QTableWidget)

        if nodes_table:
            TablesPopulator.populate_nodes_table(nodes_table, self.state.structure)
        if elements_table:
            TablesPopulator.populate_elements_table(elements_table, yaml_data)

        sections_tree = self.sections_tab_widget.findChild(QTreeWidget)
        if sections_tree:
            TablesPopulator.populate_sections_tree(sections_tree, self.state.structure)

        self.show_processing_message("Done.")

    def open_structure_builder(self):
        """Open structure builder dialog (placeholder)."""
        # TODO: Implement functional builder
        pass

    def plot_undeformed_structure(self):
        """Plot the undeformed structure mesh."""
        if not self.state.has_structure():
            return

        try:
            # Load undeformed mesh
            mesh, sections, sections_idx = MeshLoader.load_undeformed_mesh(self.state.structure, self.state.config)

            if mesh:
                self.state.meshes.undeformed = mesh
                self.state.meshes.sections = sections
                self.state.meshes.sections_idx = sections_idx

                # Setup plotter
                self.plotter.clear_actors()
                self.plot_manager.setup_initial_view()

                # Plot structure
                plot_structure(
                    self.plotter,
                    self.state.structure,
                    mesh,
                    opacity=0.5,
                )

                self.plot_manager.reset_camera()

        except Exception as e:
            print(f"Error plotting undeformed structure: {e}")

    def on_run_analysis_clicked(self):
        """Handle run analysis button click."""
        if not self.state.has_structure():
            QMessageBox.warning(self, "Error", "No structure loaded.")
            return

        try:
            # Read analysis metadata
            analysis_data = self.state.structure.metadata.get('analysis', {})
            config_data = analysis_data.get('config', {})
            analysis_type = analysis_data.get('analysis_type', 'linear')

            # Normalize analysis type
            analysis_map = {
                'linear': 'linear',
                'nao-linear': 'nonlinear',
                'não-linear': 'nonlinear',
                'nonlinear': 'nonlinear',
                'nao linear': 'nonlinear',
                'não linear': 'nonlinear',
                'flambagem': 'buckling',
                'buckling': 'buckling',
                'instabilidade': 'buckling'
            }
            normalized_type = analysis_map.get(analysis_type.lower().strip(), 'linear')

            # Update structure
            self.state.config.analysis = normalized_type
            self.state.config.subdivisions = self.state.structure.subdivisions
            self.state.config.initial_steps = config_data.get('initial_steps', 15)
            self.state.config.max_iterations = config_data.get('max_iterations', 100)
            self.state.config.nonlinear_method = config_data.get('method', 'Newton-Raphson')
            self.state.config.max_load_factor = config_data.get('max_load_factor', 1.0)
            self.state.config.psi = config_data.get('psi', 1.0)
            self.state.config.num_modes = config_data.get('buckling_modes', 1)

            # Executar análise via controlador
            self.controller.handle_analysis_request()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load parameters: {e}")

    def open_analysis_parameters_dialog(self):
        """Open dialog to edit analysis parameters."""
        if not self.state.has_structure():
            QMessageBox.warning(self, "Error", "No structure loaded.")
            return

        # Get current parameters
        current_params = self.state.structure.metadata.get('analysis', {})

        # Create and run dialog
        dialog = AnalysisParametersDialog(current_params, self)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.show_processing_message("Updating parameters...")

            # Get new parameters
            new_params = dialog.get_parameters()

            # Update structure metadata
            success = self.controller.update_analysis_parameters(new_params)

            if success:
                self.controller.save_yaml_file()

    def on_analysis_completed(self):
        """Handle analysis completion event."""
        self.populate_results_ui()

    def populate_results_ui(self):
        """Populate UI with analysis results."""
        self.show_processing_message("Populating UI...")

        # Clear tree
        self.results_tree_manager.tree.clear()
        ResultsTreePopulator.populate(self.results_tree_manager.tree, self, self.state.structure.is_buckling)

        # Reconnect item changed signal
        self.results_tree_manager.tree.itemChanged.connect(
            self.results_tree_manager.handle_item_changed
        )

        # Enable UI
        self.nav_tabs.widget(1).setEnabled(True)
        self.nav_tabs.setCurrentIndex(1)
        self.right_dock.setVisible(True)
        self.results_menu.menuAction().setVisible(True)

        # Setup buckling toolbar
        self._setup_buckling_toolbar_if_needed()

        # Plot base structure
        self.plot_manager.plot_base_structure()
        self.plot_manager.reset_camera()

        self.show_processing_message("Analysis completed.")

    def _setup_buckling_toolbar_if_needed(self):
        """Setup buckling toolbar for buckling analysis."""
        # Remove toolbar if already exists
        if self.buckling_toolbar:
            self.removeToolBar(self.buckling_toolbar)
            self.buckling_toolbar.deleteLater()
            self.buckling_toolbar = None

        # Create new toolbar
        if self.state.is_buckling_analysis():
            self.buckling_toolbar = setup_buckling_toolbar(self, self.state.results)
            self.addToolBar(self.buckling_toolbar)

    def on_results_selection_changed(self, selected_keys):
        """Handle results tree selection changes."""
        if not self.state.has_results():
            return

        # Clear result actors
        self.plot_manager.clear_result_actors()

        # Separate reactions from other results
        reactions = [item['key'] for item in selected_keys if item['is_reaction']]
        others = [item['key'] for item in selected_keys if not item['is_reaction']]

        # Get current visualization options
        viz_options = self._get_current_visualization_options()

        if reactions:
            self.plot_manager.plot_result('support_reactions', reactions_to_plot=reactions)
        elif others:
            key = others[0]
            self.plot_manager.plot_result(key, **viz_options)

        self.plotter.render()

    def on_visualization_option_changed(self):
        """Handle visualization option changes."""
        self.state.viz_options.scale = self.scale_spinbox.value()
        self.state.viz_options.viz_style = (
            'Diagram' if self.radio_diagram.isChecked() else 'Colormap'
        )

        # Replot results
        selected = self.results_tree_manager.get_selected_keys()
        if selected:
            self.on_results_selection_changed(selected)

    def on_section_visibility_changed(self, visible):
        """Handle section visibility toggle."""
        self.state.viz_options.plot_section = visible
        self.plot_manager.update_section_visibility(visible)

    def _get_current_visualization_options(self):
        """Get current visualization options as dict."""
        return {
            'visualization': self.state.viz_options.viz_style
        }

    def on_mode_changed(self, index):
        """Handle buckling mode change."""
        if self.state.config.buckling_mode == index:
            return

        self.state.set_buckling_mode(index)
        self.plot_manager.plot_base_structure()

        # Replot results if mode changed
        selected = self.results_tree_manager.get_selected_keys()
        if selected:
            self.on_results_selection_changed(selected)

    def select_previous_mode(self):
        """Select previous buckling mode."""
        if self.state.config.num_modes == 0:
            return
        new_index = (self.state.config.buckling_mode - 1 +
                    self.state.config.num_modes) % self.state.config.num_modes
        self.mode_combobox.setCurrentIndex(new_index)

    def select_next_mode(self):
        """Select next buckling mode."""
        if self.state.config.num_modes == 0:
            return
        new_index = (self.state.config.buckling_mode + 1) % self.state.config.num_modes
        self.mode_combobox.setCurrentIndex(new_index)

    def show_processing_message(self, message):
        """Update status bar processing message."""
        self.processing_label.setText(message)
        QApplication.processEvents()

    def toggle_background(self):
        """Toggle plotter background color."""
        self.plot_manager.toggle_background()

    def take_screenshot(self):
        """Save current view as screenshot."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "", "PNG Image (*.png)"
        )
        if file_path:
            self.plot_manager.take_screenshot(file_path)
            self.show_processing_message(f"Screenshot saved to: {file_path}")

    def show_about(self):
        """Show about dialog with application info."""
        app_name = "SimuFrame"
        version = "v0.3.1"
        author = "Alysson Barbosa"
        license_type = "MIT License"
        year = "2025"
        repo_link = "https://github.com/SimuStruct/SimuFrame"

        text = f"""
        <h3>{app_name} {version}</h3>
        <p><b>Nonlinear Structural Analysis Solver</b></p>

        <p>A Python-based tool for advanced structural analysis using
        Total Lagrangian formulations with Green-Lagrange strains.</p>

        <p><b>Developed by:</b> {author}<br>
        <b>Copyright © {year}</b></p>

        <p>This software is open-source and released under the
        <b>{license_type}</b>.</p>

        <p>For source code, documentation, and issues, visit:<br>
        <a href='{repo_link}'>{repo_link}</a></p>
        """

        QMessageBox.about(self, f"About {app_name}", text)

    def open_convergence_dialog(self):
        """Open convergence overview dialog."""
        if self.state.results.convergence_data is None:
            QMessageBox.warning(self, "No Data", "Convergence data not available.")
            return
        dialog = ConvergenceDialog(self.state.results.convergence_data, parent=self)
        dialog.exec()

    def open_load_displacement_dialog(self):
        """Open load-displacement history dialog."""
        if not self.state.results.history:
            QMessageBox.warning(self, "No Data", "History data not available.")
            return
        dialog = LoadDisplacementDialog(
            history=self.state.results.history,
            structure=self.state.structure,
            analysis=self.state.config.analysis,
            parent=self
        )
        dialog.exec()

    def closeEvent(self, event):
        """Handle window close event."""
        if self.plotter:
            self.plotter.close()
        event.accept()
        super().closeEvent(event)

    @staticmethod
    def add_tree_parent(parent_widget, name):
        """Add a parent item to tree widget."""
        item = QTreeWidgetItem(parent_widget, [name])
        item.setExpanded(True)
        flags = item.flags()
        flags &= ~Qt.ItemFlag.ItemIsUserCheckable
        flags |= Qt.ItemFlag.ItemIsSelectable
        item.setFlags(flags)

        # Font style
        font = QFont()
        font.setBold(True)
        item.setFont(0, font)
        return item

    @staticmethod
    def add_tree_child(parent_item, name, data_key=None):
        """Add a child item to tree widget."""
        item = QTreeWidgetItem(parent_item, [name])

        if data_key:
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Unchecked)
            item.setData(0, Qt.ItemDataRole.UserRole, data_key)
        else:
            flags = item.flags()
            flags &= ~Qt.ItemFlag.ItemIsUserCheckable
            item.setFlags(flags)

        return item

    @staticmethod
    def _get_tab_stylesheet():
        """Get stylesheet for tab widgets."""
        return """
            QTabWidget::pane {
                border: 1px solid #e2e8f0;
                background-color: white;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #f8fafc;
                color: #475569;
                padding: 10px 20px;
                margin-right: 2px;
                border: 1px solid #e2e8f0;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 13px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #3b82f6;
                font-weight: bold;
                border-bottom: 2px solid white;
            }
            QTabBar::tab:hover {
                background-color: #eff6ff;
                color: #2563eb;
            }
        """

    @staticmethod
    def _get_tree_stylesheet():
        """Get stylesheet for tree widgets."""
        return """
            QTreeWidget {
                background-color: white;
                border: none;
                color: #1e293b;
                font-size: 11px;
            }
            QTreeWidget::item {
                padding: 6px;
                border-radius: 4px;
            }
            QTreeWidget::item:hover {
                background-color: #f1f5f9;
            }
            QTreeWidget::item:selected {
                background-color: #dbeafe;
                color: #1e293b;
            }
            QHeaderView::section {
                background-color: #f8fafc;
                color: #334155;
                padding: 8px;
                border: none;
                border-bottom: 2px solid #e2e8f0;
                font-weight: bold;
                font-size: 12px;
            }
        """

    @staticmethod
    def _get_menubar_stylesheet():
        """Get stylesheet for menu bar."""
        return """
            QMenuBar {
                background-color: white;
                color: #1e293b;
                border-bottom: 1px solid #e2e8f0;
                padding: 4px;
                font-size: 13px;
            }
            QMenuBar::item {
                background-color: transparent;
                color: #1e293b;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background-color: #f1f5f9;
                color: #3b82f6;
            }
            QMenu {
                background-color: white;
                color: #1e293b;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 30px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #eff6ff;
                color: #3b82f6;
            }
        """
