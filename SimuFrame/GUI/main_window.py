# Third-party libraries
import qtawesome as qta
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QAction, QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QFileDialog, QLabel, QMessageBox,
    QTreeWidget, QApplication, QStatusBar, QDockWidget, QTableWidget,
    QTabWidget, QHeaderView, QDialog, QTreeWidgetItem
)
from pyvistaqt import QtInteractor

from SimuFrame.GUI.controller import SimuController

# Local libraries
from SimuFrame.GUI.dialogs import (
    ConvergenceDialog, LoadDisplacementDialog, AnalysisParametersDialog
)
from SimuFrame.GUI.plot_manager import PlotManager
from SimuFrame.GUI.results_tree_manager import ResultsTreeManager
from SimuFrame.GUI.state_manager import StateManager
from SimuFrame.GUI.ui_populators import (
    DataTreePopulator, TablesPopulator, ResultsTreePopulator, MeshLoader
)
from SimuFrame.GUI.widgets import (
    _create_table_tab, _create_sections_tab, create_scale_panel,
    create_results_tree, create_options_panel, setup_view_toolbar,
    setup_buckling_toolbar
)
from SimuFrame.post_processing.visualization import plot_structure


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

    def _setup_window(self):
        """Setup window properties."""
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
        """Create the central widget (PyVista QtInteractor)."""
        self.plotter = QtInteractor()
        self.dark_mode = False
        self.setCentralWidget(self.plotter)

    def _set_initial_state(self):
        """Set the initial state of the UI."""
        self.nav_tabs.widget(1).setEnabled(False)
        self.nav_tabs.setCurrentIndex(0)
        self.right_dock.setVisible(False)
        self.run_analysis_action.setEnabled(False)

    def _setup_docks(self):
        """Create docks."""
        self._create_left_dock()
        self._create_right_dock()
        self._create_bottom_dock()
    
    def _create_left_dock(self):
        """Create the left dock (navigator)."""
        self.left_dock = QDockWidget("Navigator", self)
        # noinspection PyTypeChecker
        self.left_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        
        self.nav_tabs = QTabWidget()
        self.nav_tabs.setStyleSheet(self._get_tab_stylesheet())
        
        # Tab 1: Data Tree
        self.data_tree = QTreeWidget()
        self.data_tree.setHeaderLabels(["Properties"])
        self.data_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.data_tree.setStyleSheet(self._get_tree_stylesheet())
        self.nav_tabs.addTab(self.data_tree, "Data")
        
        # Tab 2: Results Tree
        self.results_tree = create_results_tree(self)
        self.nav_tabs.addTab(self.results_tree, "Results")
        
        self.left_dock.setWidget(self.nav_tabs)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.left_dock)
    
    def _create_right_dock(self):
        """Create the right dock (control panel)."""
        self.right_dock = QDockWidget("Control Panel", self)
        # noinspection PyTypeChecker
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
        """Create the bottom dock (tables)."""
        self.bottom_dock = QDockWidget("Data Table", self)
        # noinspection PyTypeChecker
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
        """Create the menu bar."""
        self.main_menu_bar = self.menuBar()
        self.main_menu_bar.setStyleSheet(self._get_menubar_stylesheet())

        icon_color = '#475569'

        # File Menu
        self._create_file_menu(icon_color)

        # Analysis Menu
        self._create_analysis_menu(icon_color)

        # Results Menu
        self._create_results_menu()

        # View Menu
        self._create_view_menu()

        # Help Menu
        self._create_help_menu(icon_color)

    def _create_file_menu(self, icon_color):
        """Create the file menu."""
        arquivo_menu = self.main_menu_bar.addMenu("&File")

        open_action = QAction(qta.icon('fa5s.folder-open', color=icon_color), "Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.controller.load_project_from_yaml)

        new_action = QAction(qta.icon('fa5s.plus-square', color=icon_color), "New model_type...", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.open_structure_builder)

        save_action = QAction(qta.icon('fa5s.save', color=icon_color), "Save", self)
        save_action.setShortcut("Ctrl+S")

        export_graph_action = QAction(qta.icon('fa5s.image', color=icon_color), "Export...", self)
        export_graph_action.triggered.connect(self.take_screenshot)

        sair_action = QAction(qta.icon('fa5s.sign-out-alt', color=icon_color), "Exit", self)
        sair_action.setShortcut("Ctrl+Q")
        sair_action.triggered.connect(self.close)

        arquivo_menu.addActions([open_action, save_action])
        arquivo_menu.addSeparator()
        arquivo_menu.addAction(new_action)
        arquivo_menu.addSeparator()
        arquivo_menu.addAction(export_graph_action)
        arquivo_menu.addSeparator()
        arquivo_menu.addAction(sair_action)

    def _create_analysis_menu(self, icon_color):
        """Create the analysis menu."""
        analise_menu = self.main_menu_bar.addMenu("&Analysis")

        self.run_analysis_action = QAction(qta.icon('fa5s.play', color=icon_color), "Run Analysis", self)
        self.run_analysis_action.setShortcut("F5")
        self.run_analysis_action.triggered.connect(self.on_run_analysis_clicked)

        self.edit_params_action = QAction(qta.icon('fa5s.edit', color=icon_color), "Edit Parameters...", self)
        self.edit_params_action.setShortcut("Ctrl+E")
        self.edit_params_action.triggered.connect(self.open_analysis_parameters_dialog)

        analise_menu.addActions([self.run_analysis_action, self.edit_params_action])

    def _create_results_menu(self):
        """Create the results menu."""
        self.results_menu = self.main_menu_bar.addMenu("&Results")

        fxd_action = QAction("Load x displacement...", self)
        fxd_action.triggered.connect(self.abrir_dialogo_grafico_fxd)

        # TODO: Tentar resolver esse problema (o que é isso?)
        graficos_calculo_action = QAction("Gráficos de Cálculo...", self)
        graficos_calculo_action.triggered.connect(self.abrir_dialogo_graficos_calculo)

        self.results_menu.addActions([fxd_action, graficos_calculo_action])
        self.results_menu.menuAction().setVisible(False)

    def _create_view_menu(self):
        """Create the view menu."""
        self.view_menu = self.main_menu_bar.addMenu("View")
        self.view_menu.addSection("Panels")
        self.view_menu.addAction(self.left_dock.toggleViewAction())
        self.view_menu.addAction(self.right_dock.toggleViewAction())
        self.view_menu.addAction(self.bottom_dock.toggleViewAction())

    def _create_help_menu(self, icon_color):
        """Create the help menu."""
        ajuda_menu = self.main_menu_bar.addMenu("&Help")
        sobre_action = QAction(qta.icon('fa5s.info-circle', color=icon_color), "About", self)
        sobre_action.triggered.connect(self.show_about)
        ajuda_menu.addAction(sobre_action)

    def _setup_toolbars(self):
        """Create the toolbars."""
        self.view_toolbar = setup_view_toolbar(self)
        self.addToolBar(self.view_toolbar)
        self.buckling_toolbar = None

    def _setup_statusbar(self):
        """Create the status bar."""
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
        """Connect signals."""
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
        """Signal handler for when a structure is loaded."""
        # Populate UI
        self.populate_data_ui()
        
        # Plot undeformed structure
        self.plot_undeformed_structure()
        
        # Enable UI
        self.bottom_dock.setVisible(True)
        self.run_analysis_action.setEnabled(True)

    def populate_data_ui(self):
        """Populate the UI with metadata."""
        if not self.state.has_structure():
            return
        
        self.show_processing_message("Loading model_type...")
        
        # Populate metadata tree
        DataTreePopulator.populate(self.data_tree, self.state.estrutura)
        
        # Populate tables
        yaml_data = self.state.estrutura.metadata
        nodes_table = self.nodes_tab.findChild(QTableWidget)
        elements_table = self.elements_tab.findChild(QTableWidget)
        # sections_tree = self.sections_tab_widget.findChild(QTableWidget.__bases__[0].__bases__[0])
        
        if nodes_table:
            TablesPopulator.populate_nodes_table(nodes_table, self.state.estrutura)
        if elements_table:
            TablesPopulator.populate_elements_table(elements_table, yaml_data)
        
        sections_tree = self.sections_tab_widget.findChild(QTreeWidget)
        if sections_tree:
            TablesPopulator.populate_sections_tree(sections_tree, self.state.estrutura)
        
        self.show_processing_message("Done.")

    # TODO: Create a functional builder.
    def open_structure_builder(self):
        """Open builder."""
        self.controller.open_structure_creator()

    def load_created_structure(self, file_path):
        """Load created structure."""
        try:
            self.controller.load_project_from_yaml()
            self.show_processing_message(f"Loaded model_type: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"{str(e)}")

    def plot_undeformed_structure(self):
        """Plot undeformed structure."""
        if not self.state.has_structure():
            return
        
        try:
            # Load undeformed mesh
            malha, secoes, secoes_indices = MeshLoader.load_undeformed_mesh(self.state.estrutura, self.state.config)

            if malha:
                self.state.malhas.indeformada = malha
                self.state.malhas.secoes = secoes
                self.state.malhas.secoes_indices = secoes_indices
                
                # Setup plotter
                self.plotter.clear_actors()
                self.plot_manager.setup_initial_view()

                # Plot structure
                plot_structure(
                    self.plotter,
                    self.state.estrutura,
                    malha,
                    transparencia=1.0,
                )
                
                self.plot_manager.reset_camera()
        
        except Exception as e:
            print(f"Error plotting undeformed structure: {e}")

    def on_run_analysis_clicked(self):
        """Run analysis."""
        if not self.state.has_structure():
            QMessageBox.warning(self, "Error", "No structure loaded.")
            return
        
        try:
            if self.state.estrutura is not None:
                # Read analysis metadata
                analysis_data = self.state.estrutura.metadata.get('analysis', {})
                analysis_type = analysis_data.get('analysis_type', 'linear')
                subdivisions = analysis_data.get('number_of_subdivisions', 1)

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
                analysis_type_clean = analysis_type.lower().strip()
                analysis_normalized = analysis_map.get(analysis_type_clean, 'linear')

                # Update structure
                self.state.config.tipo = analysis_normalized
                self.state.config.subdivisions = subdivisions

                # Executar análise via controlador
                self.controller.handle_analysis_request()
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load parameters: {e}")
    
    def open_analysis_parameters_dialog(self):
        """Open analysis parameters dialog."""
        if not self.state.has_structure():
            QMessageBox.warning(self, "Error", "No structure loaded.")
            return
        
        # Get current parameters
        current_params = self.state.estrutura.metadata.get('analysis', {})
        
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
        """UI handler for when analysis is completed."""
        self.populate_results_ui()
    
    def populate_results_ui(self):
        """Populate UI with analysis results."""
        self.show_processing_message("Populating UI...")

        # Clear tree
        self.results_tree_manager.tree.clear()
        ResultsTreePopulator.populate(self.results_tree_manager.tree, self)
        
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
        """Setup buckling toolbar."""
        # Remove toolbar if already exists
        if self.buckling_toolbar:
            self.removeToolBar(self.buckling_toolbar)
            self.buckling_toolbar.deleteLater()
            self.buckling_toolbar = None
        
        # Create new toolbar
        if self.state.is_buckling_analysis():
            self.buckling_toolbar = setup_buckling_toolbar(self, self.state.resultados)
            self.addToolBar(self.buckling_toolbar)

    def on_results_selection_changed(self, selected_keys):
        """Slot for results selection."""
        if not self.state.has_results():
            return
        
        # Clear result actors
        self.plot_manager.clear_result_actors()
        
        # Separate reactions from other results
        reacoes = [item['key'] for item in selected_keys if item['is_reaction']]
        outros = [item['key'] for item in selected_keys if not item['is_reaction']]
        
        # Get current visualization options
        viz_options = self._get_current_visualization_options()
        
        # Plot reactions
        if reacoes:
            self.plot_manager.plot_result('reacoes_apoio', reacoes_a_plotar=reacoes)
        
        # Plot other results
        elif outros:
            key = outros[0]
            self.plot_manager.plot_result(key, **viz_options)
        
        # Render
        self.plotter.render()
    
    def on_visualization_option_changed(self):
        """Slot for visualization option change."""
        # Update state
        self.state.viz_options.escala = self.scale_spinbox.value()
        self.state.viz_options.visualizacao_tipo = (
            'Diagrama' if self.radio_diagram.isChecked() else 'Colormap'
        )
        
        # Replot results
        selected = self.results_tree_manager.get_selected_keys()
        if selected:
            self.on_results_selection_changed(selected)
    
    def on_section_visibility_changed(self, visible):
        """Slot for section visibility."""
        self.state.viz_options.plotar_secao = visible
        self.plot_manager.update_section_visibility(visible)
    
    def _get_current_visualization_options(self):
        """Returns current visualization options."""
        return {
            'visualizacao': self.state.viz_options.visualizacao_tipo
        }

    def on_mode_changed(self, index):
        """Slot for buckling mode change."""
        if self.state.config.modo_flambagem_atual == index:
            return
        
        self.state.set_buckling_mode(index)
        self.plot_manager.plot_base_structure()
        
        # Replot results if mode changed
        selected = self.results_tree_manager.get_selected_keys()
        if selected:
            self.on_results_selection_changed(selected)
    
    def select_previous_mode(self):
        """Slot to select previous mode."""
        if self.state.config.num_modos == 0:
            return
        new_index = (self.state.config.modo_flambagem_atual - 1 + 
                    self.state.config.num_modos) % self.state.config.num_modos
        self.mode_combobox.setCurrentIndex(new_index)
    
    def select_next_mode(self):
        """Slot to select next mode."""
        if self.state.config.num_modos == 0:
            return
        new_index = (self.state.config.modo_flambagem_atual + 1) % self.state.config.num_modos
        self.mode_combobox.setCurrentIndex(new_index)

    def show_processing_message(self, message):
        """Update processing label."""
        self.processing_label.setText(message)
        QApplication.processEvents()
    
    def toggle_background(self):
        """Change background color."""
        self.plot_manager.toggle_background()
    
    def take_screenshot(self):
        """Save screenshot."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "", "PNG Image (*.png)"
        )
        if file_path:
            self.plot_manager.take_screenshot(file_path)
            self.show_processing_message(f"Screenshot saved to: {file_path}")
    
    def show_about(self):
        """Show about dialog."""
        app_name = "SimuFrame"
        version = "v0.4.0"
        author = "Alysson Barbosa"
        license_type = "MIT License"
        year = "2025"
        
        # GitHub repository link
        repo_link = "https://github.com/SimuStruct/SimuFrame"
        
        # HTML text
        text = f"""
        <h3>{app_name} {version}</h3>
        <p><b>Nonlinear Structural Analysis Solver</b></p>
        
        <p>A Python-based tool for advanced structural analysis using 
        Total Lagrangian formulations with Arc-Length control.</p>
        
        <p><b>Developed by:</b> {author}<br>
        <b>Copyright © {year}</b></p>
        
        <p>This software is open-source and released under the 
        <b>{license_type}</b>.</p>
        
        <p>For source code, documentation, and issues, visit:<br>
        <a href='{repo_link}'>{repo_link}</a></p>
        """

        QMessageBox.about(self, f"About {app_name}", text)
    
    def abrir_dialogo_graficos_calculo(self):
        """Open dialog for the equilibrium path graph."""
        if self.state.resultados.convergence_data is None:
            QMessageBox.warning(self, "No Data", "Equilibrium path metadata not available.")
            return
        dialog = ConvergenceDialog(self.state.resultados.convergence_data, parent=self)
        dialog.exec()
    
    def abrir_dialogo_grafico_fxd(self):
        """Open dialog for the force-displacement graph."""
        if not self.state.resultados.f_vs_d:
            QMessageBox.warning(self, "No Data", "Data not available.")
            return
        dialog = LoadDisplacementDialog(
            f_vs_d_history=self.state.resultados.f_vs_d,
            estrutura=self.state.estrutura,
            analise=self.state.config.tipo,
            parent=self
        )
        dialog.exec()
    
    def closeEvent(self, event):
        """Setup close event."""
        if self.plotter:
            self.plotter.close()
        event.accept()

    @staticmethod
    def add_tree_parent(parent_widget, name):
        """Add a parent item to the tree."""
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
        """Add a child item to the tree."""
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
        """Setup stylesheet for QTabWidget."""
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
        """Setup stylesheet for QTreeWidget."""
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
        """Setup stylesheet for QMenuBar."""
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
