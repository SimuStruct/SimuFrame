# Third-party libraries
import qtawesome as qta
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QLabel, QGroupBox, QRadioButton,
    QTreeWidget, QDoubleSpinBox, QFrame, QCheckBox, QTableWidget, QHeaderView, QToolBar
)
from PySide6.QtGui import QIcon, QAction, QPixmap, QPainter, QFont
from PySide6.QtCore import QSize, Qt
from PySide6.QtSvg import QSvgRenderer


def _create_table_tab(headers):
    """Create a tab with a table."""
    tab = QWidget()
    tab_layout = QVBoxLayout(tab)
    tab_layout.setContentsMargins(5, 5, 5, 5)  # Menos margem

    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.setAlternatingRowColors(True)
    table.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked)
    
    table.setStyleSheet("""
        QTableWidget {
            background-color: white; border: none;
            gridline-color: #e2e8f0;
        }
        QTableWidget::item { padding: 6px; color: #1e293b; }
        QTableWidget::item:selected { background-color: #dbeafe; color: #1e293b; }
        QHeaderView::section {
            background-color: #f8fafc; color: #334155;
            padding: 8px; border: none;
            border-bottom: 2px solid #e2e8f0; font-weight: bold;
        }
    """)
    tab_layout.addWidget(table)
    return tab

def _create_sections_tab(main_window):
    """Create a tab with a tree widget to display section properties."""
    tab = QWidget()
    layout = QVBoxLayout(tab)
    layout.setContentsMargins(5, 5, 5, 5)

    main_window.sections_tree_table = QTreeWidget()
    main_window.sections_tree_table.setHeaderLabels(["Description", "Symbol", "Value", "Unit"])
    main_window.sections_tree_table.setAlternatingRowColors(True)
    main_window.sections_tree_table.setColumnCount(4)
    main_window.sections_tree_table.setStyleSheet("""
        QTreeWidget {
            border: none; background-color: white;
            font-size: 13px; color: #1e293b;
        }
        QTreeWidget::item { padding: 6px; border-radius: 3px; }
        QTreeWidget::item:hover { background-color: #f1f5f9; }
        QHeaderView::section {
            background-color: #f8fafc; color: #475569;
            padding: 8px; font-weight: bold;
            border: none; border-bottom: 1px solid #e2e8f0;
        }
    """)
    main_window.sections_tree_table.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    main_window.sections_tree_table.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
    main_window.sections_tree_table.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
    main_window.sections_tree_table.header().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

    layout.addWidget(main_window.sections_tree_table)
    return tab

def create_scale_panel(main_window):
    group_box = QGroupBox("Visualization Scale")
    group_box.setStyleSheet("""
        QGroupBox {
            font-size: 13px; font-weight: bold; color: #1e293b;
            border: 1px solid #e2e8f0; border-radius: 8px;
            margin-top: 12px; padding-top: 12px; background-color: white;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 10px;
            padding: 0 5px; color: #334155;
        }
    """)

    layout = QVBoxLayout(group_box)
    layout.setContentsMargins(12, 20, 12, 12)

    main_window.scale_spinbox = QDoubleSpinBox()
    main_window.scale_spinbox.setRange(0.01, 500.0)
    main_window.scale_spinbox.setValue(1.00)
    main_window.scale_spinbox.setSingleStep(0.1)
    main_window.scale_spinbox.setDecimals(2)
    main_window.scale_spinbox.setSuffix(" x")
    main_window.scale_spinbox.setFont(QFont("Segoe UI", 10))
    main_window.scale_spinbox.setStyleSheet("""
        QDoubleSpinBox {
            background-color: #f8fafc; border: 1px solid #e2e8f0;
            border-radius: 6px; padding: 8px; color: #1e293b; font-weight: bold;
        }
        QDoubleSpinBox:focus { border-color: #3b82f6; background-color: white; }
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
            subcontrol-origin: border; width: 22px;
            border-left: 1px solid #e2e8f0; background-color: transparent;
        }
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background-color: #f1f5f9; }
        QDoubleSpinBox::up-button { subcontrol-position: top right; border-top-right-radius: 6px; }
        QDoubleSpinBox::down-button { subcontrol-position: bottom right; border-bottom-right-radius: 6px; }
        QDoubleSpinBox::up-arrow {
            border-left: 5px solid transparent; border-right: 5px solid transparent;
            border-bottom: 5px solid #64748b; width: 0px; height: 0px;
        }
        QDoubleSpinBox::up-arrow:hover { border-bottom-color: #1e293b; }
        QDoubleSpinBox::down-arrow {
            border-left: 5px solid transparent; border-right: 5px solid transparent;
            border-top: 5px solid #64748b; width: 0px; height: 0px;
        }
        QDoubleSpinBox::down-arrow:hover { border-top-color: #1e293b; }
    """)
    layout.addWidget(main_window.scale_spinbox)
    return group_box

def create_results_tree(main_window):
    tree = QTreeWidget()
    tree.setHeaderLabel("Results")
    tree.setStyleSheet("""
        QTreeWidget {
            font-size: 13px; color: #1e293b;
            border: none; background-color: white;
        }
        QHeaderView::section {
            background-color: #f8fafc; color: #334155;
            padding: 8px; border: none;
            border-bottom: 2px solid #e2e8f0; font-weight: bold; font-size: 13px;
        }
        QTreeWidget::item { padding: 7px; border-radius: 4px; color: #1e293b; }
        QTreeWidget::item:hover { background-color: #f1f5f9; }
        QTreeWidget::item:selected { background-color: #dbeafe; color: #1e293b; }
        QTreeWidget::indicator { width: 18px; height: 18px; border-radius: 4px; }
        QTreeWidget::indicator:unchecked { border: 2px solid #cbd5e1; background-color: white; }
        QTreeWidget::indicator:unchecked:hover { border-color: #3b82f6; }
        QTreeWidget::indicator:checked { background-color: #fbbf24; border: 2px solid #fbbf24; }
        QTreeWidget::indicator:checked:hover { background-color: #f59e0b; border: 2px solid #f59e0b; }
    """)

    # Populate tree with global displacements options
    parent_deform = main_window.add_tree_parent(tree, "Global Displacements")
    main_window.add_tree_child(parent_deform, "|u|", "u")
    main_window.add_tree_child(parent_deform, "ux", "ux")
    main_window.add_tree_child(parent_deform, "uy", "uy")
    main_window.add_tree_child(parent_deform, "uz", "uz")
    main_window.add_tree_child(parent_deform, "θx", "θx")
    main_window.add_tree_child(parent_deform, "θy", "θy")
    main_window.add_tree_child(parent_deform, "θz", "θz")

    parent_members = main_window.add_tree_parent(tree, "Membros")
    child_forces = main_window.add_tree_parent(parent_members, "Esforços Internos")
    main_window.add_tree_child(child_forces, "N - Axial", "fx")
    main_window.add_tree_child(child_forces, "Vy - Cortante Y", "fy")
    main_window.add_tree_child(child_forces, "Vz - Cortante Z", "fz")
    main_window.add_tree_child(child_forces, "Mt - Torçor", "mx")
    main_window.add_tree_child(child_forces, "My - Fletor Y", "my")
    main_window.add_tree_child(child_forces, "Mz - Fletor Z", "mz")

    parent_reactions = main_window.add_tree_parent(tree, "Reações de Apoio")
    main_window.add_tree_child(parent_reactions, "Px", "rx")
    main_window.add_tree_child(parent_reactions, "Py", "ry")
    main_window.add_tree_child(parent_reactions, "Pz", "rz")
    main_window.add_tree_child(parent_reactions, "Mx", "rmx")
    main_window.add_tree_child(parent_reactions, "My", "rmy")
    main_window.add_tree_child(parent_reactions, "Mz", "rmz")

    return tree
    

def create_options_panel(main_window):
    group_box = QGroupBox("Opções de Visualização")
    group_box.setStyleSheet("""
        QGroupBox {
            font-size: 13px; font-weight: bold; color: #1e293b;
            border: 1px solid #e2e8f0; border-radius: 8px;
            margin-top: 12px; padding-top: 12px; background-color: white;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 10px;
            padding: 0 5px; color: #334155;
        }
    """)

    layout = QVBoxLayout(group_box)
    layout.setContentsMargins(12, 20, 12, 12)
    layout.setSpacing(8)

    main_window.radio_diagram = QRadioButton("Com Diagrama")
    main_window.radio_colormap = QRadioButton("Apenas Colormap")
    main_window.radio_colormap.setChecked(True)
    radio_button_style = """
        QRadioButton {
            font-size: 13px; color: #475569; spacing: 8px;
            padding: 5px; background-color: transparent;
        }
        QRadioButton::indicator {
            width: 18px; height: 18px; border-radius: 9px;
            border: 2px solid #cbd5e1; background-color: white;
        }
        QRadioButton::indicator:hover { border-color: #3b82f6; }
        QRadioButton::indicator:checked { background-color: #3b82f6; border-color: #3b82f6; }
    """
    main_window.radio_diagram.setStyleSheet(radio_button_style)
    main_window.radio_colormap.setStyleSheet(radio_button_style)

    layout.addWidget(main_window.radio_diagram)
    layout.addWidget(main_window.radio_colormap)

    separator = QFrame()
    separator.setFrameShape(QFrame.Shape.HLine)
    separator.setStyleSheet("border-top: 1px solid #e2e8f0; margin: 5px 0;")
    layout.addWidget(separator)

    main_window.show_section_checkbox = QCheckBox("Exibir seção transversal")
    main_window.show_section_checkbox.setChecked(True)
    checkbox_style = """
        QCheckBox {
            font-size: 13px; color: #475569; spacing: 8px;
            padding: 5px; background-color: transparent;
        }
        QCheckBox::indicator {
            width: 18px; height: 18px; border-radius: 4px;
        }
        QCheckBox::indicator:unchecked { border: 2px solid #cbd5e1; background-color: white; }
        QCheckBox::indicator:unchecked:hover { border-color: #3b82f6; }
        QCheckBox::indicator:checked { background-color: #3b82f6; border: 2px solid #3b82f6; }
    """
    main_window.show_section_checkbox.setStyleSheet(checkbox_style)
    layout.addWidget(main_window.show_section_checkbox)
    
    return group_box

def create_view_icon(h_axis: str, v_axis: str, view_label: str) -> QIcon:
    """Cria dinamicamente um ícone vetorial (SVG) para vista ortogonal."""
    colors = {'X': "#F57033", 'Y': '#24A148', 'Z': '#0F62FE'}
    h_color = colors.get(h_axis, 'black')
    v_color = colors.get(v_axis, 'black')
    svg_template = f"""
    <svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrow_h_{h_color.replace('#', '')}" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{h_color}" /></marker>
        <marker id="arrow_v_{v_color.replace('#', '')}" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{v_color}" /></marker>
      </defs>
      <circle cx="12" cy="12" r="6" fill="#FDD13A" stroke="#333333" stroke-width="2"/>
      <line x1="12" y1="12" x2="12" y2="52" stroke="{v_color}" stroke-width="7" marker-end="url(#arrow_v_{v_color.replace('#', '')})" />
      <line x1="12" y1="12" x2="52" y2="12" stroke="{h_color}" stroke-width="7" marker-end="url(#arrow_h_{h_color.replace('#', '')})" />
      <text x="52" y="52" font-family="Arial, sans-serif" font-size="24" font-weight="bold" fill="#1e293b" text-anchor="middle" dominant-baseline="middle">{view_label}</text>
    </svg>
    """
    svg_bytes = svg_template.encode('utf-8')
    renderer = QSvgRenderer(svg_bytes)
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)

def setup_view_toolbar(main_window):
    """Cria uma barra de ferramentas com botões para controle de vistas."""
    toolbar = QToolBar("Barra de Vistas")
    toolbar.setIconSize(QSize(22, 22)) # Ligeiramente menor
    toolbar.setStyleSheet("""
        QToolBar {
            background-color: white;
            border-bottom: 1px solid #e2e8f0;
            padding: 4px;
            spacing: 4px;
        }
        QToolBar QToolButton {
            background-color: transparent;
            border-radius: 4px;
            padding: 6px;
        }
        QToolBar QToolButton:hover {
            background-color: #f1f5f9;
            color: #1e293b;
        }
        QToolBar QToolButton:checked {
            background-color: #e0f2fe;
        }
        QToolBar::separator {
            width: 1px;
            background-color: #e2e8f0;
            margin: 4px 6px;
        }
    """)
    color = '#475569'

    action_reset_cam = QAction(qta.icon('fa5s.expand-arrows-alt', color=color), "Resetar Câmera", main_window)
    action_reset_cam.triggered.connect(lambda: main_window.plotter.reset_camera())
    toolbar.addAction(action_reset_cam)
    toolbar.addSeparator()

    action_iso = QAction(qta.icon('fa5s.cube', color=color), "Vista Isométrica", main_window)
    action_iso.triggered.connect(lambda: main_window.plotter.isometric_view())
    toolbar.addAction(action_iso)

    icon_xy = create_view_icon(h_axis='X', v_axis='Y', view_label='Z')
    action_xy = QAction(icon_xy, "Vista Superior (de Z)", main_window)
    action_xy.triggered.connect(lambda: main_window.plotter.view_xy())
    toolbar.addAction(action_xy)

    icon_yz = create_view_icon(h_axis='Y', v_axis='Z', view_label='X')
    action_yz = QAction(icon_yz, "Vista Frontal (de X)", main_window)
    action_yz.triggered.connect(lambda: main_window.plotter.view_yz())
    toolbar.addAction(action_yz)

    icon_xz = create_view_icon(h_axis='X', v_axis='Z', view_label='Y')
    action_xz = QAction(icon_xz, "Vista Lateral (de Y)", main_window)
    action_xz.triggered.connect(lambda: main_window.plotter.view_xz())
    toolbar.addAction(action_xz)
    toolbar.addSeparator()

    action_bg = QAction(qta.icon('fa5s.adjust', color=color), "Alternar Cor de Fundo", main_window)
    action_bg.triggered.connect(main_window.toggle_background)
    toolbar.addAction(action_bg)

    action_screenshot = QAction(qta.icon('fa5s.camera', color=color), "Capturar Tela", main_window)
    action_screenshot.triggered.connect(main_window.take_screenshot)
    toolbar.addAction(action_screenshot)
    return toolbar

def setup_buckling_toolbar(main_window, results):
    """Cria a barra de ferramentas para seleção de modos de flambagem."""
    buckling_toolbar = QToolBar("Modos de Flambagem")
    buckling_toolbar.setMovable(False)
    buckling_toolbar.setObjectName("bucklingToolbar")
    buckling_toolbar.setStyleSheet("""
        #bucklingToolbar {
            background-color: white;
            border-bottom: 1px solid #e2e8f0;
            padding: 4px;
            spacing: 8px;
        }
        #bucklingToolbar QToolButton {
            background-color: transparent;
            border-radius: 4px;
            padding: 6px;
        }
        #bucklingToolbar QToolButton:hover {
            background-color: #f1f5f9;
        }
        #bucklingToolbar QLabel {
            font-weight: bold;
            color: #1e293b;
            font-size: 13px;
            padding-left: 5px;
        }
    """)

    header_label = QLabel("Modo de Flambagem:")

    main_window.mode_combobox = QComboBox()
    main_window.mode_combobox.setFont(QFont("Consolas, Courier New", 10))
    main_window.mode_combobox.setMinimumWidth(250)
    main_window.mode_combobox.setStyleSheet("""
        QComboBox {
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 6px 10px;
            background-color: white;
            color: #1e293b;
        }
        QComboBox:hover {
            border-color: #3b82f6;
        }
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        QComboBox::down-arrow {
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 4px solid #475569;
            width: 0px;
            height: 0px;
            margin-right: 5px;
        }
        QComboBox QAbstractItemView {
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            background-color: white;
            outline: 0px;
            color: #1e293b;
            selection-background-color: #3b82f6;
            selection-color: white;
        }
    """)

    icon_color = '#475569'
    main_window.prev_mode_action = QAction(qta.icon('fa5s.chevron-left', color=icon_color), "Modo Anterior", main_window)
    main_window.prev_mode_action.triggered.connect(main_window.select_previous_mode)
    main_window.next_mode_action = QAction(qta.icon('fa5s.chevron-right', color=icon_color), "Próximo Modo", main_window)
    main_window.next_mode_action.triggered.connect(main_window.select_next_mode)

    buckling_toolbar.addWidget(header_label)
    buckling_toolbar.addAction(main_window.prev_mode_action)
    buckling_toolbar.addWidget(main_window.mode_combobox)
    buckling_toolbar.addAction(main_window.next_mode_action)

    if results.autovalores is not None:
        for i, eigenvalue in enumerate(results.autovalores):
            item_text = f"Modo {i + 1:<3} | λ = {eigenvalue:>9.3f}"
            main_window.mode_combobox.addItem(item_text)

    main_window.mode_combobox.currentIndexChanged.connect(main_window.on_mode_changed)
    return buckling_toolbar