"""
Criador Interativo de Estruturas - SimuFrame
Interface moderna para criar e visualizar estruturas em tempo real
"""

import sys
import numpy as np
from ruamel.yaml import YAML
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton,
    QLabel, QLineEdit, QComboBox, QGroupBox, QFormLayout,
    QTabWidget, QDoubleSpinBox, QSpinBox, QMessageBox, QFileDialog,
    QCheckBox
)
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QIcon
import qtawesome as qta
from pyvistaqt import QtInteractor


class StructureCreatorDialog(QDialog):
    """Interface para criação interativa de estruturas."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Criador de Estruturas - SimuFrame")
        self.resize(1600, 900)

        # Dados do projeto
        self.project_data = {
            'info': {
                'project_name': 'Novo Projeto',
                'description': '',
                'project_version': '1.0',
                'author': '',
                'created_at': ''
            },
            'analysis': {
                'analysis_type': 'linear',
                'element_type': 'beam',
                'number_of_subdivisions': 4
            },
            'materials': {},
            'section_data': {},
            'nodes': [],
            'elements': [],
            'supports': [],
            'nodal_loads': [],
            'distributed_loads': []
        }

        # Timer para atualização suave do plot
        self.plot_timer = QTimer()
        self.plot_timer.setSingleShot(True)
        self.plot_timer.timeout.connect(self._update_plot_now)

        # Configurar YAML
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.default_flow_style = None
        self.yaml.indent(mapping=2, sequence=2, offset=0)
        self.yaml.width = 4096

        self._setup_ui()
        self._init_defaults()
        self._update_plot()

    def _init_defaults(self):
        """Adiciona dados padrão para facilitar o início."""
        # Material padrão
        self.project_data['materials']['CONCRETE'] = {'E': 25000000.0, 'nu': 0.2}
        # Seção padrão
        self.project_data['section_data']['RECT_DEFAULT'] = {
            'geometry': 'rectangular',
            'material_id': 'CONCRETE',
            'geometry_params': {'width': 0.3, 'height': 0.5}
        }
        self._refresh_tables()

    def _setup_ui(self):
        """Configura a interface completa."""
        # Stylesheet global aprimorado
        self.setStyleSheet("""
            QDialog {
                background-color: #f8fafc;
                color: #1e293b;
            }

            QLabel {
                color: #334155;
                font-size: 13px;
                font-weight: 500;
            }

            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: white;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 6px;
                color: #1e293b;
            }

            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #3b82f6;
            }

            QGroupBox {
                font-weight: bold;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                margin-top: 1.2em;
                background-color: white;
                color: #1e293b;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #1e293b;
            }

            QTableWidget {
                background-color: white;
                border: 1px solid #e2e8f0;
                gridline-color: #f1f5f9;
                color: #1e293b;
            }

            QTableWidget::item {
                color: #1e293b;
            }

            QHeaderView::section {
                background-color: #f1f5f9;
                color: #475569;
                padding: 8px;
                border: none;
                border-bottom: 2px solid #e2e8f0;
                font-weight: bold;
            }

            QPushButton#ActionBtn {
                background-color: #e2e8f0;
                color: #334155;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }

            QPushButton#ActionBtn:hover {
                background-color: #cbd5e1;
                color: #1e293b;
            }

            QPushButton#PrimaryBtn {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
            }

            QPushButton#PrimaryBtn:hover {
                background-color: #2563eb;
            }

            QCheckBox {
                color: #1e293b;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Splitter principal
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Painel esquerdo (formulários)
        left_panel = self._create_left_panel()

        # Painel direito (visualização)
        right_panel = self._create_right_panel()

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        layout.addWidget(splitter)

    def _create_left_panel(self):
        """Cria o painel esquerdo com abas."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Abas principais
        self.tabs = QTabWidget()
        self.tabs.setIconSize(QSize(20, 20))

        # CORREÇÃO: Estilo das abas com botões de navegação visíveis
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background-color: #f8fafc;
                border-right: 1px solid #e2e8f0;
            }

            QTabBar::tab {
                background-color: #f1f5f9;
                color: #64748b;
                padding: 10px 15px;
                border: none;
                border-bottom: 1px solid #e2e8f0;
            }

            QTabBar::tab:selected {
                background-color: white;
                color: #3b82f6;
                font-weight: bold;
                border-left: 3px solid #3b82f6;
            }

            QTabBar::tab:hover {
                background-color: #e2e8f0;
            }

            /* CORREÇÃO: Botões de navegação visíveis */
            QTabBar::scroller {
                width: 30px;
            }

            QTabBar QToolButton {
                background-color: #e2e8f0;
                color: #475569;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 4px;
            }

            QTabBar QToolButton:hover {
                background-color: #cbd5e1;
            }

            QTabBar QToolButton::right-arrow {
                image: none;
                border: none;
                width: 0px;
            }

            QTabBar QToolButton::left-arrow {
                image: none;
                border: none;
                width: 0px;
            }
        """)

        # Adicionar abas
        self.tabs.addTab(self._create_general_tab(), qta.icon('fa5s.info-circle', color='#64748b'), "Geral")
        self.tabs.addTab(self._create_materials_tab(), qta.icon('fa5s.layer-group', color='#64748b'), "Materiais")
        self.tabs.addTab(self._create_sections_tab(), qta.icon('fa5s.shapes', color='#64748b'), "Seções")
        self.tabs.addTab(self._create_nodes_tab(), qta.icon('fa5s.dot-circle', color='#64748b'), "Nós")
        self.tabs.addTab(self._create_elements_tab(), qta.icon('fa5s.project-diagram', color='#64748b'), "Elementos")
        self.tabs.addTab(self._create_supports_tab(), qta.icon('fa5s.anchor', color='#64748b'), "Apoios")
        self.tabs.addTab(self._create_loads_tab(), qta.icon('fa5s.weight-hanging', color='#64748b'), "Cargas")

        layout.addWidget(self.tabs)

        # Barra de ação inferior
        action_bar = QWidget()
        action_bar.setStyleSheet("background-color: white; border-top: 1px solid #e2e8f0;")
        ab_layout = QHBoxLayout(action_bar)
        ab_layout.setContentsMargins(20, 15, 20, 15)

        btn_save = QPushButton(" Salvar Projeto")
        btn_save.setObjectName("PrimaryBtn")
        btn_save.setIcon(qta.icon('fa5s.save', color='white'))
        btn_save.clicked.connect(self.save_structure)

        ab_layout.addStretch()
        ab_layout.addWidget(btn_save)

        layout.addWidget(action_bar)
        return panel

    def _create_right_panel(self):
        """Cria o painel direito com visualização 3D."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Cabeçalho
        header = QWidget()
        header.setStyleSheet("background-color: white; border-bottom: 1px solid #e2e8f0;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(15, 10, 15, 10)
        h_layout.addWidget(QLabel("<b>Visualização 3D em Tempo Real</b>"))
        h_layout.addStretch()

        btn_reset = QPushButton(qta.icon('fa5s.compress-arrows-alt', color='#64748b'), "")
        btn_reset.setFlat(True)
        btn_reset.setToolTip("Resetar Câmera")
        btn_reset.clicked.connect(lambda: self.plotter.reset_camera())
        h_layout.addWidget(btn_reset)

        # Plotter PyVista
        self.plotter = QtInteractor()
        self.plotter.set_background('white')
        self.plotter.add_axes()

        layout.addWidget(header)
        layout.addWidget(self.plotter)
        return panel

    def _create_header(self, title, desc):
        """Cria cabeçalho de seção."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 20)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1e293b;")

        desc_label = QLabel(desc)
        desc_label.setStyleSheet("color: #64748b; font-size: 13px;")
        desc_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        return widget

    def _create_action_button(self, text, icon_name, slot):
        """Cria botão de ação."""
        btn = QPushButton(f" {text}")
        btn.setObjectName("ActionBtn")
        btn.setIcon(qta.icon(icon_name, color='#475569'))
        btn.clicked.connect(slot)
        return btn

    def _update_plot(self):
        self.plot_timer.start(300)

    # =========================================================================
    # ABAS
    # =========================================================================

    def _create_general_tab(self):
        """Aba de informações gerais."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(self._create_header("Informações Gerais", "Configurações básicas do projeto e análise"))

        # Metadados
        grp_info = QGroupBox("Metadados do Projeto")
        frm_info = QFormLayout(grp_info)
        frm_info.setSpacing(15)

        self.inp_name = QLineEdit("Novo Projeto")
        self.inp_desc = QLineEdit()
        self.inp_author = QLineEdit()

        frm_info.addRow("Nome do Projeto:", self.inp_name)
        frm_info.addRow("Descrição:", self.inp_desc)
        frm_info.addRow("Autor:", self.inp_author)
        layout.addWidget(grp_info)

        # Configurações de análise
        grp_sol = QGroupBox("Configurações de Análise")
        frm_sol = QFormLayout(grp_sol)
        frm_sol.setSpacing(15)

        self.cb_type = QComboBox()
        self.cb_type.addItems(['linear', 'nonlinear', 'buckling'])

        self.cb_elem = QComboBox()
        self.cb_elem.addItems(['beam', 'truss'])

        self.sb_subdiv = QSpinBox()
        self.sb_subdiv.setRange(1, 20)
        self.sb_subdiv.setValue(4)

        frm_sol.addRow("Tipo de Análise:", self.cb_type)
        frm_sol.addRow("Tipo de Elemento:", self.cb_elem)
        frm_sol.addRow("Número de Subdivisões:", self.sb_subdiv)
        layout.addWidget(grp_sol)

        layout.addStretch()
        return page

    def _create_materials_tab(self):
        """Aba de materiais."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(self._create_header("Materiais", "Defina as propriedades elásticas dos materiais"))

        self.table_mat = QTableWidget(0, 3)
        self.table_mat.setHorizontalHeaderLabels(["ID", "E (Pa)", "ν (Poisson)"])
        self.table_mat.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        btn_layout = QHBoxLayout()
        btn_add = self._create_action_button("Adicionar Material", 'fa5s.plus', self._add_material)
        btn_rem = self._create_action_button("Remover Selecionado", 'fa5s.trash',
                                             lambda: self._remove_row(self.table_mat))

        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_rem)
        btn_layout.addStretch()

        layout.addWidget(self.table_mat)
        layout.addLayout(btn_layout)
        return page

    def _create_sections_tab(self):
        """Aba de seções transversais."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(self._create_header("Seções Transversais", "Defina a geometry_type dos perfis estruturais"))

        self.table_sec = QTableWidget(0, 4)
        self.table_sec.setHorizontalHeaderLabels(["ID", "Geometria", "Material", "Parâmetros"])
        self.table_sec.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        btn_layout = QHBoxLayout()
        btn_add = self._create_action_button("Criar Nova Seção", 'fa5s.magic', self._open_section_editor)
        btn_rem = self._create_action_button("Remover Selecionada", 'fa5s.trash',
                                             lambda: self._remove_row(self.table_sec))

        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_rem)
        btn_layout.addStretch()

        layout.addWidget(self.table_sec)
        layout.addLayout(btn_layout)
        return page

    def _create_nodes_tab(self):
        """Aba de nós."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(self._create_header("Nós", "Defina as coordenadas dos pontos nodais"))

        self.table_nodes = QTableWidget(0, 4)
        self.table_nodes.setHorizontalHeaderLabels(["ID", "X (m)", "Y (m)", "Z (m)"])
        self.table_nodes.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_nodes.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table_nodes.setColumnWidth(0, 50)

        btn_layout = QHBoxLayout()
        btn_add = self._create_action_button("Adicionar Nó", 'fa5s.plus', self._add_node_row)
        btn_rem = self._create_action_button("Remover Selecionado", 'fa5s.trash',
                                             lambda: self._remove_row(self.table_nodes))

        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_rem)
        btn_layout.addStretch()

        layout.addWidget(self.table_nodes)
        layout.addLayout(btn_layout)

        self.table_nodes.cellChanged.connect(self._schedule_plot)
        return page

    def _create_elements_tab(self):
        """Aba de elements."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(self._create_header("Elementos", "Conecte os nós para formar elements estruturais"))

        self.table_elems = QTableWidget(0, 3)
        self.table_elems.setHorizontalHeaderLabels(["Nó Inicial", "Nó Final", "Seção"])
        self.table_elems.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        btn_layout = QHBoxLayout()
        btn_add = self._create_action_button("Adicionar Elemento", 'fa5s.plus', self._add_elem_row)
        btn_rem = self._create_action_button("Remover Selecionado", 'fa5s.trash',
                                             lambda: self._remove_row(self.table_elems))

        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_rem)
        btn_layout.addStretch()

        layout.addWidget(self.table_elems)
        layout.addLayout(btn_layout)

        self.table_elems.cellChanged.connect(self._schedule_plot)
        return page

    def _create_supports_tab(self):
        """Aba de apoios."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(self._create_header("Apoios", "Defina as restrições de movimento dos nós"))

        self.table_supp = QTableWidget(0, 2)
        self.table_supp.setHorizontalHeaderLabels(["Nós (ex: 0,1,2)", "Restrições (ex: 0,1,2)"])
        self.table_supp.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        btn_layout = QHBoxLayout()
        btn_add = self._create_action_button("Adicionar Apoio", 'fa5s.plus', self._add_support)
        btn_rem = self._create_action_button("Remover Selecionado", 'fa5s.trash',
                                             lambda: self._remove_row(self.table_supp))

        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_rem)
        btn_layout.addStretch()

        layout.addWidget(self.table_supp)
        layout.addLayout(btn_layout)

        self.table_supp.cellChanged.connect(self._schedule_plot)
        return page

    def _create_loads_tab(self):
        """Aba de cargas."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(self._create_header("Cargas Nodais", "Aplique forças e momentos nodes nós"))

        self.table_loads = QTableWidget(0, 7)
        self.table_loads.setHorizontalHeaderLabels(
            ["Nó", "Fx (kN)", "Fy (kN)", "Fz (kN)", "Mx (kN.m)", "My (kN.m)", "Mz (kN.m)"])
        self.table_loads.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        btn_layout = QHBoxLayout()
        btn_add = self._create_action_button("Adicionar Carga", 'fa5s.plus', self._add_load)
        btn_rem = self._create_action_button("Remover Selecionada", 'fa5s.trash',
                                             lambda: self._remove_row(self.table_loads))

        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_rem)
        btn_layout.addStretch()

        layout.addWidget(self.table_loads)
        layout.addLayout(btn_layout)
        return page

    # =========================================================================
    # LÓGICA DE MANIPULAÇÃO DE DADOS
    # =========================================================================

    def _add_material(self):
        """Abre diálogo para adicionar material."""
        dialog = AddMaterialDialog(self)
        if dialog.exec():
            data = dialog.get_data()
            mat_id = data['id']

            # Salvar no dicionário
            self.project_data['materials'][mat_id] = {
                'E': data['E'],
                'nu': data['nu']
            }

            # Adicionar na tabela
            row = self.table_mat.rowCount()
            self.table_mat.insertRow(row)
            self.table_mat.setItem(row, 0, QTableWidgetItem(mat_id))
            self.table_mat.setItem(row, 1, QTableWidgetItem(str(data['E'])))
            self.table_mat.setItem(row, 2, QTableWidgetItem(str(data['nu'])))

    def _add_node_row(self):
        """Adiciona uma nova linha de nó."""
        row = self.table_nodes.rowCount()
        self.table_nodes.insertRow(row)

        # ID não editável
        item = QTableWidgetItem(str(row))
        item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
        self.table_nodes.setItem(row, 0, item)

        # Coordenadas
        for c in range(1, 4):
            self.table_nodes.setItem(row, c, QTableWidgetItem("0.0"))

    def _add_elem_row(self):
        """Adiciona uma nova linha de elemento."""
        row = self.table_elems.rowCount()
        self.table_elems.insertRow(row)

        self.table_elems.setItem(row, 0, QTableWidgetItem("0"))
        self.table_elems.setItem(row, 1, QTableWidgetItem("1"))

        # ComboBox de seções
        cb = QComboBox()
        secs = list(self.project_data['section_data'].keys())
        if not secs:
            secs = ["RECT_DEFAULT"]
        cb.addItems(secs)
        cb.currentTextChanged.connect(self._schedule_plot)
        self.table_elems.setCellWidget(row, 2, cb)

    def _add_support(self):
        """Abre diálogo para adicionar apoio."""
        num_nodes = self.table_nodes.rowCount()
        if num_nodes == 0:
            QMessageBox.warning(self, "Aviso", "Adicione nós primeiro.")
            return

        dialog = AddSupportDialog(num_nodes, self)
        if dialog.exec():
            data = dialog.get_data()

            row = self.table_supp.rowCount()
            self.table_supp.insertRow(row)
            self.table_supp.setItem(row, 0, QTableWidgetItem(str(data['nodes'])))
            self.table_supp.setItem(row, 1, QTableWidgetItem(str(data['boundary_conditions'])))

            self._schedule_plot()

    def _add_load(self):
        """Abre diálogo para adicionar carga."""
        num_nodes = self.table_nodes.rowCount()
        if num_nodes == 0:
            QMessageBox.warning(self, "Aviso", "Adicione nós primeiro.")
            return

        dialog = AddLoadDialog(num_nodes, self)
        if dialog.exec():
            data = dialog.get_data()

            row = self.table_loads.rowCount()
            self.table_loads.insertRow(row)
            self.table_loads.setItem(row, 0, QTableWidgetItem(str(data['node'])))
            for i, val in enumerate(data['load']):
                self.table_loads.setItem(row, i + 1, QTableWidgetItem(str(val)))

    def _remove_row(self, table):
        """Remove linha selecionada de uma tabela."""
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)
            self._schedule_plot()

    def _open_section_editor(self):
        """Abre editor de seções."""
        mats = list(self.project_data['materials'].keys())
        if not mats:
            QMessageBox.warning(self, "Aviso", "Adicione materiais primeiro.")
            return

        dialog = SectionEditorDialog(mats, self)
        if dialog.exec():
            data = dialog.get_data()
            sec_id = data['id']

            # Salvar no dicionário
            self.project_data['section_data'][sec_id] = {
                'geometry': data['geometry'],
                'material_id': data['material_id'],
                'geometry_params': data['params']
            }

            # Adicionar na tabela
            row = self.table_sec.rowCount()
            self.table_sec.insertRow(row)
            self.table_sec.setItem(row, 0, QTableWidgetItem(sec_id))
            self.table_sec.setItem(row, 1, QTableWidgetItem(data['geometry']))
            self.table_sec.setItem(row, 2, QTableWidgetItem(data['material_id']))

            # Formatar parâmetros
            p_str = ", ".join([f"{k}={v:.3f}" for k, v in data['params'].items()])
            self.table_sec.setItem(row, 3, QTableWidgetItem(p_str))

            self._schedule_plot()

    def _refresh_tables(self):
        """Atualiza tabelas com dados atuais."""
        # Material padrão na tabela
        if 'CONCRETE' in self.project_data['materials']:
            mat = self.project_data['materials']['CONCRETE']
            self._add_row(self.table_mat, ["CONCRETE", str(mat['E']), str(mat['nu'])])

        # Seção padrão na tabela
        if 'RECT_DEFAULT' in self.project_data['section_data']:
            sec = self.project_data['section_data']['RECT_DEFAULT']
            params = sec['geometry_params']
            p_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            self._add_row(self.table_sec, ["RECT_DEFAULT", sec['geometry'], sec['material_id'], p_str])

    def _add_row(self, table, values):
        """Adiciona linha genérica em tabela."""
        row = table.rowCount()
        table.insertRow(row)
        for i, val in enumerate(values):
            table.setItem(row, i, QTableWidgetItem(str(val)))

    def _scrape_all_data(self):
        """Extrai todos os dados das tabelas para o dicionário."""
        # Info
        self.project_data['info']['project_name'] = self.inp_name.text()
        self.project_data['info']['description'] = self.inp_desc.text()
        self.project_data['info']['author'] = self.inp_author.text()

        # Analysis
        self.project_data['analysis']['analysis_type'] = self.cb_type.currentText()
        self.project_data['analysis']['element_type'] = self.cb_elem.currentText()
        self.project_data['analysis']['number_of_subdivisions'] = self.sb_subdiv.value()

        # Materials (da tabela)
        mats = {}
        for r in range(self.table_mat.rowCount()):
            try:
                mid = self.table_mat.item(r, 0).text()
                e = float(self.table_mat.item(r, 1).text())
                nu = float(self.table_mat.item(r, 2).text())
                mats[mid] = {'E': e, 'nu': nu}
            except:
                continue
        if mats:
            self.project_data['materials'] = mats

        # Nodes
        nodes = []
        for r in range(self.table_nodes.rowCount()):
            try:
                x = float(self.table_nodes.item(r, 1).text())
                y = float(self.table_nodes.item(r, 2).text())
                z = float(self.table_nodes.item(r, 3).text())
                nodes.append([x, y, z])
            except:
                continue
        self.project_data['nodes'] = nodes

        # Elements
        elems = []
        for r in range(self.table_elems.rowCount()):
            try:
                n1 = int(self.table_elems.item(r, 0).text())
                n2 = int(self.table_elems.item(r, 1).text())
                sec = self.table_elems.cellWidget(r, 2).currentText()
                if n1 < len(nodes) and n2 < len(nodes):
                    elems.append({'connec': [n1, n2], 'section_id': sec})
            except:
                continue
        self.project_data['elements'] = elems

        # Supports
        supps = []
        for r in range(self.table_supp.rowCount()):
            try:
                n_str = self.table_supp.item(r, 0).text()
                bc_str = self.table_supp.item(r, 1).text()
                n_ids = [int(x.strip()) for x in n_str.split(',') if x.strip()]
                bcs = [int(x.strip()) for x in bc_str.split(',') if x.strip()]
                supps.append({'node_id': n_ids if len(n_ids) > 1 else n_ids[0], 'boundary_conditions': bcs})
            except:
                continue
        self.project_data['supports'] = supps

        # Loads
        loads = []
        for r in range(self.table_loads.rowCount()):
            try:
                nid = int(self.table_loads.item(r, 0).text())
                vals = [float(self.table_loads.item(r, c).text()) for c in range(1, 7)]
                loads.append({'node_id': nid, 'load': vals})
            except:
                continue
        self.project_data['nodal_loads'] = loads

    def _schedule_plot(self):
        """Agenda atualização do plot."""
        self.plot_timer.start(300)

    def _update_plot_now(self):
        """Atualiza visualização 3D."""
        self._scrape_all_data()

        # Atualizar combos de elements se necessário
        curr_secs = list(self.project_data['section_data'].keys())
        for r in range(self.table_elems.rowCount()):
            cb = self.table_elems.cellWidget(r, 2)
            if cb and cb.count() != len(curr_secs):
                curr = cb.currentText()
                cb.clear()
                cb.addItems(curr_secs)
                if curr in curr_secs:
                    cb.setCurrentText(curr)

        # Chamar controlador para atualizar visualização
        self.controller.update_structure_preview(self.project_data, self.plotter)

    def save_structure(self):
        """Salva estrutura em arquivo YAML."""
        self._scrape_all_data()

        if len(self.project_data['nodes']) < 2:
            QMessageBox.warning(self, "Validação", "São necessários pelo menos 2 nós.")
            return

        if len(self.project_data['elements']) < 1:
            QMessageBox.warning(self, "Validação", "É necessário pelo menos 1 elemento.")
            return

        # Solicitar local de salvamento
        default_name = self.inp_name.text().replace(' ', '_')
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar Estrutura",
            f"SimuFrame/elements/{default_name}.yaml",
            "YAML Files (*.yaml *.yml)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                self.yaml.dump(self.project_data, f)

            QMessageBox.information(self, "Sucesso", f"Estrutura salva com sucesso em:\n{file_path}")

            # Fechar diálogo apenas após salvamento bem-sucedido
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao salvar estrutura:\n{str(e)}")

    def closeEvent(self, event):
        """CORREÇÃO VTK: Limpar recursos antes de fechar."""
        self.plot_timer.stop()

        if hasattr(self, 'plotter'):
            try:
                # Remover do layout
                if self.plotter.parent():
                    self.plotter.setParent(None)

                # Fechar e limpar
                self.plotter.close()
                self.plotter.deleteLater()
            except:
                pass

        event.accept()


# =============================================================================
# DIÁLOGOS AUXILIARES
# =============================================================================

class AddMaterialDialog(QDialog):
    """Diálogo para adicionar material."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adicionar Material")
        self.resize(400, 250)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("""
            QDialog { background-color: #f8fafc; color: #1e293b; }
            QLineEdit, QDoubleSpinBox { padding: 6px; border: 1px solid #cbd5e1; border-radius: 4px; background: white; color: #1e293b; }
            QLabel { font-weight: bold; color: #475569; }
            QPushButton { background-color: #3b82f6; color: white; padding: 8px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #2563eb; }
        """)

        layout = QFormLayout(self)
        layout.setSpacing(15)

        self.inp_id = QLineEdit("STEEL")

        self.spin_e = QDoubleSpinBox()
        self.spin_e.setRange(1e6, 1e12)
        self.spin_e.setDecimals(0)
        self.spin_e.setValue(210e9)
        self.spin_e.setSuffix(" Pa")

        self.spin_nu = QDoubleSpinBox()
        self.spin_nu.setRange(0.0, 0.5)
        self.spin_nu.setDecimals(3)
        self.spin_nu.setValue(0.3)

        layout.addRow("ID do Material:", self.inp_id)
        layout.addRow("Módulo de Elasticidade (E):", self.spin_e)
        layout.addRow("Coeficiente de Poisson (ν):", self.spin_nu)

        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("Confirmar")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setStyleSheet("background-color: #e2e8f0; color: #334155;")
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_ok)
        layout.addRow(btn_layout)

    def get_data(self):
        return {
            'id': self.inp_id.text(),
            'E': self.spin_e.value(),
            'nu': self.spin_nu.value()
        }


class AddSupportDialog(QDialog):
    """Diálogo para adicionar apoio."""

    def __init__(self, num_nodes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adicionar Apoio")
        self.resize(450, 400)
        self.num_nodes = num_nodes
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("""
            QDialog { background-color: #f8fafc; color: #1e293b; }
            QLineEdit { padding: 6px; border: 1px solid #cbd5e1; border-radius: 4px; background: white; color: #1e293b; }
            QLabel { color: #475569; }
            QGroupBox { font-weight: bold; border: 1px solid #e2e8f0; border-radius: 6px; margin-top: 1em; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
            QCheckBox { color: #1e293b; }
            QPushButton { background-color: #3b82f6; color: white; padding: 8px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #2563eb; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Nós
        form = QFormLayout()
        self.inp_nodes = QLineEdit()
        self.inp_nodes.setPlaceholderText("Ex: 0,1,2")
        form.addRow("Nós (IDs separados por vírgula):", self.inp_nodes)
        layout.addLayout(form)

        # Restrições
        group = QGroupBox("Graus de Liberdade Restritos")
        group_layout = QVBoxLayout()

        self.checks = []
        dofs = [
            ('X (Translation)', 0),
            ('Y (Translation)', 1),
            ('Z (Translation)', 2),
            ('RX (Rotation)', 3),
            ('RY (Rotation)', 4),
            ('RZ (Rotation)', 5)
        ]

        for label, _ in dofs:
            check = QCheckBox(label)
            group_layout.addWidget(check)
            self.checks.append(check)

        group.setLayout(group_layout)
        layout.addWidget(group)

        # Presets
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))

        btn_fixed = QPushButton("Engaste")
        btn_fixed.setStyleSheet("background-color: #e2e8f0; color: #334155; padding: 6px;")
        btn_fixed.clicked.connect(lambda: self._set_preset([0, 1, 2, 3, 4, 5]))

        btn_pinned = QPushButton("Rótula")
        btn_pinned.setStyleSheet("background-color: #e2e8f0; color: #334155; padding: 6px;")
        btn_pinned.clicked.connect(lambda: self._set_preset([0, 1, 2]))

        btn_roller = QPushButton("Apoio Móvel")
        btn_roller.setStyleSheet("background-color: #e2e8f0; color: #334155; padding: 6px;")
        btn_roller.clicked.connect(lambda: self._set_preset([1]))

        preset_layout.addWidget(btn_fixed)
        preset_layout.addWidget(btn_pinned)
        preset_layout.addWidget(btn_roller)
        preset_layout.addStretch()

        layout.addLayout(preset_layout)

        # Botões
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("Confirmar")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setStyleSheet("background-color: #e2e8f0; color: #334155;")
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_ok)
        layout.addLayout(btn_layout)

    def _set_preset(self, dofs):
        for i, check in enumerate(self.checks):
            check.setChecked(i in dofs)

    def get_data(self):
        node_str = self.inp_nodes.text()
        nodes = [int(x.strip()) for x in node_str.split(',') if x.strip()]
        bc = [i for i, check in enumerate(self.checks) if check.isChecked()]

        return {
            'nodes': nodes if len(nodes) > 1 else nodes[0],
            'boundary_conditions': bc
        }


class AddLoadDialog(QDialog):
    """Diálogo para adicionar carga nodal."""

    def __init__(self, num_nodes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adicionar Carga Nodal")
        self.resize(400, 400)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("""
            QDialog { background-color: #f8fafc; color: #1e293b; }
            QSpinBox, QDoubleSpinBox { padding: 6px; border: 1px solid #cbd5e1; border-radius: 4px; background: white; color: #1e293b; }
            QLabel { font-weight: bold; color: #475569; }
            QPushButton { background-color: #3b82f6; color: white; padding: 8px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #2563eb; }
        """)

        layout = QFormLayout(self)
        layout.setSpacing(15)

        self.spin_node = QSpinBox()
        self.spin_node.setRange(0, 1000)

        self.spin_fx = QDoubleSpinBox()
        self.spin_fx.setRange(-10000, 10000)
        self.spin_fx.setSuffix(" kN")

        self.spin_fy = QDoubleSpinBox()
        self.spin_fy.setRange(-10000, 10000)
        self.spin_fy.setSuffix(" kN")

        self.spin_fz = QDoubleSpinBox()
        self.spin_fz.setRange(-10000, 10000)
        self.spin_fz.setSuffix(" kN")

        self.spin_mx = QDoubleSpinBox()
        self.spin_mx.setRange(-10000, 10000)
        self.spin_mx.setSuffix(" kN.m")

        self.spin_my = QDoubleSpinBox()
        self.spin_my.setRange(-10000, 10000)
        self.spin_my.setSuffix(" kN.m")

        self.spin_mz = QDoubleSpinBox()
        self.spin_mz.setRange(-10000, 10000)
        self.spin_mz.setSuffix(" kN.m")

        layout.addRow("Nó:", self.spin_node)
        layout.addRow("Força em X (Fx):", self.spin_fx)
        layout.addRow("Força em Y (Fy):", self.spin_fy)
        layout.addRow("Força em Z (Fz):", self.spin_fz)
        layout.addRow("Momento em X (Mx):", self.spin_mx)
        layout.addRow("Momento em Y (My):", self.spin_my)
        layout.addRow("Momento em Z (Mz):", self.spin_mz)

        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("Confirmar")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setStyleSheet("background-color: #e2e8f0; color: #334155;")
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_ok)
        layout.addRow(btn_layout)

    def get_data(self):
        return {
            'node': self.spin_node.value(),
            'load': [
                self.spin_fx.value(),
                self.spin_fy.value(),
                self.spin_fz.value(),
                self.spin_mx.value(),
                self.spin_my.value(),
                self.spin_mz.value()
            ]
        }


class SectionEditorDialog(QDialog):
    """Diálogo avançado para edição de seções com abas dinâmicas."""

    # Mapeamento de geometrias e seus parâmetros
    GEOMETRY_PARAMS = {
        'rectangular': {
            'params': ['width', 'height'],
            'labels': ['Width (Largura)', 'Height (Altura)'],
            'defaults': [0.3, 0.5]
        },
        'circular': {
            'params': ['diameter'],
            'labels': ['Diameter (Diâmetro)'],
            'defaults': [0.4]
        },
        'rhs': {
            'params': ['width', 'height', 'thickness'],
            'labels': ['Width (Largura)', 'Height (Altura)', 'Thickness (Espessura)'],
            'defaults': [0.2, 0.4, 0.01]
        },
        'chs': {
            'params': ['outer_diameter', 'thickness'],
            'labels': ['Outer Diameter (Diâmetro Externo)', 'Thickness (Espessura)'],
            'defaults': [0.3, 0.008]
        },
        'i_section': {
            'params': ['height', 'width', 'tf', 'tw'],
            'labels': ['Height (Altura Total)', 'Width (Largura do Flange)', 'Flange Thickness (tf)',
                       'Web Thickness (tw)'],
            'defaults': [0.4, 0.2, 0.012, 0.008]
        },
        't_section': {
            'params': ['height', 'width', 'tf', 'tw'],
            'labels': ['Height (Altura)', 'Width (Largura)', 'Flange Thickness (tf)', 'Web Thickness (tw)'],
            'defaults': [0.3, 0.15, 0.01, 0.008]
        },
        'c_channel': {
            'params': ['height', 'width', 'tf', 'tw'],
            'labels': ['Height (Altura)', 'Width (Largura)', 'Flange Thickness (tf)', 'Web Thickness (tw)'],
            'defaults': [0.25, 0.1, 0.01, 0.008]
        },
        'angle': {
            'params': ['height', 'width', 'thickness'],
            'labels': ['Height (Perna Vertical)', 'Width (Perna Horizontal)', 'Thickness (Espessura)'],
            'defaults': [0.1, 0.1, 0.008]
        }
    }

    def __init__(self, materials, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Editor de Seção Transversal")
        self.resize(500, 600)
        self.materials = materials
        self.param_inputs = {}
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("""
            QDialog { background-color: #f8fafc; color: #1e293b; }
            QLineEdit, QComboBox, QDoubleSpinBox {
                padding: 6px; border: 1px solid #cbd5e1;
                border-radius: 4px; background: white; color: #1e293b;
            }
            QLabel { color: #475569; font-weight: 500; }
            QGroupBox {
                font-weight: bold; border: 1px solid #e2e8f0;
                border-radius: 6px; margin-top: 1em; padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left;
                padding: 0 5px; color: #1e293b;
            }
            QPushButton {
                background-color: #3b82f6; color: white;
                padding: 8px 16px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2563eb; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Cabeçalho
        header = QLabel("Defina a Geometria da Seção Transversal")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #1e293b;")
        layout.addWidget(header)

        # ID da seção
        form = QFormLayout()
        self.inp_id = QLineEdit("NEW_SECTION")
        form.addRow("ID da Seção:", self.inp_id)

        # Material
        self.cb_material = QComboBox()
        self.cb_material.addItems(self.materials)
        form.addRow("Material:", self.cb_material)

        # Tipo de geometry_type
        self.cb_geometry = QComboBox()
        self.cb_geometry.addItems(list(self.GEOMETRY_PARAMS.keys()))
        self.cb_geometry.currentTextChanged.connect(self._update_params)
        form.addRow("Tipo de Geometria:", self.cb_geometry)

        layout.addLayout(form)

        # Área de parâmetros dinâmicos
        self.params_group = QGroupBox("Parâmetros Geométricos (em metros)")
        self.params_layout = QFormLayout(self.params_group)
        self.params_layout.setSpacing(12)
        layout.addWidget(self.params_group)

        # Inicializar com primeira geometry_type
        self._update_params(self.cb_geometry.currentText())

        layout.addStretch()

        # Botões
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("Confirmar")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setStyleSheet("background-color: #e2e8f0; color: #334155;")
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_ok)
        layout.addLayout(btn_layout)

    def _update_params(self, geo_type):
        """Atualiza campos de parâmetros baseado na geometry_type selecionada."""
        # Limpar campos anteriores
        while self.params_layout.count():
            child = self.params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.param_inputs = {}

        # Obter configuração da geometry_type
        config = self.GEOMETRY_PARAMS.get(geo_type, {})
        params = config.get('params', [])
        labels = config.get('labels', [])
        defaults = config.get('defaults', [])

        # Criar campos dinâmicos
        for param, label, default in zip(params, labels, defaults):
            spin = QDoubleSpinBox()
            spin.setRange(0.001, 10.0)
            spin.setDecimals(3)
            spin.setValue(default)
            spin.setSuffix(" m")

            self.params_layout.addRow(f"{label}:", spin)
            self.param_inputs[param] = spin

    def get_data(self):
        """Retorna dados da seção."""
        params = {k: v.value() for k, v in self.param_inputs.items()}
        return {
            'id': self.inp_id.text(),
            'geometry': self.cb_geometry.currentText(),
            'material_id': self.cb_material.currentText(),
            'params': params
        }