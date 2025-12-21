# Third-party libraries
import numpy as np
import pyqtgraph as pg
import matplotlib
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QSpinBox,
    QTableWidget, QTableWidgetItem, QWidget, QLabel, QDialogButtonBox,
    QPushButton, QHeaderView, QGroupBox, QComboBox, QFormLayout,
    QListWidget, QSplitter, QFileDialog,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from sklearn.metrics import r2_score
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from SimuFrame.GUI.interactive.InteractiveCursor import InteractiveCursor
matplotlib.use('QtAgg')


class ConvergenceDialog(QDialog):
    """
    Diálogo para visualizar os dados de convergência do método Arc-Length.
    """
    def __init__(self, convergence_data, parent=None):
        """
        Args:
            convergence_data (dict): Dicionário contendo os dados de convergência:
                - 'increments': lista de incrementos
                - 'lambda_history': histórico de fatores de carga
                - 'max_displ_history': histórico de deslocamentos máximos
                - 'iteration_details': detalhes de cada iteração
        """
        super().__init__(parent)
        self.convergence_data = convergence_data
        self.setWindowTitle("Gráficos de Cálculo - Método Arc-Length")
        self.resize(1200, 800)
        
        self.setup_ui()
        self.populate_data()
    
    def setup_ui(self):
        """Configura a interface do usuário."""
        layout = QVBoxLayout(self)
        
        # Criar o TabWidget principal
        self.tab_widget = QTabWidget()
        
        # Aba 1: Main (Informações gerais)
        self.main_tab = self.create_main_tab()
        self.tab_widget.addTab(self.main_tab, "Main")
        
        # Aba 2: Table (Tabela de incrementos)
        self.table_tab = self.create_table_tab()
        self.tab_widget.addTab(self.table_tab, "Tabela")
        
        # Aba 3: Diagram (Diagrama carga x deslocamento)
        self.diagram_tab = self.create_diagram_tab()
        self.tab_widget.addTab(self.diagram_tab, "Diagrama")
        
        # Aba 4: Convergence Table (Tabela de convergência detalhada)
        self.convergence_tab = self.create_convergence_tab()
        self.tab_widget.addTab(self.convergence_tab, "Tabela de Convergência")
        
        layout.addWidget(self.tab_widget)
        
        # Botões de ação
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        export_btn = QPushButton("Exportar...")
        close_btn = QPushButton("Fechar")
        close_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(export_btn)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def create_main_tab(self):
        """Cria a aba Main com informações gerais."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Título
        title = QLabel("Análise Não Linear - Método Arc-Length")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Informações gerais
        info_layout = QVBoxLayout()
        
        self.info_labels = {}
        info_keys = [
            ("Tipo de Análise:", "analysis_type"),
            ("Método:", "method"),
            ("Total de Incrementos:", "total_increments"),
            ("Incrementos Aceitos:", "accepted_increments"),
            ("Incrementos Rejeitados:", "rejected_increments"),
            ("Fator de Carga Final (λ):", "final_lambda"),
            ("Deslocamento Máximo:", "max_displacement"),
            ("Status:", "status")
        ]
        
        for label_text, key in info_keys:
            h_layout = QHBoxLayout()
            label = QLabel(label_text)
            label.setMinimumWidth(200)
            value_label = QLabel("--")
            value_label.setStyleSheet("font-weight: bold;")
            
            h_layout.addWidget(label)
            h_layout.addWidget(value_label)
            h_layout.addStretch()
            
            info_layout.addLayout(h_layout)
            self.info_labels[key] = value_label
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        return widget
    
    def create_table_tab(self):
        """Cria a aba Table com tabela de incrementos."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Descrição
        desc_label = QLabel("Resumo dos Incrementos de Carga")
        desc_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(desc_label)
        
        # Tabela
        self.increment_table = QTableWidget()
        headers = ["Inc.", "λ", "Δλ", "Δl", "Iterações", "||R|| Final", 
                   "Deslocamento Máx.", "Status", "Observações"]
        self.increment_table.setColumnCount(len(headers))
        self.increment_table.setHorizontalHeaderLabels(headers)
        self.increment_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.increment_table.horizontalHeader().setStretchLastSection(True)
        self.increment_table.setAlternatingRowColors(True)
        
        layout.addWidget(self.increment_table)
        
        return widget
    
    def create_diagram_tab(self):
        """Cria a aba Diagram com o gráfico carga x deslocamento."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Descrição
        desc_label = QLabel("Caminho de Equilíbrio - Fator de Carga × Deslocamento Máximo")
        desc_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(desc_label)
        
        # Gráfico
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', "Fator de Carga (λ)", color='k', size='12pt')
        self.plot_widget.setLabel('bottom', "Deslocamento Máximo (m)", color='k', size='12pt')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        
        # Configurar cores para fundo branco
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color='k', width=2))
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(color='k', width=2))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color='k'))
        self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color='k'))
        
        layout.addWidget(self.plot_widget)
        
        return widget
    
    def create_convergence_tab(self):
        """Cria a aba Convergence Table com detalhes de cada iteração."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Descrição
        desc_label = QLabel("Detalhes de Convergência por Incremento e Iteração")
        desc_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(desc_label)
        
        # Tabela detalhada
        self.convergence_table = QTableWidget()
        headers = ["Inc.", "Iter.", "λ", "Δλ", "δλ", "||R||", "||δd||", 
                   "Critério Força", "Critério Deslocamento", "Critério Energia", "Convergiu"]
        self.convergence_table.setColumnCount(len(headers))
        self.convergence_table.setHorizontalHeaderLabels(headers)
        self.convergence_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.convergence_table.setAlternatingRowColors(True)
        
        layout.addWidget(self.convergence_table)
        
        return widget
    
    def populate_data(self):
        """Preenche todas as abas com os dados de convergência."""
        if not self.convergence_data:
            return
        
        # Popular aba Main
        self.populate_main_tab()
        
        # Popular aba Table
        self.populate_table_tab()
        
        # Popular aba Diagram
        self.populate_diagram_tab()
        
        # Popular aba Convergence Table
        self.populate_convergence_tab()
    
    def populate_main_tab(self):
        """Preenche a aba Main com informações gerais."""
        data = self.convergence_data
        
        self.info_labels['analysis_type'].setText("Não Linear Geométrica (NLG)")
        self.info_labels['method'].setText("Arc-Length (Crisfield - Controle Esférico)")
        self.info_labels['total_increments'].setText(str(data.get('total_increments', 0)))
        self.info_labels['accepted_increments'].setText(str(data.get('accepted_increments', 0)))
        self.info_labels['rejected_increments'].setText(str(data.get('rejected_increments', 0)))
        
        final_lambda = data.get('final_lambda', 0.0)
        self.info_labels['final_lambda'].setText(f"{final_lambda:.3f}")
        
        max_displ = data.get('max_displacement', 0.0)
        self.info_labels['max_displacement'].setText(f"{max_displ:.3f} m")
        
        status = "✓ Convergiu" if data.get('converged', False) else "✗ Não Convergiu"
        status_color = "color: green;" if data.get('converged', False) else "color: red;"
        self.info_labels['status'].setText(status)
        self.info_labels['status'].setStyleSheet(f"font-weight: bold; {status_color}")
    
    def populate_table_tab(self):
        """Preenche a tabela de incrementos."""
        increments = self.convergence_data.get('increments', [])
        self.increment_table.setRowCount(len(increments))
        
        for row, inc_data in enumerate(increments):
            # Incremento
            self.increment_table.setItem(row, 0, 
                QTableWidgetItem(str(inc_data.get('increment', ''))))
            
            # Lambda
            lambda_val = inc_data.get('lambda', 0.0)
            self.increment_table.setItem(row, 1, 
                QTableWidgetItem(f"{lambda_val:.6f}"))
            
            # Delta Lambda
            delta_lambda = inc_data.get('delta_lambda', 0.0)
            self.increment_table.setItem(row, 2, 
                QTableWidgetItem(f"{delta_lambda:.6e}"))
            
            # Arc-length
            arc_length = inc_data.get('arc_length', 0.0)
            self.increment_table.setItem(row, 3, 
                QTableWidgetItem(f"{arc_length:.6e}"))
            
            # Iterações
            iterations = inc_data.get('iterations', 0)
            self.increment_table.setItem(row, 4, 
                QTableWidgetItem(str(iterations)))
            
            # Norma do resíduo final
            norm_R = inc_data.get('norm_R', 0.0)
            self.increment_table.setItem(row, 5, 
                QTableWidgetItem(f"{norm_R:.2e}"))
            
            # Deslocamento máximo
            max_displ = inc_data.get('max_displacement', 0.0)
            self.increment_table.setItem(row, 6, 
                QTableWidgetItem(f"{max_displ:.6e}"))
            
            # Status
            status = "Aceito" if inc_data.get('accepted', True) else "Rejeitado"
            status_item = QTableWidgetItem(status)
            if not inc_data.get('accepted', True):
                status_item.setForeground(Qt.GlobalColor.red)
            self.increment_table.setItem(row, 7, status_item)
            
            # Observações
            obs = inc_data.get('observations', '')
            self.increment_table.setItem(row, 8, QTableWidgetItem(obs))
    
    def populate_diagram_tab(self):
        """Preenche o diagrama carga x deslocamento."""
        lambda_history = self.convergence_data.get('lambda_history', [])
        max_displ_history = self.convergence_data.get('max_displ_history', [])
        
        if not lambda_history or not max_displ_history:
            return
        
        # Plotar curva principal
        pen = pg.mkPen(color=(0, 150, 255), width=3)
        self.plot_widget.plot(max_displ_history, lambda_history, 
                            pen=pen, symbol='o', symbolSize=8, 
                            symbolBrush=(0, 150, 255), 
                            name='Caminho de Equilíbrio')
        
        # Adicionar pontos rejeitados se houver
        rejected_data = self.convergence_data.get('rejected_points', {'displ': [], 'lambda': []})
        if rejected_data['displ']:
            self.plot_widget.plot(rejected_data['displ'], rejected_data['lambda'],
                                pen=None, symbol='x', symbolSize=10,
                                symbolBrush='r', symbolPen='r',
                                name='Incrementos Rejeitados')
    
    def populate_convergence_tab(self):
        """Preenche a tabela de convergência detalhada."""
        iteration_details = self.convergence_data.get('iteration_details', [])
        self.convergence_table.setRowCount(len(iteration_details))
        
        for row, iter_data in enumerate(iteration_details):
            # Incremento
            self.convergence_table.setItem(row, 0, 
                QTableWidgetItem(str(iter_data.get('increment', ''))))
            
            # Iteração
            self.convergence_table.setItem(row, 1, 
                QTableWidgetItem(str(iter_data.get('iteration', ''))))
            
            # Lambda
            lambda_val = iter_data.get('lambda', 0.0)
            self.convergence_table.setItem(row, 2, 
                QTableWidgetItem(f"{lambda_val:.6f}"))
            
            # Delta Lambda (total do incremento)
            delta_lambda = iter_data.get('delta_lambda', 0.0)
            self.convergence_table.setItem(row, 3, 
                QTableWidgetItem(f"{delta_lambda:.6e}"))
            
            # delta lambda (correção da iteração)
            delta_lambda_iter = iter_data.get('delta_lambda_iter', 0.0)
            self.convergence_table.setItem(row, 4, 
                QTableWidgetItem(f"{delta_lambda_iter:.6e}"))
            
            # Norma do resíduo
            norm_R = iter_data.get('norm_R', 0.0)
            self.convergence_table.setItem(row, 5, 
                QTableWidgetItem(f"{norm_R:.2e}"))
            
            # Norma do incremento de deslocamento
            norm_delta_d = iter_data.get('norm_delta_d', 0.0)
            self.convergence_table.setItem(row, 6, 
                QTableWidgetItem(f"{norm_delta_d:.2e}"))
            
            # Critério de força
            force_criterion = iter_data.get('force_criterion', False)
            force_item = QTableWidgetItem("✓" if force_criterion else "✗")
            force_item.setForeground(Qt.GlobalColor.green if force_criterion else Qt.GlobalColor.red)
            force_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.convergence_table.setItem(row, 7, force_item)
            
            # Critério de deslocamento
            displ_criterion = iter_data.get('displ_criterion', False)
            displ_item = QTableWidgetItem("✓" if displ_criterion else "✗")
            displ_item.setForeground(Qt.GlobalColor.green if displ_criterion else Qt.GlobalColor.red)
            displ_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.convergence_table.setItem(row, 8, displ_item)
            
            # Critério de energia
            energy_criterion = iter_data.get('energy_criterion', False)
            energy_item = QTableWidgetItem("✓" if energy_criterion else "✗")
            energy_item.setForeground(Qt.GlobalColor.green if energy_criterion else Qt.GlobalColor.red)
            energy_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.convergence_table.setItem(row, 9, energy_item)

            # Convergiu?
            converged = iter_data.get('converged', False)
            conv_item = QTableWidgetItem("SIM" if converged else "NÃO")
            if converged:
                conv_item.setForeground(Qt.GlobalColor.green)
                font = conv_item.font()
                font.setBold(True)
                conv_item.setFont(font)
            else:
                conv_item.setForeground(Qt.GlobalColor.red)
                font = conv_item.font()
                font.setBold(True)
                conv_item.setFont(font)
            self.convergence_table.setItem(row, 10, conv_item)


class MplCanvas(FigureCanvas):
    """ Classe para criar um widget de gráfico Matplotlib. """
    def __init__(self, parent=None, width=5, height=4, dpi=120):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.fig.tight_layout()
        super(MplCanvas, self).__init__(self.fig)


class LoadDisplacementDialog(QDialog):
    """
    Diálogo para visualizar gráficos de Fator de Carga × Deslocamento.
    Interface similar ao RFEM com seleção interativa de dados.
    """
    
    def __init__(self, f_vs_d_history, estrutura, analise="Não Linear", parent=None):
        """
        Args:
            f_vs_d_history (list): Lista de tuplas (lambda, force, displacement)
            estrutura: Objeto da estrutura com informações dos nós
            analysis_type (str): Tipo de análise ("Linear", "Não Linear", "Flambagem Linear")
            parent: Widget pai
        """
        super().__init__(parent)
        
        self.f_vs_d_history = f_vs_d_history
        self.estrutura = estrutura
        self.analise = analise
        self.num_nodes = len(estrutura.original_nodes)
        self.cursors = []  # Lista para gerenciar múltiplos cursores

        # Dados teóricos do Abaqus
        self.dados_abaqus = {
        'Ux': np.array([
            [0.0, 0.0],
            [0.000283076, 0.01],
            [0.00113109, 0.02],
            [0.00345385, 0.035],
            [0.00925385, 0.0575],
            [0.0229045, 0.09125],
            [0.0532512, 0.141875],
            [0.115604, 0.217812],
            [0.214647, 0.317813],
            [0.318434, 0.417812],
            [0.418179, 0.517812],
            [0.510137, 0.617813],
            [0.593319, 0.717812],
            [0.667977, 0.817813],
            [0.73485, 0.917813],
            [0.784932, 1.0]
        ]),
        'Uy': np.array([
            [0.0, 0.0],
            [0.0441567, 0.01],
            [0.088256, 0.02],
            [0.154173, 0.035],
            [0.252159, 0.0575],
            [0.395971, 0.09125],
            [0.601251, 0.141875],
            [0.878251, 0.217812],
            [1.18011, 0.317813],
            [1.41601, 0.417812],
            [1.59905, 0.517812],
            [1.74196, 0.617813],
            [1.85497, 0.717812],
            [1.94569, 0.817813],
            [2.0196, 0.917813],
            [2.07097, 1.0]
        ]),
        'Uz': np.array([
            [0.0, 0.0],
            [-0.000283076, 0.01],
            [-0.00113109, 0.02],
            [-0.00345385, 0.035],
            [-0.00925385, 0.0575],
            [-0.0229045, 0.09125],
            [-0.0532512, 0.141875],
            [-0.115604, 0.217812],
            [-0.214647, 0.317813],
            [-0.318434, 0.417812],
            [-0.418179, 0.517812],
            [-0.510137, 0.617813],
            [-0.593319, 0.717812],
            [-0.667977, 0.817813],
            [-0.73485, 0.917813],
            [-0.784932, 1.0]
        ]),
        'Rx': np.array([
            [0.0, 0.0],
            [0.0140807, 0.01],
            [0.0281465, 0.02],
            [0.0491852, 0.035],
            [0.0805126, 0.0575],
            [0.12668, 0.09125],
            [0.193201, 0.141875],
            [0.284779, 0.217812],
            [0.388227, 0.317813],
            [0.472977, 0.417812],
            [0.541973, 0.517812],
            [0.598367, 0.617813],
            [0.644885, 0.717812],
            [0.683661, 0.817813],
            [0.716325, 0.917813],
            [0.739601, 1.0]
        ]),
        'Ry': np.array([
            [0.0, 0.0],
            [0.0, 0.01],
            [0.0, 0.02],
            [0.0, 0.035],
            [0.0, 0.0575],
            [0.0, 0.09125],
            [0.0, 0.141875],
            [0.0, 0.217812],
            [0.0, 0.317813],
            [0.0, 0.417812],
            [0.0, 0.517812],
            [0.0, 0.617813],
            [0.0, 0.717812],
            [0.0, 0.817813],
            [0.0, 0.917813],
            [0.0, 1.0]
        ]),
        'Rz': np.array([
            [0.0, 0.0],
            [0.0140807, 0.01],
            [0.0281465, 0.02],
            [0.0491852, 0.035],
            [0.0805126, 0.0575],
            [0.12668, 0.09125],
            [0.193201, 0.141875],
            [0.284779, 0.217812],
            [0.388227, 0.317813],
            [0.472977, 0.417812],
            [0.541973, 0.517812],
            [0.598367, 0.617813],
            [0.644885, 0.717812],
            [0.683661, 0.817813],
            [0.716325, 0.917813],
            [0.739601, 1.0]
        ])
        }

        # Dados teóricos do RFEM
        self.dados_rfem = {
        'Ux': np.array([
            [0.0, 0.0],
            [12.4e-3, 0.067],    # Deslocamento em metros (mm / 1000)
            [47.3e-3, 0.133],
            [99.4e-3, 0.200],
            [162.4e-3, 0.267],
            [230.6e-3, 0.333],
            [299.9e-3, 0.400],
            [367.9e-3, 0.467],
            [432.9e-3, 0.533],
            [494.4e-3, 0.600],
            [551.9e-3, 0.667],
            [605.5e-3, 0.733],
            [655.4e-3, 0.800],
            [701.7e-3, 0.867],
            [744.8e-3, 0.933],
            [784.8e-3, 1.000],
        ]),
        'Uy': np.array([
            [0.0, 0.0],
            [291.7e-3, 0.067],   # Deslocamento em metros (mm / 1000)
            [567.9e-3, 0.133],
            [817.4e-3, 0.200],
            [1035.4e-3, 0.267],
            [1221.9e-3, 0.333],
            [1379.8e-3, 0.400],
            [1513.1e-3, 0.467],
            [1625.7e-3, 0.533],
            [1721.3e-3, 0.600],
            [1803.0e-3, 0.667],
            [1873.2e-3, 0.733],
            [1933.9e-3, 0.800],
            [1986.8e-3, 0.867],
            [2033.1e-3, 0.933],
            [2074.0e-3, 1.000],
        ]),
        'Uz': np.array([
            [0.0, 0.0],
            [-12.4e-3, 0.067],   # Deslocamento em metros (mm / 1000)
            [-47.3e-3, 0.133],
            [-99.4e-3, 0.200],
            [-162.4e-3, 0.267],
            [-230.6e-3, 0.333],
            [-299.9e-3, 0.400],
            [-367.9e-3, 0.467],
            [-432.9e-3, 0.533],
            [-494.4e-3, 0.600],
            [-551.9e-3, 0.667],
            [-605.5e-3, 0.733],
            [-655.4e-3, 0.800],
            [-701.7e-3, 0.867],
            [-744.8e-3, 0.933],
            [-784.8e-3, 1.000],
        ]),
        'Rx': np.array([
            [0.0, 0.0],
            [-93.2e-3, 0.067],   # Rotação em radianos (mrad / 1000)
            [-182.4e-3, 0.133],
            [-264.5e-3, 0.200],
            [-338.2e-3, 0.267],
            [-403.1e-3, 0.333],
            [-459.8e-3, 0.400],
            [-509.3e-3, 0.467],
            [-552.4e-3, 0.533],
            [-590.1e-3, 0.600],
            [-623.3e-3, 0.667],
            [-652.5e-3, 0.733],
            [-678.5e-3, 0.800],
            [-701.6e-3, 0.867],
            [-722.2e-3, 0.933],
            [-740.8e-3, 1.000],
        ]),
        'Ry': np.array([
            [0.0, 0.0],
            [0.0, 0.067],       # Rotação em radianos (mrad / 1000)
            [0.0, 0.133],
            [0.0, 0.200],
            [0.0, 0.267],
            [0.0, 0.333],
            [0.0, 0.400],
            [0.0, 0.467],
            [0.0, 0.533],
            [0.0, 0.600],
            [0.0, 0.667],
            [0.0, 0.733],
            [0.0, 0.800],
            [0.0, 0.867],
            [0.0, 0.933],
            [0.0, 1.000],
        ]),
        'Rz': np.array([
            [0.0, 0.0],
            [-93.2e-3, 0.067],   # Rotação em radianos (mrad / 1000)
            [-182.4e-3, 0.133],
            [-264.5e-3, 0.200],
            [-338.2e-3, 0.267],
            [-403.1e-3, 0.333],
            [-459.8e-3, 0.400],
            [-509.3e-3, 0.467],
            [-552.4e-3, 0.533],
            [-590.1e-3, 0.600],
            [-623.3e-3, 0.667],
            [-652.5e-3, 0.733],
            [-678.5e-3, 0.800],
            [-701.6e-3, 0.867],
            [-722.2e-3, 0.933],
            [-740.8e-3, 1.000],
        ]),
        }
        
        # Extrair dados do histórico
        self.lambda_values = np.array([item[0] for item in f_vs_d_history])
        self.displacement_history = np.hstack([item[2] for item in f_vs_d_history]).T
        
        # Configurações dos eixos
        self.dof_names = ['|U|', 'Ux', 'Uy', 'Uz', 'Rx', 'Ry', 'Rz']
        self.dof_units = ['m', 'm', 'm', 'm', 'rad', 'rad', 'rad']
        self.dof_descriptions = [
            'Deslocamento Resultante',
            'Deslocamento em X',
            'Deslocamento em Y', 
            'Deslocamento em Z',
            'Rotação em torno de X',
            'Rotação em torno de Y',
            'Rotação em torno de Z'
        ]
        
        # Dados selecionados (padrão)
        self.selected_h_dof = 'Ux'  # Eixo horizontal
        self.selected_v_dof = None  # Eixo vertical (None = Load Factor)
        self.selected_h_nodes = [1]  # Nós selecionados para eixo horizontal
        self.selected_v_nodes = []   # Nós selecionados para eixo vertical
        
        self.setWindowTitle("Gráfico Carga x Deslocamento")
        self.resize(1400, 900)
        
        self.setup_ui()
        self.update_plot()
    
    def setup_ui(self):
        """Configura a interface do usuário."""
        layout = QVBoxLayout(self)
        
        # TabWidget principal
        self.tab_widget = QTabWidget()
        
        # Aba 1: Diagram (Seleção + Pré-visualização)
        self.diagram_tab = self.create_diagram_tab()
        self.tab_widget.addTab(self.diagram_tab, "Diagrama")
        
        # Aba 2: Table (Tabela de dados)
        self.table_tab = self.create_table_tab()
        self.tab_widget.addTab(self.table_tab, "Tabela")
        
        # Aba 3: Full Diagram (Gráfico isolado)
        self.full_diagram_tab = self.create_full_diagram_tab()
        self.tab_widget.addTab(self.full_diagram_tab, "Gráfico Completo")
        
        layout.addWidget(self.tab_widget)
        
        # Botões de ação
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        export_btn = QPushButton("Exportar Imagem...")
        export_btn.clicked.connect(self.export_image)
        
        close_btn = QPushButton("Fechar")
        close_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(export_btn)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)

        # Define o estado inicial e atualiza os gráficos
        self.h_result_combo.setCurrentIndex(1)
        self.v_result_combo.setCurrentIndex(0)
        self.on_h_result_changed()
        self.on_v_result_changed()
        
        # Conectar mudança de aba para atualizar gráfico completo
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
    
    def create_diagram_tab(self):
        """Cria a aba Diagram com seleção à esquerda e gráfico à direita."""
        widget = QWidget()
        main_layout = QHBoxLayout(widget)
        
        # ====================================================================
        # PAINEL ESQUERDO - SELEÇÃO DE DADOS
        # ====================================================================
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        
        # Tipo de Análise
        analysis_group = QGroupBox("Tipo de Análise")
        analysis_layout = QVBoxLayout()
        analysis_label = QLabel(self.analise)
        analysis_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #0066cc;")
        analysis_layout.addWidget(analysis_label)
        analysis_group.setLayout(analysis_layout)
        left_layout.addWidget(analysis_group)
        
        # --- Eixo Horizontal ---
        h_axis_group = QGroupBox("Eixo Horizontal")
        h_axis_layout = QVBoxLayout()
        
        # Tipo de resultado
        h_result_layout = QHBoxLayout()
        h_result_layout.addWidget(QLabel("Tipo de Resultado:"))
        self.h_result_combo = QComboBox()
        self.h_result_combo.addItem("Load factor (λ)")
        self.h_result_combo.addItem("Deformações Globais - Nós")
        self.h_result_combo.currentIndexChanged.connect(self.on_h_result_changed) # Conexão adicionada
        h_result_layout.addWidget(self.h_result_combo)
        h_axis_layout.addLayout(h_result_layout)
        
        # Dado (tipo de deslocamento)
        self.h_data_layout = QHBoxLayout() # Modificado para self.
        self.h_data_layout.addWidget(QLabel("Dado:"))
        self.h_data_combo = QComboBox()
        for i, (name, desc) in enumerate(zip(self.dof_names, self.dof_descriptions)):
            self.h_data_combo.addItem(f"{name} - {desc}", name)
        self.h_data_combo.setCurrentText("Ux - Deslocamento em X")
        self.h_data_combo.currentIndexChanged.connect(self.on_h_data_changed)
        self.h_data_layout.addWidget(self.h_data_combo, 1)
        h_axis_layout.addLayout(self.h_data_layout)
        
        # Seleção de nós
        self.h_nodes_label = QLabel("Nós:") # Modificado para self.
        h_axis_layout.addWidget(self.h_nodes_label)
        self.h_nodes_list = QListWidget()
        self.h_nodes_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for i in range(1, self.num_nodes + 1):
            self.h_nodes_list.addItem(f"Nó {i}")
        self.h_nodes_list.item(0).setSelected(True)  # Selecionar nó 1 por padrão
        self.h_nodes_list.itemSelectionChanged.connect(self.on_h_nodes_changed)
        self.h_nodes_list.setMaximumHeight(150)
        h_axis_layout.addWidget(self.h_nodes_list)
        
        h_axis_group.setLayout(h_axis_layout)
        left_layout.addWidget(h_axis_group)
        
        # --- Eixo Vertical ---
        v_axis_group = QGroupBox("Eixo Vertical")
        v_axis_layout = QVBoxLayout()
        
        # Tipo de resultado
        v_result_layout = QHBoxLayout()
        v_result_layout.addWidget(QLabel("Tipo de Resultado:"))
        self.v_result_combo = QComboBox()
        self.v_result_combo.addItem("Load Factor (λ)")
        self.v_result_combo.addItem("Deformações Globais - Nós")
        self.v_result_combo.currentIndexChanged.connect(self.on_v_result_changed)
        v_result_layout.addWidget(self.v_result_combo)
        v_axis_layout.addLayout(v_result_layout)
        
        # Dado (tipo de deslocamento) - inicialmente oculto
        self.v_data_layout = QHBoxLayout()
        self.v_data_layout.addWidget(QLabel("Dado:"))
        self.v_data_combo = QComboBox()
        for i, (name, desc) in enumerate(zip(self.dof_names, self.dof_descriptions)):
            self.v_data_combo.addItem(f"{name} - {desc}", name)
        self.v_data_combo.currentIndexChanged.connect(self.on_v_data_changed)
        self.v_data_layout.addWidget(self.v_data_combo, 1)
        v_axis_layout.addLayout(self.v_data_layout)
        
        # Seleção de nós - inicialmente oculto
        self.v_nodes_label = QLabel("Nós:")
        self.v_nodes_list = QListWidget()
        self.v_nodes_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for i in range(1, self.num_nodes + 1):
            self.v_nodes_list.addItem(f"Nó {i}")
        self.v_nodes_list.itemSelectionChanged.connect(self.on_v_nodes_changed)
        self.v_nodes_list.setMaximumHeight(150)
        
        v_axis_layout.addWidget(self.v_nodes_label)
        v_axis_layout.addWidget(self.v_nodes_list)
        
        v_axis_group.setLayout(v_axis_layout)
        left_layout.addWidget(v_axis_group)
        
        left_layout.addStretch()
        
        # ====================================================================
        # PAINEL DIREITO - PRÉ-VISUALIZAÇÃO DO GRÁFICO
        # ====================================================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Gráfico
        self.preview_plot = self.create_plot_widget()
        right_layout.addWidget(self.preview_plot)
        
        # ====================================================================
        # SPLITTER PARA AJUSTE DE TAMANHO
        # ====================================================================
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)  # Painel esquerdo
        splitter.setStretchFactor(1, 3)  # Painel direito (maior)
        
        main_layout.addWidget(splitter)
        
        return widget
    
    def create_table_tab(self):
        """Cria a aba Table com os dados tabelados."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Descrição
        desc_label = QLabel("Dados do Gráfico")
        desc_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(desc_label)
        
        # Tabela
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        layout.addWidget(self.data_table)
        
        return widget
    
    def create_full_diagram_tab(self):
        """Cria a aba Full Diagram com o gráfico isolado."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Gráfico completo
        self.full_plot = self.create_plot_widget()
        layout.addWidget(self.full_plot)
        
        return widget
    
    def create_plot_widget(self):
        """Cria um widget de gráfico Matplotlib configurado."""        
        return MplCanvas(self, width=5, height=4, dpi=100)
    
    def on_h_result_changed(self):
        """Callback quando o tipo de resultado do eixo horizontal muda."""
        is_displacement = self.h_result_combo.currentIndex() == 1
        
        # Mostrar/ocultar campos de deslocamento
        self.h_data_combo.setVisible(is_displacement)
        self.h_data_layout.itemAt(0).widget().setVisible(is_displacement)
        self.h_nodes_label.setVisible(is_displacement)
        self.h_nodes_list.setVisible(is_displacement)
        
        if is_displacement:
            # Garante que os valores sejam atualizados quando os campos se tornam visíveis
            self.selected_h_dof = self.h_data_combo.currentData()
            selected_items = self.h_nodes_list.selectedItems()
            self.selected_h_nodes = [int(item.text().split()[1]) for item in selected_items]
        else:
            self.selected_h_dof = None
            self.selected_h_nodes = []
        
        self.update_plot()
        self.update_table()

    def on_h_data_changed(self):
        """Callback quando o tipo de dado do eixo horizontal muda."""
        self.selected_h_dof = self.h_data_combo.currentData()
        self.update_plot()
        self.update_table()
    
    def on_h_nodes_changed(self):
        """Callback quando a seleção de nós do eixo horizontal muda."""
        selected_items = self.h_nodes_list.selectedItems()
        self.selected_h_nodes = [int(item.text().split()[1]) for item in selected_items]
        self.update_plot()
        self.update_table()
    
    def on_v_result_changed(self):
        """Callback quando o tipo de resultado do eixo vertical muda."""
        is_displacement = self.v_result_combo.currentIndex() == 1
        
        # Mostrar/ocultar campos de deslocamento
        self.v_data_combo.setVisible(is_displacement)
        self.v_data_layout.itemAt(0).widget().setVisible(is_displacement)
        self.v_nodes_label.setVisible(is_displacement)
        self.v_nodes_list.setVisible(is_displacement)
        
        if is_displacement:
            self.selected_v_dof = self.v_data_combo.currentData()
            selected_items = self.v_nodes_list.selectedItems()
            self.selected_v_nodes = [int(item.text().split()[1]) for item in selected_items]
        else:
            self.selected_v_dof = None
            self.selected_v_nodes = []
        
        self.update_plot()
        self.update_table()
    
    def on_v_data_changed(self):
        """Callback quando o tipo de dado do eixo vertical muda."""
        self.selected_v_dof = self.v_data_combo.currentData()
        self.update_plot()
        self.update_table()
    
    def on_v_nodes_changed(self):
        """Callback quando a seleção de nós do eixo vertical muda."""
        selected_items = self.v_nodes_list.selectedItems()
        self.selected_v_nodes = [int(item.text().split()[1]) for item in selected_items]
        self.update_plot()
        self.update_table()
    
    def _get_axis_data_and_labels(self, selected_dof, selected_nodes, axis_prefix):
        """
        Função auxiliar generalizada para obter os dados e rótulos de um eixo.
        Retorna uma lista de séries de dados e um rótulo principal para o eixo.
        """
        if selected_dof is None:  # Caso seja "Fator de Carga"
            label = "Load Factor"
            series = [{
                'metadata': self.lambda_values,
                'legend': label,
                'node': None # Sem nó específico para o fator de carga
            }]
            return series, label
        else:  # Caso seja "Deslocamento"
            if not selected_nodes:
                return [], "Nenhum nó selecionado"

            dof_idx = self.dof_names.index(selected_dof)
            desc = self.dof_names[dof_idx]
            unit = self.dof_units[dof_idx]
            main_label = f"{desc} ({unit})"
            
            series = []
            for node in selected_nodes:
                node_data = self.get_displacement_data(node, selected_dof)
                legend_label = f"{axis_prefix}: Node {node} ({selected_dof})"
                series.append({
                    'metadata': node_data,
                    'legend': legend_label,
                    'node': node
                })
            return series, main_label

    def get_displacement_data(self, node, dof_name):
        """
        Obtém os dados de deslocamento para um nó e grau de liberdade específicos.
        
        Args:
            node (int): Número do nó (1-indexed)
            dof_name (str): Nome do grau de liberdade ('ux', 'uy', etc.)
        
        Returns:
            np.ndarray: Array com os valores de deslocamento
        """
        if dof_name == '|U|':
            # Deslocamento resultante
            ux_idx = (node - 1) * 6 + 0
            uy_idx = (node - 1) * 6 + 1
            uz_idx = (node - 1) * 6 + 2
            
            ux = self.displacement_history[:, ux_idx]
            uy = self.displacement_history[:, uy_idx]
            uz = self.displacement_history[:, uz_idx]
            
            return np.sqrt(ux**2 + uy**2 + uz**2)
        else:
            # Deslocamento individual
            dof_index = self.dof_names.index(dof_name)
            gl_index = (node - 1) * 6 + dof_index - 1
            
            return self.displacement_history[:, gl_index]
    
    def update_plot(self):
        """Atualiza o gráfico de pré-visualização de forma generalizada."""
        self._draw_plot(self.preview_plot.axes, is_full_plot=False)

    def calcular_r2(self, x_sim, y_sim, x_ext, y_ext, tol=1e-9):
        """
        Calcula o R² entre os dados do SimuFrame e dados externos,
        usando interpolação para alinhar os pontos.
        """
        # Para evitar erros em dados com poucos pontos ou sem variação
        if len(x_sim) < 2 or len(x_ext) < 2:
            return np.nan
        
        # Verifica se os deslocamentos são todos nulos (ou quase nulos)
        is_x_programa_nulo = np.ptp(x_sim) < tol
        is_x_externo_nulo = np.ptp(x_ext) < tol

        # Ambos os deslocamentos são nulos
        if is_x_programa_nulo and is_x_externo_nulo:
            return 1.0

        # Um dos deslocamentos é nulo
        if is_x_programa_nulo or is_x_externo_nulo:
            return np.nan
        
        try:            
            # Garante que os pontos de interpolação estejam ordenados
            sort_indices = np.argsort(x_ext)
            x_externo_sorted = x_ext[sort_indices]
            y_externo_sorted = y_ext[sort_indices]
            
            # Remove duplicatas nodes pontos x
            unique_indices = np.unique(x_externo_sorted, return_index=True)[1]
            x_externo_unique = x_externo_sorted[unique_indices]
            y_externo_unique = y_externo_sorted[unique_indices]

            if len(x_externo_unique) < 2:
                # Não há pontos suficientes para interpolar após remover duplicatas
                return np.nan

            # Interpola os valores externos para os pontos de deslocamento do seu programa
            y_externo_interpolado = np.interp(x_sim, x_externo_unique, y_externo_unique)
            
            # Calcula o R²
            r2 = r2_score(y_sim, y_externo_interpolado)
            
            return r2
        
        except Exception:
            return np.nan # Retorna NaN se o cálculo falhar

    def _draw_plot(self, ax, is_full_plot=False):
        """
        Desenha o gráfico e a validação.
        """
        # Limpar cursor anterior se existir
        self._clear_cursor(ax)

        ax.clear()
        tol = 1e-6

        # Obter os valores das séries do programa
        x_series, x_label = self._get_axis_data_and_labels(self.selected_h_dof, self.selected_h_nodes, "H")
        y_series, y_label = self._get_axis_data_and_labels(self.selected_v_dof, self.selected_v_nodes, "V")

        if not x_series or not y_series:
            ax.figure.canvas.draw()
            return
        
        all_x_data, all_y_data = [], []
        color_idx = 0
        colors = matplotlib.pyplot.get_cmap('tab10').colors
        simuframe_curves = []

        # Armazena os dados das linhas para o cursor
        lines_data_for_cursor = []

        for x_item in x_series:
            for y_item in y_series:
                if x_item['metadata'] is y_item['metadata']:
                    continue

                legend = y_item['legend'] if x_item['node'] is None else x_item['legend']
                legend = f"SimuFrame - {legend.split(': ')[1]}"

                ax.plot(x_item['metadata'], y_item['metadata'], marker='o',
                        markersize=4, linestyle='-', label=legend, 
                        color=colors[color_idx % len(colors)])

                # Adicionar dados para o cursor
                lines_data_for_cursor.append({
                    'x': np.array(x_item['metadata']),
                    'y': np.array(y_item['metadata']),
                    'label': legend,
                    'color': colors[color_idx % len(colors)]
                })

                # Coleta os dados para controle de limites
                all_x_data.append(x_item['metadata'])
                all_y_data.append(y_item['metadata'])
                simuframe_curves.append({'x': x_item['metadata'], 'y': y_item['metadata']})
                color_idx += 1
        
        # Validar os dados
        is_h_disp = self.selected_h_dof is not None
        is_v_load = self.selected_v_dof is None
        is_h_load = self.selected_h_dof is None
        is_v_disp = self.selected_v_dof is not None

        # Condição: um eixo é deslocamento e o outro é fator de carga
        if (is_h_disp and is_v_load) or (is_h_load and is_v_disp):
            dof_key = self.selected_h_dof if is_h_disp else self.selected_v_dof
            
            # Plotar dados do Abaqus
            if dof_key in self.dados_abaqus:
                dados_ext = self.dados_abaqus[dof_key]
                u_ext, lambda_ext = dados_ext[:, 0], dados_ext[:, 1]
                
                # O sinal da rotação no Abaqus pode ser invertido
                if dof_key.startswith('R'):
                    u_ext = -u_ext
                
                # Calcula R² com a primeira curva do SimuFrame
                r2 = self.calcular_r2(simuframe_curves[0]['x'], simuframe_curves[0]['y'], u_ext, lambda_ext)
                label = f'Abaqus (R²={r2:.4f})' if not np.isnan(r2) else 'Abaqus'

                if is_h_disp:  # Gráfico é U vs Lambda
                    ax.plot(u_ext, lambda_ext, 'x--', color='red', 
                           label=label, linewidth=1.5, markersize=6)
                    
                else:  # Gráfico é Lambda vs U
                    ax.plot(lambda_ext, u_ext, 'x--', color='red', 
                           label=label, linewidth=1.5, markersize=6)
                    
            # Plotar dados do RFEM
            if dof_key in self.dados_rfem:
                dados_ext = self.dados_rfem[dof_key]
                u_ext, lambda_ext = dados_ext[:, 0], dados_ext[:, 1]
                
                r2 = self.calcular_r2(simuframe_curves[0]['x'], simuframe_curves[0]['y'], u_ext, lambda_ext)
                label = f'RFEM (R²={r2:.4f})' if not np.isnan(r2) else 'RFEM'

                if is_h_disp:  # Gráfico é U vs Lambda
                    ax.plot(u_ext, lambda_ext, '+:', color='green', 
                           label=label, linewidth=1.5, markersize=7)
                    
                else:  # Gráfico é Lambda vs U
                    ax.plot(lambda_ext, u_ext, '+:', color='green', 
                           label=label, linewidth=1.5, markersize=7)
                    
        # Controle de limite dos eixos
        # Verifica o eixo X
        if all_x_data:
            x_values = np.concatenate(all_x_data)
            if np.ptp(x_values) < tol:
                # Se a variação for desprezível, centraliza o gráfico em 0
                media_x = np.mean(x_values)
                ax.set_xlim(media_x - tol, media_x + tol)

        # Verifica o eixo Y
        if all_y_data:
            y_values = np.concatenate(all_y_data)
            if np.ptp(y_values) < tol:
                media_y = np.mean(y_values)
                ax.set_ylim(media_y - tol, media_y + tol)

        # Configuração final do gráfico
        ax.set_xlabel(x_label, fontsize=11, weight='bold')
        ax.set_ylabel(y_label, fontsize=11, weight='bold')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.margins(x=0.05, y=0.05)
        ax.figure.tight_layout()

        # Adicionar cursor interativo
        if lines_data_for_cursor:
            # Determinar precisão baseada nodes dados
            precision = self._determine_precision(all_x_data, all_y_data)
            
            cursor = InteractiveCursor(
                ax, 
                lines_data_for_cursor,
                x_label=x_label,
                y_label=y_label,
                precision=precision
            )
            
            # Armazenar referência ao cursor
            self.cursors.append((ax, cursor))

        ax.figure.canvas.draw()

    def _determine_precision(self, all_x_data, all_y_data):
        """
        Determina a precisão ideal para exibição baseada na magnitude dos dados.
        
        Returns:
            int: Número de dígitos significativos
        """
        if not all_x_data or not all_y_data:
            return 4
        
        # Concatenar todos os dados
        all_data = []
        for data in all_x_data + all_y_data:
            if len(data) > 0:
                all_data.extend(data)
        
        if not all_data:
            return 4
        
        # Encontrar ordem de magnitude
        max_val = np.max(np.abs(all_data))
        
        if max_val == 0:
            return 4
        
        # Ajustar precisão baseada na ordem de magnitude
        magnitude = np.floor(np.log10(max_val))
        
        if magnitude >= 3:  # Valores grandes (>= 1000)
            return 3
        elif magnitude >= 0:  # Valores normais (1-999)
            return 4
        elif magnitude >= -3:  # Valores pequenos (0.001-0.999)
            return 5
        else:  # Valores muito pequenos
            return 6

    def _clear_cursor(self, ax):
        """
        Remove cursor anterior associado a este axes.
        """
        # Procurar e desconectar cursor antigo
        for i, (stored_ax, cursor) in enumerate(self.cursors):
            if stored_ax is ax:
                cursor.disconnect()
                self.cursors.pop(i)
                break
    
    def closeEvent(self, event):
        """
        Sobrescrever closeEvent para limpar cursores ao fechar o diálogo.
        """
        # Desconectar todos os cursores
        for ax, cursor in self.cursors:
            cursor.disconnect()
        
        self.cursors.clear()
        
        # Chamar closeEvent pai
        super().closeEvent(event)

    def update_table(self):
        """Atualiza a tabela com os dados do gráfico de forma generalizada."""
        x_series, _ = self._get_axis_data_and_labels(self.selected_h_dof, self.selected_h_nodes, "H")
        y_series, _ = self._get_axis_data_and_labels(self.selected_v_dof, self.selected_v_nodes, "V")

        if not x_series or not y_series:
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(0)
            return

        headers = ["Incremento"]
        all_series = []

        # Coletar cabeçalhos e séries, evitando duplicatas (caso Lambda seja selecionado em ambos)
        temp_legends = set()
        for series in x_series + y_series:
            if series['legend'] not in temp_legends:
                headers.append(series['legend'])
                all_series.append(series['metadata'])
                temp_legends.add(series['legend'])
        
        num_rows = len(self.lambda_values)
        self.data_table.setRowCount(num_rows)
        self.data_table.setColumnCount(len(headers))
        self.data_table.setHorizontalHeaderLabels(headers)

        for row in range(num_rows):
            self.data_table.setItem(row, 0, QTableWidgetItem(str(row))) # Incremento
            for col, data_array in enumerate(all_series, start=1):
                value = data_array[row]
                item_text = f"{value:.6e}" if isinstance(value, np.floating) else f"{value:.6f}"
                self.data_table.setItem(row, col, QTableWidgetItem(item_text))

        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    
    def on_tab_changed(self, index):
        """Callback quando a aba é alterada."""
        if index == 2:  # Aba "Gráfico Completo"
            self.update_full_plot()
    
    def update_full_plot(self):
        """Atualiza o gráfico completo (cópia do preview)."""
        self._draw_plot(self.full_plot.axes, is_full_plot=True)
    
    def export_image(self):
        """Exporta o gráfico atual como imagem ou gráfico."""
        if not self.full_plot:
            return

        # Atualizar o gráfico para a seleção mais recente
        self.update_full_plot()

        # Abre o diálogo de "Salvar Arquivo"
        fileName, selected_filter = QFileDialog.getSaveFileName(self,
            "Salvar Gráfico",
            "", # Diretório inicial
            "SVG (*.svg);;PDF (*.pdf);;PNG (*.png);;JPG (*.jpg)"
        )

        if fileName:
            # Usa o método savefig da figura do Matplotlib
            self.full_plot.fig.savefig(fileName, bbox_inches='tight')


class AnalysisParametersDialog(QDialog):
    """
    Uma janela de diálogo para modificar os parâmetros básicos da análise,
    como tipo de análise e número de subdivisões.
    """
    def __init__(self, current_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Editar Parâmetros da Análise")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(400)

        # Widgets de edição
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems(['Linear', 'Não Linear', 'Flambagem'])
        self.analysis_combo.setCurrentText(current_params.get('analysis_type', 'Linear'))

        self.subdiv_spin = QSpinBox()
        self.subdiv_spin.setRange(1, 512)
        self.subdiv_spin.setValue(current_params.get('number_of_subdivisions', 1))

        # Botões OK/Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Layout
        form_layout = QFormLayout()
        form_layout.addRow("Tipo de Análise:", self.analysis_combo)
        form_layout.addRow("Nº de Subdivisões por Elemento:", self.subdiv_spin)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(form_layout)
        main_layout.addWidget(button_box)

        # Estilo
        self.setStyleSheet("""
            QDialog {
                background-color: #f8fafc;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
                color: #1e293b; /* Garante texto escuro no fundo claro */
            }

            QLabel#DialogTitle {
                font-size: 16px;
                font-weight: bold;
                color: #1e293b;
                margin-bottom: 10px;
            }

            QLabel {
                font-weight: 600;
                color: #475569; /* Cinza médio para labels */
            }

            QComboBox, QSpinBox {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 6px 10px;
                min-width: 150px;
                color: #1e293b; /* Texto escuro dentro dos inputs */
                selection-background-color: #3b82f6;
            }

            QComboBox:hover, QSpinBox:hover {
                border: 1px solid #94a3b8;
            }

            QComboBox:focus, QSpinBox:focus {
                border: 2px solid #3b82f6; /* Destaque azul ao focar */
                padding: 5px 9px; /* Ajuste para compensar a borda de 2px */
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 0px;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }

            /* Estilo dos Botões */
            QPushButton {
                font-weight: 600;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                min-width: 80px;
            }

            /* Botão OK (Primário) */
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

            /* Botão Cancel (Secundário) */
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

    def get_parameters(self):
        """Retorna os valores atuais dos widgets em um dicionário."""
        return {
            'analysis_type': self.analysis_combo.currentText(),
            'number_of_subdivisions': self.subdiv_spin.value()
        }
