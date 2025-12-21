# Built-in libraries
import time
import copy
from enum import Enum
from functools import lru_cache
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass, field

# Third-party libraries
import numpy as np
from tqdm import tqdm
import pyqtgraph as pg
import numpy.typing as npt
from scipy.linalg import eigh, qr
from scipy.sparse import csc_array, issparse
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from scipy.sparse.linalg import factorized, spsolve

# Local libraries
from .model import Structure
from SimuFrame.core.assembly import assemble_geometric_stiffness_matrix, shape_derivatives, shape_functions
from SimuFrame.utils.helpers import static_condensation, assemble_sparse_matrix, \
    atribuir_deslocamentos, check_convergence


class AnalysisType(Enum):
    NEWTON_RAPHSON = "Newton-Raphson (Load Control)"
    ARC_LENGTH = "Arc-Length (Kadapa)"

@dataclass
class NewtonRaphsonParams:
    num_passos_inicial: int = 50
    max_iter: int = 1000
    iter_ideal: int = 5
    fator_aumento: float = 1.20
    fator_reducao_lenta: float = 0.75
    fator_reducao_rapida: float = 0.5
    max_reducoes: int = 8
    abs_tol: float = 1e-8
    rel_tol: float = 1e-8
    close_delay_sec: float = 0.5

@dataclass
class ArcLengthParams:
    """Arc-length parameters."""
    lmbda0: float = 0.1
    max_iter: int = 25
    max_reducoes: int = 8
    close_delay_sec: float = 0.5
    allow_lambda_exceed: bool = False
    max_lambda: float = 2.0
    psi: float = 0.0
    abs_tol: float = 1e-8
    rel_tol: float = 1e-8

@dataclass
class IncrementState:
    """Estado de um incremento."""
    d: np.ndarray
    Fint: np.ndarray
    Kt: csc_array
    λ: float
    Δλ: float
    Ra: np.ndarray
    fe: np.ndarray
    de: np.ndarray

@dataclass
class ConvergenceData:
    """Dados de convergência da análise."""
    increments: List[Dict] = field(default_factory=list)
    iteration_details: List[Dict] = field(default_factory=list)
    lambda_history: List[float] = field(default_factory=lambda: [0.0])
    max_displ_history: List[float] = field(default_factory=lambda: [0.0])
    rejected_points: Dict[str, List] = field(default_factory=lambda: {'displ': [], 'lambda': []})
    total_increments: int = 0
    accepted_increments: int = 0
    rejected_increments: int = 0
    final_lambda: float = 0.0
    max_displacement: float = 0.0
    converged: bool = False

class SolverVisualizer:
    """
    Visualizador integrado para o método Arc-Length.
    Mostra o caminho de equilíbrio em tempo real com indicadores de status.
    """
    
    def __init__(self, analysis_type: AnalysisType, show_window: bool = True):
        """
        Args:
            show_window: Se False, não cria janela (modo headless)
        """
        self.show_window = show_window
        self.analysis_type = analysis_type
        
        if not show_window:
            self.app = None
            return
        
        # Criar aplicação Qt
        self.app = pg.mkQApp(f"Nonlinea Analysis - {analysis_type.value}")
        
        # Janela principal
        self.win = QtWidgets.QDialog()
        self.win.setWindowTitle(f"Equilibrium Path - {analysis_type.value}")
        self.win.resize(900, 700)
        
        # Layout principal
        main_layout = QtWidgets.QVBoxLayout(self.win)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Título
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
        
        # Widget do gráfico
        plot_widget: Any = pg.GraphicsLayoutWidget()
        self.plot = plot_widget.addPlot(title="Equilibrium Path")
        self.plot.setLabel('bottom', "Maximum displacement (m)", **{'font-size': '12pt'})
        self.plot.setLabel('left', "Load factor (λ)", **{'font-size': '12pt'})
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Curva principal (caminho aceito)
        self.curva_principal = self.plot.plot(
            pen=pg.mkPen(color='#3498db', width=3),
            symbol='o',
            symbolSize=8,
            symbolBrush='#3498db'
        )
        
        # Pontos rejeitados
        self.pontos_rejeitados = self.plot.plot(
            pen=None,
            symbol='x',
            symbolSize=12,
            symbolBrush='#e74c3c',
            symbolPen=pg.mkPen(color='#e74c3c', width=2)
        )
        
        # Linhas de referência
        self.linha_vertical = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen('#95a5a6', style=QtCore.Qt.PenStyle.DashLine, width=2)
        )
        
        self.linha_horizontal = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen('#95a5a6', style=QtCore.Qt.PenStyle.DashLine, width=2)
        )
        
        # Texto de informação
        self.texto_incremento = pg.TextItem(
            anchor=(0, 1),
            color='#2c3e50',
            fill=pg.mkBrush(255, 255, 255, 200),
            border=pg.mkPen('#2c3e50', width=2)
        )
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.texto_incremento.setFont(font)
        
        # Adicionar elements ao gráfico
        self.plot.addItem(self.linha_vertical)
        self.plot.addItem(self.linha_horizontal)
        self.plot.addItem(self.texto_incremento)
        
        plot_layout.addWidget(plot_widget)
        
        # Barra de status vertical
        self.status_bar = self._create_status_bar()
        plot_layout.addWidget(self.status_bar)
        
        main_layout.addLayout(plot_layout)
        
        # Painel de informações
        info_panel = self._create_info_panel()
        main_layout.addWidget(info_panel)
        
        # Mostrar janela
        if self.app:
            self.win.show()
            self.app.processEvents()
    
    def _create_status_bar(self) -> QtWidgets.QWidget:
        """Cria a barra de status vertical colorida."""
        status_bar = QtWidgets.QWidget()
        status_bar.setAutoFillBackground(True)
        status_bar.setFixedWidth(30)
        status_bar.setMinimumHeight(400)
        
        # Tooltip
        status_bar.setToolTip("Green: Converging\nOrange: Rejected\nRed: Waiting")
        
        self.set_status_color(status_bar, '#e74c3c')  # Vermelho inicial
        return status_bar
    
    @staticmethod
    def set_status_color(status_bar: QtWidgets.QWidget, color: str):
        """Define a cor da barra de status."""
        palette = status_bar.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(color))
        status_bar.setPalette(palette)
    
    def _create_info_panel(self) -> QtWidgets.QWidget:
        """Cria painel de informações em tempo real."""
        panel = QtWidgets.QGroupBox("Informações do Incremento")
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
            ("Status:", "status")
        ]

        newton_raphson_items = [
            ("Step:", "step"),
            ("Iterations:", "iterations"),
            ("Load Factor (λ):", "lambda"),
            ("||R||:", "residue"),
            ("Status:", "status")
        ]

        # Select items based on analysis type
        items = arc_length_items if self.analysis_type == AnalysisType.ARC_LENGTH else newton_raphson_items
        
        for row, (label_text, key) in enumerate(items):
            label = QtWidgets.QLabel(label_text)
            label.setStyleSheet("font-weight: bold; font-size: 12px;")
            
            value = QtWidgets.QLabel("--")
            value.setStyleSheet("font-size: 12px; color: #34495e;")
            
            layout.addWidget(label, row, 0)
            layout.addWidget(value, row, 1)
            
            self.info_labels[key] = value
        
        return panel
    
    def update(self, 
               max_displ_history: List[float], 
               lambda_history: List[float],
               step: int, 
               iteracao: int, 
               λ: float, 
               max_displ: float,
               delta_s: float = 0.0, 
               residuo: float = 0.0
               ) -> None:
        """
        Atualiza a visualização com novos dados.
        
        Args:
            max_displ_history: Histórico de deslocamentos máximos aceitos
            lambda_history: Histórico de lambdas aceitos
            incremento: Número do incremento atual
            iteracao: Número de iterações do incremento
            λ: Lambda atual
            max_displ: Deslocamento máximo atual
            delta_s: Comprimento de arco atual
            residuo: Norma do resíduo
        """
        if not self.show_window:
            return
        
        # Atualizar curva principal
        self.curva_principal.setData(max_displ_history, lambda_history)
        
        # Atualizar linhas de referência
        self.linha_vertical.setValue(max_displ)
        self.linha_horizontal.setValue(float(λ))
        
        # Atualizar texto no gráfico
        if self.analysis_type == AnalysisType.NEWTON_RAPHSON:
            texto = f"Step: {step}\nIter: {iteracao}\nλ = {λ:.6f}"
        else:
            texto = f"Step: {step}\nIter: {iteracao}\nλ = {λ:.6f}\nΔs = {delta_s:.6e}"
        self.texto_incremento.setText(texto)
        
        # Posicionar texto
        x_pos = max_displ * 0.8
        y_pos = float(λ) * 0.8
        self.texto_incremento.setPos(x_pos, y_pos)
        
        # Atualizar painel de informações
        self.info_labels['step'].setText(f"{step}")
        self.info_labels['iterations'].setText(f"{iteracao}")
        self.info_labels['lambda'].setText(f"{λ:.6f}")
        if self.analysis_type == AnalysisType.ARC_LENGTH:
            self.info_labels['delta_s'].setText(f"{delta_s:.6e}")
        self.info_labels['residue'].setText(f"{residuo:.6e}")
        self.info_labels['status'].setText("✓ Convergindo")
        self.info_labels['status'].setStyleSheet("font-size: 12px; color: #27ae60; font-weight: bold;")
        
        # Change status to converging (green)
        self.set_status_color(self.status_bar, '#27ae60')
        
        # Process events
        if self.app:
            self.app.processEvents()
    
    def show_failure(self, 
                     max_displ: float = 0.0, 
                     λ: float = 0.0, 
                     rejected_displ: List[float] = [], 
                     rejected_lambda: List[float] = []):
        """
        Mostra falha no incremento.
        
        Args:
            max_displ: Deslocamento do ponto rejeitado
            λ: Lambda do ponto rejeitado
            rejected_displ: Lista completa de deslocamentos rejeitados
            rejected_lambda: Lista completa de lambdas rejeitados
        """
        if not self.show_window:
            return
        
        # Adicionar ponto rejeitado se fornecido
        if max_displ > 0 and λ > 0:
            if not rejected_displ:
                rejected_displ = [max_displ]
                rejected_lambda = [λ]
            else:
                rejected_displ.append(max_displ)
                rejected_lambda.append(λ)
        
        # Atualizar pontos rejeitados
        if rejected_displ and rejected_lambda:
            self.pontos_rejeitados.setData(rejected_displ, rejected_lambda)
        
        # Status laranja (rejeitado)
        self.set_status_color(self.status_bar, '#e67e22')
        
        # Atualizar painel
        self.info_labels['status'].setText("✗ Rejeitado")
        self.info_labels['status'].setStyleSheet("font-size: 12px; color: #e74c3c; font-weight: bold;")
        
        if self.app:
            self.app.processEvents()
    
    def finalize(self, converged: bool, final_lambda: float, total_increments: int,
                accepted_increments: int):
        """
        Finaliza a visualização com estatísticas finais.
        
        Args:
            converged: Se a análise convergiu
            final_lambda: Lambda final
            total_increments: Total de incrementos tentados
            accepted_increments: Incrementos aceitos
        """
        if not self.show_window:
            return
        
        # Cor final da barra
        if converged:
            self.set_status_color(self.status_bar, '#27ae60')  # Verde
            status_text = "✓ Concluído"
            status_color = "#27ae60"
        else:
            self.set_status_color(self.status_bar, '#e74c3c')  # Vermelho
            status_text = "✗ Falhou"
            status_color = "#e74c3c"
        
        # Atualizar status
        self.info_labels['status'].setText(status_text)
        self.info_labels['status'].setStyleSheet(f"font-size: 12px; color: {status_color}; font-weight: bold;")
        
        # Adicionar texto final no gráfico
        texto_final = (f"ANALYSIS COMPLETED\n"
                      f"λ_final = {final_lambda:.6f}\n"
                      f"Steps: {accepted_increments}/{total_increments}")
        
        self.texto_incremento.setText(texto_final)
        
        if self.app:
            self.app.processEvents()
    
    def wait_and_close(self, delay_sec: float):
        """
        Aguarda por delay_sec segundos e fecha a janela.
        
        Args:
            delay_sec: Tempo de espera em segundos
        """
        if not self.app:
            return
        
        end_time = time.time() + delay_sec
        while time.time() < end_time:
            self.app.processEvents()
            time.sleep(0.01)
        
        self.win.close()


class ArcLengthSolver:
    """Solver do método Arc-Length."""

    def __init__(self,
                 F: npt.NDArray[np.float64], 
                 KE: csc_array, 
                 estrutura: Structure, 
                 propriedades: Dict[str, Any],
                 numDOF: int, 
                 GLL: npt.NDArray[np.bool_], 
                 GLe: npt.NDArray[np.integer], 
                 T: npt.NDArray[np.float64],
                 NLG: bool, 
                 params: ArcLengthParams
                 ):
        self.F = F
        self.estrutura = estrutura
        self.propriedades = propriedades
        self.numDOF = numDOF
        self.GLL = GLL
        self.GLe = GLe
        self.T = T
        self.NLG = NLG
        self.params = params

        # Histórico para extrapolação (n-1 e n)
        self.d_n = np.zeros_like(F)      # d no passo n (último convergido)
        self.d_n_1 = np.zeros_like(F)    # d no passo n-1 (penúltimo convergido)
        self.λ_n = 0.0                   # λ no passo n
        self.λ_n_1 = 0.0                 # λ no passo n-1

        # FF = F_ext^T * F_ext usado nas fórmulas (escalares)
        self.F_ext_vec = self.F.copy()
        self.FF = float(np.dot(self.F.T, self.F)[0, 0])

        # Estado inicial
        self.state = IncrementState(
            λ=0.0,
            Δλ=0.0,
            Kt=KE.copy(),
            d=np.zeros_like(F),
            Fint=np.zeros_like(F),
            Ra=np.zeros((0, 0)),
            fe=np.zeros((0, 0)),
            de=np.zeros((0, 0))
        )

        # Históricos
        self.f_vs_d = [(0.0, np.zeros((numDOF, 1)), np.zeros((numDOF, 1)))]
        self.convergence_data = ConvergenceData()

        # Visualizador (apenas se NLG=True)
        self.visualizer = SolverVisualizer(AnalysisType.ARC_LENGTH, show_window=NLG)

        # Flags de controle
        self.counter = 0
        self.converged = True
        self.converged_prev = True

    def _save_state(self) -> IncrementState:
        """Saves the current state."""
        return copy.deepcopy(self.state)

    def _restore_state(self, backup: IncrementState):
        """Restore the last converged state."""
        self.state = copy.deepcopy(backup)

    def _initial_step(self) -> bool:
        """Initial step with Newton-Raphson (force control)."""
        # Definir os parâmetros de controle de passos
        self.state.λ = getattr(self.params, 'lmbda0', 1/100)

        iteracao = 0
        norm0 = 1.0

        while iteracao < self.params.max_iter:
            # Calcular o vetor de forças residuais
            R = self.state.λ * self.F - self.state.Fint
            norm = np.linalg.norm(R)

            if iteracao == 0:
                norm0 = norm if norm > 1e-14 else 1.0

            # Norma relativa
            rel_norm = norm / norm0

            print(f'  Iteração {iteracao+1}: |R| = {norm:.4e}, |R|/|R0| = {rel_norm:.4e}')

            # Verificar convergência
            if norm < self.params.abs_tol or rel_norm < self.params.rel_tol:
                # Calcular Δs baseado na solução convergida
                self.Δs = np.sqrt(
                    np.dot(self.state.d.T, self.state.d)[0, 0] +
                    self.params.psi * self.state.λ ** 2 * self.FF
                )

                # Definir limites com base na solução inicial
                self.Δs_n = self.Δs
                self.Δs_max = self.Δs
                self.Δs_min = self.Δs / 1024.0

                # Atualizar o contador
                self.counter = 1

                return True

            # Resolver o incremento de deslocamento Δd
            Δd = spsolve(self.state.Kt, R).reshape(-1, 1)

            # Atualizar deslocamentos
            self.state.d += Δd

            # Atualizar a matriz de rigidez tangente e o vetor de forças internas
            self.state.Kt, self.state.Fint, self.state.Ra, self.state.fe, self.state.de = analise_global(
                self.state.d, self.estrutura, self.propriedades,
                self.numDOF, self.GLL, self.GLe, self.T, self.NLG
            )

            iteracao += 1

        return False

    def _compute_predictor(self) -> Tuple[np.ndarray, float]:
        """Calcula o passo preditor."""
        if self.counter == 1:
            # Primeiro passo arc-length após inicialização
            self.d_n = np.zeros_like(self.state.d)
            self.d_n_1 = np.zeros_like(self.state.d)
            self.λ_n = 0.0
            self.λ_n_1 = 0.0

        else:
            # Extrapolação linear: u_pred = (1+α)*u_n - α*u_{n-1}
            alpha = self.Δs / self.Δs_n
            d_pred = (1 + alpha) * self.d_n - alpha * self.d_n_1
            λ_pred = (1 + alpha) * self.λ_n - alpha * self.λ_n_1

            # Aplicar predição
            self.state.d = d_pred.copy()
            self.state.λ = λ_pred

        # Incrementos desde o último passo convergido
        Δd = self.state.d - self.d_n
        Δλ = self.state.λ - self.λ_n

        return Δd, Δλ


    def _compute_corrector(self, Δd: np.ndarray, Δλ: float,
                           du_1: np.ndarray, du_2: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calcula a correção usando a restrição Arc-Length."""
        # Termos da fórmula
        a = 2.0 * Δd
        b = 2.0 * self.params.psi * Δλ * self.FF
        A = np.dot(Δd.T, Δd)[0, 0] + self.params.psi * Δλ ** 2 * self.FF - self.Δs ** 2

        # Produtos internos
        a_dot_du1 = np.dot(a.T, du_1)[0, 0]
        a_dot_du2 = np.dot(a.T, du_2)[0, 0]

        # Calcular dlmbda
        dlmbda = (a_dot_du2 - A) / (b + a_dot_du1)

        # Calcular du
        du = -du_2 + dlmbda * du_1

        return dlmbda, du

    def _arc_length_iteration(self, Δd: np.ndarray, Δλ: float) -> Tuple[bool, int, float]:
        """Executa iterações Arc-Length com corretor."""
        converged = False
        iter = 0
        norm = 1.0
        norm0 = None

        while iter < self.params.max_iter:
            # Atualizar estado
            self.state.d = self.d_n + Δd
            self.state.λ = self.λ_n + Δλ

            # Recalcular rigidez e forças internas
            self.state.Kt, self.state.Fint, self.state.Ra, self.state.fe, self.state.de = analise_global(
                self.state.d, self.estrutura, self.propriedades,
                self.numDOF, self.GLL, self.GLe, self.T, self.NLG
            )

            # Resíduo: R = λ*F - Fint
            R = self.state.Fint - self.state.λ * self.F
            norm_R = np.linalg.norm(R)

            # Restrição Arc-Length
            A = np.dot(Δd.T, Δd)[0, 0] + self.params.psi * (Δλ ** 2) * self.FF - self.Δs ** 2

            # Norma total combinada
            norm = np.sqrt(norm_R ** 2 + A ** 2)

            # Define relative residual for arc-length solver iteration
            if iter == 0:
                norm0 = norm if norm > 1e-14 else 1.0
            rel_norm = norm / norm0

            print(f'    Iter {iter}: |Total| = {norm:.4e}, |R| = {norm_R:.4e}, '
                  f'|A| = {abs(A):.4e}, Rel = {rel_norm:.4e}')

            # Verificar convergência
            if norm < self.params.abs_tol or rel_norm < self.params.rel_tol:
                converged = True
                break

            # Verificar divergência
            if norm > 1e10 or np.isnan(norm):
                print('    Divergência detectada!')
                return False, iter, float(norm)

            # Resolver sistemas lineares
            try:
                du_1 = spsolve(self.state.Kt, self.F_ext_vec).reshape(-1, 1)
                du_2 = spsolve(self.state.Kt, R).reshape(-1, 1)
            except (RuntimeError, np.linalg.LinAlgError):
                print('    Erro ao resolver sistema linear')
                return False, iter, float(norm)

            # Calcular correção
            try:
                dlmbda, du = self._compute_corrector(Δd, Δλ, du_1, du_2)
            except ValueError as e:
                print(f'    Erro no corretor: {e}')
                return False, iter, float(norm)

            # Atualizar incrementos totais
            Δd += du
            Δλ += dlmbda

            # Atualizar estado
            # self.state.d += du
            # self.state.λ += dlmbda

            # Recalcular rigidez e forças internas
            # self.state.Kt, self.state.Fint, self.state.Ra, self.state.fe, self.state.de = analise_global(
            #     self.state.d, self.estrutura, self.propriedades,
            #     self.numDOF, self.GLL, self.GLe, self.T, self.NLG
            # )

            # Armazenar dados da iteração
            self.convergence_data.iteration_details.append({
                'iteration': iter,
                'lambda': float(self.state.λ),
                'delta_lambda': float(Δλ),
                'delta_lambda_iter': float(dlmbda),
                'norm_R': float(norm_R),
                'norm_A': float(abs(A)),
                'norm': float(norm),
                'norm_delta_d': float(np.linalg.norm(du)),
                'converged': False
            })

            iter += 1
        
        if converged:
            self.state.d = self.d_n + Δd
            self.state.λ = self.λ_n + Δλ
            
            # Recalcular uma última vez com o estado convergido
            self.state.Kt, self.state.Fint, self.state.Ra, self.state.fe, self.state.de = analise_global(
                self.state.d, self.estrutura, self.propriedades,
                self.numDOF, self.GLL, self.GLe, self.T, self.NLG
            )

        # Marcar última iteração como convergida se for o caso
        if converged and self.convergence_data.iteration_details:
            self.convergence_data.iteration_details[-1]['converged'] = True

        return converged, iter, float(norm)

    def _adapt_arc_length(self):
        """Adapta o comprimento de arco baseado na convergência."""
        if self.converged:
            if self.converged_prev:
                # Dobrar Δs se convergiu e anterior também
                novo_Δs = min(2.0 * self.Δs, self.Δs_max)
                print(f'  Adaptação Δs: {self.Δs:.6e} → {novo_Δs:.6e} (dobrar, ambos convergiram)')
                self.Δs = novo_Δs
        else:
            if self.converged_prev:
                # Dividir por 2 se falhou, mas anterior convergiu
                novo_Δs = max(self.Δs / 2.0, self.Δs_min)
                print(f'  Adaptação Δs: {self.Δs:.6e} → {novo_Δs:.6e} (÷2, anterior convergiu)')
                self.Δs = novo_Δs
            else:
                # Dividir por 4 se falhou e anterior também
                novo_Δs = max(self.Δs / 4.0, self.Δs_min)
                print(f'  Adaptação Δs: {self.Δs:.6e} → {novo_Δs:.6e} (÷4, ambos falharam)')
                self.Δs = novo_Δs

    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List, np.ndarray, ConvergenceData]:
        """Executa o método Arc-Length."""
        # Passo inicial: Newton-Raphson puro (counter vai de 0 → 1)
        if not self._initial_step():
            print("\nFALHA no passo inicial!")
            return self.state.d, self.state.de, self.state.fe, self.f_vs_d, self.state.Ra, self.convergence_data

        # Registrar passo inicial
        force = np.zeros((self.numDOF, 1))
        force[self.GLL] = self.state.λ * self.F
        displ = np.zeros((self.numDOF, 1))
        displ[self.GLL] = self.state.d
        self.f_vs_d.append((float(self.state.λ), force, displ))

        max_displ = np.max(np.abs(self.state.d))
        self.convergence_data.lambda_history.append(self.state.λ)
        self.convergence_data.max_displ_history.append(max_displ)
        self.convergence_data.accepted_increments += 1

        success = False
        falhas_consecutivas = 0

        # Loop principal Arc-Length
        # while self.counter < self.params.num_incrementos_max:
        while True:
            print(f'\n{"=" * 70}')
            print(f'Passo Arc-Length {self.counter}, Δs = {self.Δs:.6e}')
            print(f'{"=" * 70}')

            # Salvar estado antes de fazer qualquer coisa
            backup = self._save_state()
            backup_Δs = self.Δs

            # Passo preditor (com ou sem extrapolação dependendo do counter)
            Δd, Δλ = self._compute_predictor()

            # Recalcular forças internas
            self.state.Kt, self.state.Fint, *_ = analise_global(
                self.state.d, self.estrutura, self.propriedades,
                self.numDOF, self.GLL, self.GLe, self.T, self.NLG
            )

            # Atualizar flag de convergência anterior
            self.converged_prev = self.converged
            self.converged = False

            # Iterações corretoras Arc-Length
            converged, iteracao, norm_total = self._arc_length_iteration(Δd, Δλ)

            # Sucesso - aceitar incremento
            if converged:
                self.converged = True
                self.convergence_data.accepted_increments += 1
                falhas_consecutivas = 0

                print(f'\n  ✓ CONVERGIU em {iteracao} iterações')
                print(f'    λ = {self.state.λ:.6f}')

                # Salvar Δs usado neste passo antes de adaptá-lo
                self.Δs_n = self.Δs

                # Atualizar histórico: n-1 ← n, n ← atual
                self.d_n_1 = self.d_n.copy()
                self.λ_n_1 = self.λ_n
                self.d_n = self.state.d.copy()
                self.λ_n = self.state.λ

                # Salvar para gráfico
                force = np.zeros((self.numDOF, 1))
                force[self.GLL] = self.state.λ * self.F
                displ = np.zeros((self.numDOF, 1))
                displ[self.GLL] = self.state.d
                self.f_vs_d.append((float(self.state.λ), force, displ))

                max_displ = np.max(np.abs(self.state.d))
                self.convergence_data.lambda_history.append(self.state.λ)
                self.convergence_data.max_displ_history.append(max_displ)

                # Atualizar visualização
                self.visualizer.update(
                    self.convergence_data.max_displ_history,
                    self.convergence_data.lambda_history,
                    self.counter, iteracao, self.state.λ, max_displ,
                    self.Δs, norm_total
                )

                # Verificar critérios de parada
                if self.params.allow_lambda_exceed:
                    if self.state.λ >= self.params.max_lambda:
                        success = True
                        print(f'\n✓ Atingiu λ_max = {self.params.max_lambda}')
                        break
                else:
                    if self.state.λ >= 0.999:
                        success = True
                        print('\n✓ Atingiu λ ≈ 1.0')
                        break

                # Adaptar Δs para o próximo passo
                self._adapt_arc_length()

                # Incrementar counter para próximo passo
                self.counter += 1

            # Falha - restaurar estado e reduzir Δs
            else:
                print(f'\n  ✗ FALHOU após {iteracao} iterações')
                self._restore_state(backup)
                self.Δs = backup_Δs
                self.converged = False
                falhas_consecutivas += 1

                if self.visualizer:
                    self.visualizer.show_failure()

                # Adaptar Δs para tentar novamente
                self._adapt_arc_length()

                # Verificar se Δs ficou muito pequeno
                if self.Δs <= self.Δs_min or falhas_consecutivas > self.params.max_reducoes:
                    print('\n✗ Impossível continuar:')
                    print(f'  Δs = {self.Δs:.6e} (mínimo = {self.Δs_min:.6e})')
                    print(f'  Falhas consecutivas = {falhas_consecutivas}')
                    break

        # Finalizar visualizador
        self.visualizer.finalize(
            success,
            self.state.λ,
            self.convergence_data.total_increments,
            self.convergence_data.accepted_increments
        )

        if self.visualizer.show_window:
            self.visualizer.wait_and_close(self.params.close_delay_sec)

        return self.state.d, self.state.de, self.state.fe, self.f_vs_d, self.state.Ra, self.convergence_data

class NewtonRaphsonSolver:
    def __init__(self,
                 F: npt.NDArray[np.float64], 
                 KE: csc_array, 
                 estrutura: Structure, 
                 propriedades: Dict[str, Any],
                 numDOF: int, 
                 GLL: npt.NDArray[np.bool_], 
                 GLe: npt.NDArray[np.integer], 
                 T: npt.NDArray[np.float64],
                 NLG: bool, 
                 params: NewtonRaphsonParams
                 ):
        self.F = F
        self.estrutura = estrutura
        self.propriedades = propriedades
        self.numDOF = numDOF
        self.GLL = GLL
        self.GLe = GLe
        self.T = T
        self.NLG = NLG
        self.params = params

        # Initial state
        self.state = IncrementState(
            λ=0.0,
            Δλ=0.0,
            Kt=KE.copy(),
            d=np.zeros_like(F),
            Fint=np.zeros_like(F),
            Ra=np.zeros((0, 0)),
            fe=np.zeros((0, 0)),
            de=np.zeros((0, 0))
        )

        # Load control parameters
        self.Δλ: int | float = 1.0 / params.num_passos_inicial
        self.Δλ_min: int | float = self.Δλ / (params.fator_aumento**params.max_reducoes)

        # Output history
        self.f_vs_d = [(0.0, np.zeros((numDOF, 1)), np.zeros((numDOF, 1)))]
        self.convergence_data = ConvergenceData()

        # Visualizer (only if NLG=True)
        self.visualizer = SolverVisualizer(AnalysisType.NEWTON_RAPHSON, show_window=NLG)

    def _save_state(self) -> IncrementState:
        """Saves the current state."""
        # return IncrementState(
        #     d=self.state.d.copy(),
        #     Fint=self.state.Fint.copy(),
        #     Kt=self.state.Kt.copy(),
        #     Ra=self.state.Ra.copy(),
        #     fe=self.state.fe.copy(),
        #     de=self.state.de.copy(),
        #     λ=self.state.λ
        # )
        return copy.deepcopy(self.state)

    def _restore_state(self, backup: IncrementState):
        """Restores a saved state."""
        # self.state = IncrementState(
        #     d=backup.d.copy(),
        #     Fint=backup.Fint.copy(),
        #     Kt=backup.Kt.copy(),
        #     Ra=backup.Ra.copy(),
        #     fe=backup.fe.copy(),
        #     de=backup.de.copy(),
        #     λ=backup.λ
        # )
        self.state: IncrementState = copy.deepcopy(backup)
        
    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List, np.ndarray, ConvergenceData]:
        step = 0
        success = False
        consecutive_failures = 0

        # Load control loop
        while self.state.λ < 1.0:
            step += 1
            self.convergence_data.total_increments += 1

            print(f'\n{"=" * 70}')
            print(f'Passo de Carga {step}, Δλ = {self.Δλ:.6e}')
            print(f'{"=" * 70}')

            # Save state before step
            backup = self._save_state()
            # backup_Δλ = self.Δλ

            # Limit load factor to not exceed 1.0
            lambda_target = min(self.state.λ + self.Δλ, 1.0)

            # Update load factor
            self.state.λ = lambda_target

            iter = 0
            norm = 1.0
            norm0 = 1.0
            converged = False

            # Newton-Raphson iterations
            while iter < self.params.max_iter:
                # Residual forces: : R = λF - Fint
                R = self.state.λ * self.F - self.state.Fint
                norm = np.linalg.norm(R)

                # Check divergence
                if norm > 1e10 or np.isnan(norm):
                    converged = False
                    break

                if iter == 0:
                    norm0 = norm if norm > 1e-14 else 1.0

                rel_norm = norm / norm0

                print(f'  Iteração {iter+1}: |R| = {norm:.4e}, |R|/|R0| = {rel_norm:.4e}')

                # # Check convergence
                # if norm < self.params.abs_tol or rel_norm < self.params.rel_tol:
                #     converged = True
                #     break

                # Solve for displacement increment Δd
                Δd = spsolve(self.state.Kt, R).reshape(-1, 1)

                # Update displacements
                self.state.d += Δd

                # Check convergence based on displacement increment
                converged = check_convergence(self.state.d, Δd, self.state.λ * self.F, R)

                # Exit if converged
                if converged:
                    break

                # Update tangent stiffness and internal forces
                self.state.Kt, self.state.Fint, self.state.Ra, self.state.fe, self.state.de = analise_global(
                    self.state.d, self.estrutura, self.propriedades,
                    self.numDOF, self.GLL, self.GLe, self.T, self.NLG
                )

                iter += 1
            
            # Success - accept increment
            if converged:
                self.converged = True
                self.convergence_data.accepted_increments += 1
                consecutive_failures = 0

                print(f'\n  ✓ CONVERGIU em {iter} iterações')
                print(f'    λ = {self.state.λ:.6f}')

                # Salvar para gráfico
                force = np.zeros((self.numDOF, 1), dtype=float)
                force[self.GLL] = self.state.λ * self.F
                displ = np.zeros((self.numDOF, 1), dtype=float)
                displ[self.GLL] = self.state.d
                self.f_vs_d.append((float(self.state.λ), force, displ))

                max_displ = np.max(np.abs(self.state.d))
                self.convergence_data.lambda_history.append(self.state.λ)
                self.convergence_data.max_displ_history.append(max_displ)

                # Update visualizer
                self.visualizer.update(
                    self.convergence_data.max_displ_history,
                    self.convergence_data.lambda_history,
                    step, iter, self.state.λ, max_displ,
                    residuo=float(norm)
                )

                # Step size adaptation
                if iter <= 3:
                    factor = 1.5

                elif iter <= self.params.iter_ideal:
                    factor = self.params.fator_aumento

                elif iter <= self.params.iter_ideal * 1.5:
                    factor = 1.0

                else:
                    factor = self.params.fator_reducao_lenta

                # Adapt step size
                new_Δλ = np.clip(factor * self.Δλ, self.Δλ_min, 1.0)
                print(f'  Adaptação Δλ: {self.Δλ:.6e} → {new_Δλ:.6e}')
                self.Δλ = new_Δλ

                # Check stopping criteria
                if self.state.λ >= 0.999:
                    success = True
                    print('\n✓ Atingiu λ ≈ 1.0')
                    break

            # Failure - restore state and reduce Δλ
            else:
                print(f'\n  ✗ FALHOU após {iter} iterações')
                self._restore_state(backup)
                # self.Δλ = backup_Δλ
                self.Δλ *= self.params.fator_reducao_rapida
                self.converged = False
                consecutive_failures += 1

                if self.visualizer:
                    self.visualizer.show_failure()

                # Decrease step size
                step -= 1

                # Verificar se Δs ficou muito pequeno
                if self.Δλ <= self.Δλ_min or consecutive_failures > self.params.max_reducoes:
                    print('\n✗ Impossível continuar:')
                    print(f'  Δλ = {self.Δλ:.6e} (mínimo = {self.Δλ_min:.6e})')
                    print(f'  Falhas consecutivas = {consecutive_failures}')
                    break
    
        # End visualizer
        self.visualizer.finalize(success, 
                                    self.state.λ, 
                                    self.convergence_data.total_increments, 
                                    self.convergence_data.accepted_increments
                                )
        self.visualizer.wait_and_close(self.params.close_delay_sec)
    
        return self.state.d, self.state.de, self.state.fe, self.f_vs_d, self.state.Ra, self.convergence_data


def run_nonlinear_solver(analysis_type: AnalysisType,
                         F: npt.NDArray[np.float64], 
                         KE: csc_array,
                         estrutura: Structure,
                         propriedades: Dict[str, Any],
                         numDOF: int,
                         GLL: npt.NDArray[np.bool_],
                         GLe: npt.NDArray[np.integer],
                         T: npt.NDArray[np.float64],
                         NLG: bool,
                         **solver_kwargs
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List, np.ndarray, ConvergenceData]:
    """Run nonlinear solver based on the specified analysis type."""
    if analysis_type == AnalysisType.ARC_LENGTH:
        params = ArcLengthParams(**solver_kwargs)
        solver = ArcLengthSolver(F, KE, estrutura, propriedades, numDOF, GLL, GLe, T, NLG, params)
    
    elif analysis_type == AnalysisType.NEWTON_RAPHSON:
        params = NewtonRaphsonParams(**solver_kwargs)
        solver = NewtonRaphsonSolver(F, KE, estrutura, propriedades, numDOF, GLL, GLe, T, NLG, params)
    
    else:
        raise ValueError(f"Analysis not recognized: {analysis_type}")

    return solver.solve()

def metodo_subespaco(KE: csc_array,
                     KG: csc_array,
                     p: int, 
                     max_iter=40, 
                     tol=1e-6):
    """
    Método de subespaço para resolver o problema de autovalor generalizado (K - λ KG) ϕ = 0.

    Parâmetros:
        K: Matriz de rigidez elástica (n x n).
        KG: Matriz de rigidez geométrica (n x n).
        m: Número de autovalores desejados.
        max_iter: Número máximo de iterações.
        tol: Tolerância para convergência.

    Retorna:
        autovalores: Autovalores calculados.
        autovetores: Autovetores correspondentes.
    """
    n = KE.shape[0]

    # Aumento do subespaço
    m = min(2 * p, p + 8)
    if m > n:
        m = n

    # Converter para matrizes esparsas, se necessário
    if not issparse(KE):
        KE = csc_array(KE)
    if not issparse(KG):
        KG = csc_array(KG)

    # Inicialização com vetores aleatórios ortonormalizados
    Q, *_ = qr(np.random.rand(n, m), mode='economic')

    # Vetor para armazenar os autovalores anteriores
    autovalores_finais = np.zeros(p)

    # Fatorizar a matriz KE
    try:
        solve_KE = factorized(KE)
    except Exception as e:
        raise RuntimeError(f"Falha ao fatorar a matriz de rigidez KE. Verifique se ela é singular. Erro: {e}")

    for _ in tqdm(range(max_iter), desc='Executando análise de estabilidade estrutural', leave=False):
        # Resolver sistema linear K @ Y = KG @ Q
        Y = solve_KE(KG @ Q)

        # Ortogonalizar via QR
        Q, *_ = qr(Y, mode='economic')

        # Projeção de Rayleigh-Ritz
        KE_tilde = Q.T @ KE @ Q
        KG_tilde = Q.T @ KG @ Q

        # Resolução do problema de autovalores no subespaço reduzido
        autovalores, autovetores_subespaco = eigh(-KG_tilde, KE_tilde)

        # Ordena os resultados do menor para o maior autovalor
        idx_sorted = np.argsort(autovalores)[::-1]
        autovalores = autovalores[idx_sorted]
        autovetores_subespaco = autovetores_subespaco[:, idx_sorted]

        # Atualizar subespaço com os novos autovetores
        Q = Q @ autovetores_subespaco

        # Filtrar autovalores negativos (tração ou estabilidade nula)
        mask_compressao = autovalores > 1e-15

        # Se nada estiver comprimido, não há flambagem
        if not np.any(mask_compressao):
            autovalores_finais = np.full(p, np.inf)
            break

        # Fatores de carga de compressão
        load_factors = 1.0 / autovalores[mask_compressao]

        # Pegar apenas os 'p' primeiros modos positivos
        n_found = len(load_factors)
        if n_found < p:
            # Preencher com infinito se faltar modos
            load_factors = np.pad(load_factors, (0, p - n_found), constant_values=np.inf)
        else:
            load_factors = load_factors[:p]

        # Verificar convergência
        if np.allclose(load_factors, autovalores_finais, rtol=tol):
            autovalores_finais = load_factors
            break

        autovalores_finais = load_factors

    return autovalores_finais, Q[:, :p]


def eigen_analysis(estrutura, propriedades, numDOF, GLe, GLL, T, Ke, fl, num_modos=5):
    """
    Análise de estabilidade estrutural (flambagem linear) com base no método de subespaço.

    Parâmetros:
        model_type (str): Modelo matemático empregado (treliça, viga, etc.).
        propriedades (dict): Propriedades dos elements.
        numDOF (int): Número de graus de liberdade.
        dofs_per_node (list): Graus de liberdade de cada nó.
        GLe (list): Vinculações de cada elemento.
        GLL (list): Graus de liberdade livres.
        T (list): Matrizes de transformação.
        Ke_reduzida (np.array): Matriz de rigidez reduzida.
        fl (list): Vetor de forças internas na configuração deformada.
        num_modos (int): Número de modos de flambagem a serem calculados (padrão: 5).

    Retorna:
        num_modos (int): Número de modos de flambagem encontrados.
        autovalores (list): Autovalores encontrados.
        d_flamb (list): Vetores de deslocamento para cada modo de flambagem.
    """
    # Dados iniciais
    nos = estrutura.num_nodes

    # Calcula a matriz de rigidez geométrica global
    Kg = assemble_geometric_stiffness_matrix(estrutura, propriedades, numDOF, GLe, GLL, T, fl)

    # Reduzir o intervalo de busca caso as matrizes sejam inferiores ao limite pré-estabelecido de 5
    num_modos = min(num_modos, Ke.shape[0])

    try:
        # Solução do sistema de autovalores e autovetores generalizados, [K_e] + λ * [K_g] = 0
        autovalores, autovetores = metodo_subespaco(Ke, Kg, p=num_modos)
    except Exception as e:
        raise ValueError(f'A estrutura possivelmente não está sujeita a forças de flambagem (normais).\nErro: {e}')

    # Filtra autovalores positivos e autovetores válidos.
    mascara_validos = autovalores > 1e-6  # Apenas fatores de carga positivos são fisicamente relevantes
    autovalores_finais = autovalores[mascara_validos]
    autovetores_finais = autovetores[:, mascara_validos]

    # Ordena do menor fator de carga para o maior
    indices_ordenados = np.argsort(autovalores_finais)
    autovalores_ordenados = autovalores_finais[indices_ordenados]
    autovetores_ordenados = autovetores_finais[:, indices_ordenados]

    # Número de modos de flambagem
    num_modos = len(autovalores_ordenados)

    # Inicializa o vetor de deslocamento global completo
    d_flamb = np.zeros((num_modos, numDOF, 1))

    # Preenche o vetor completo com os resultados dos GDLs livres
    d_flamb[:, GLL, 0] = autovetores_ordenados.T

    # Normalização dos Modos (Escala Unitária)
    for i in range(num_modos):
        modo_atual = d_flamb[i, :]

        # Remodela o vetor completo para agrupar os GDLs por nó
        desloc_por_no = modo_atual.reshape(nos, estrutura.dofs_per_node)

        # Isola apenas os componentes translacionais
        desloc_translacional_por_no = desloc_por_no[:, :3]

        # Calcula a magnitude (norma) do deslocamento resultante em cada nó
        magnitudes_translacionais = np.linalg.norm(desloc_translacional_por_no, axis=1)

        # Encontra o maior deslocamento translacional resultante
        fator_normalizacao = np.max(magnitudes_translacionais)

        # Normaliza o vetor do modo por este fator
        if fator_normalizacao > 1e-9:
            d_flamb[i, :] /= fator_normalizacao

    return num_modos, autovalores_ordenados, d_flamb


def timoshenko_kinematics(structure, xi, invJ, de, NLG, type):
    """
    Calcula as matrizes cinemáticas linear e não-linear para múltiplos elements.

    Args:
        estrutura (Estrutura): Instância da classe Estrutura.
        ξ: Ponto de Gauss ao longo do comprimento (escalar).
        L: Array com os comprimentos dos elements.
        de: Array com os deslocamentos nodais.
        NLG: Se True, calcula a matriz não-linear BbNL.
    Returns:
        BL (np.ndarray): Matriz cinematica linear.
        BNL (np.ndarray): Matriz cinematica nao-linear.
        Ge (np.ndarray): Matriz de gradiente das funções de forma.
    """
    # Initial data
    num_elements = structure.num_elements
    dofs = structure.dofs_per_element

    # Initialize matrices: [BL], [BNL] and [Ge]
    BL = np.zeros((num_elements, 6, dofs))
    BNL = np.zeros((num_elements, 6, dofs))
    Ge = np.zeros((num_elements, 6, dofs))

    # Get shape derivatives
    dN = shape_derivatives(structure, xi)
    dN_dx = dN * invJ

    # Determine if the component is for bending or shear
    if type == 'full':
        # Assemble bending cinematic matrix
        BL[:, 0, 0::6] = dN_dx
        BL[:, 1, 5::6] = dN_dx
        BL[:, 2, 4::6] = dN_dx
        BL[:, 3, 3::6] = dN_dx

        # Assemble gradient and nonlinear cinematic matrices
        if NLG:            
            # Displacement gradients
            # Ge[:, 0, 0::6] = dN_dx  # du/dx
            Ge[:, 1, 1::6] = dN_dx  # dv/dx
            Ge[:, 2, 2::6] = dN_dx  # dw/dx

            # Nonlinear cinematic matrix, {[Ge]{de}}^T @ [Ge]
            BbNL = (Ge @ de).transpose(0, 2, 1) @ Ge

            # Assemble nonlinear cinematic matrix
            BNL[:, 0:1, :] = BbNL
        
    elif type == 'reduced':
        # Get shape functions
        N, _ = shape_functions(structure, xi)

        # Assemble shear cinematic matrix
        BL[:, 4, 1::6] = dN_dx
        BL[:, 4, 5::6] = -N
        BL[:, 5, 2::6] = dN_dx
        BL[:, 5, 4::6] = N

    return BL, BNL, Ge

def matrizes_cinematicas(estrutura, ξ, L, rp, de, NLG):
    """
    Calcula as matrizes cinemáticas linear e não-linear para múltiplos elements.

    Args:
        estrutura (Estrutura): Instância da classe Estrutura.
        ξ: Ponto de Gauss ao longo do comprimento (escalar).
        L: Array com os comprimentos dos elements.
        de: Array com os deslocamentos nodais.
        NLG: Se True, calcula a matriz não-linear BbNL.
    Returns:
        BL (np.ndarray): Matriz cinematica linear.
        BNL (np.ndarray): Matriz cinematica nao-linear.
        Ge (np.ndarray): Matriz de gradiente das funções de forma.
    """
    # Dados iniciais
    num_elementos = estrutura.num_elements

    # Inicializar a matriz cinemática linear, [BL]
    BL = np.zeros((num_elementos, 4, 12))

    # Inversa do Jacobiano de x(ξ)
    J = 2.0 / L

    # Gradientes dos termos da matriz do campo de deslocamentos
    # Termos axiais e de torção (dNu/dξ e dNθx/dξ)
    BL[:, 0, 0] = -J * 0.5
    BL[:, 0, 6] =  J * 0.5
    BL[:, 3, 3] = -J * 0.5
    BL[:, 3, 9] =  J * 0.5

    # Termos de curvatura no plano xy (d²Nv/dξ² e d²Nθz/dξ²)
    BL[:, 1, 1] = J**2 * (3 / 2) * ξ
    BL[:, 1, 5] = J**2 * (L / 4) * (3 * ξ - 1)
    BL[:, 1, 7] = -BL[:, 1, 1]
    BL[:, 1, 11] = J**2 * (L / 4) * (3 * ξ + 1)

    # Termos de curvatura no plano xz (d²Nw/dξ² e d²Nθy/dξ²)
    BL[:, 2, 2] =   BL[:, 1, 1]
    BL[:, 2, 4] =  -BL[:, 1, 5]
    BL[:, 2, 8] =   BL[:, 1, 7]
    BL[:, 2, 10] = -BL[:, 1, 11]

    # Inicializar a matriz de gradientes
    Ge = np.zeros((num_elementos, 4, 12))
    BNL = np.zeros((num_elementos, 4, 12))

    # Matriz não-linear de deformação-deslocamento, [BbNL]
    if NLG:
        # Termos axiais (dNu/dξ)
        # Ge[:, 0, 0] = -J * 0.5
        # Ge[:, 0, 6] =  J * 0.5

        # Termos de curvatura no plano xy (dNv/dξ e dNθz/dξ)
        Ge[:, 1, 1] = J * (3 / 4) * (ξ ** 2 - 1)
        Ge[:, 1, 5] = J * (L / 8) * (3 * ξ ** 2 - 2 * ξ - 1)
        Ge[:, 1, 7] = -J * (3 / 4) * (ξ ** 2 - 1)
        Ge[:, 1, 11] = J * (L / 8) * (3 * ξ ** 2 + 2 * ξ - 1)

        # Termos de curvatura no plano xz (dNw/dξ e dNθy/dξ)
        Ge[:, 2, 2] =   Ge[:, 1, 1]
        Ge[:, 2, 4] =  -Ge[:, 1, 5]
        Ge[:, 2, 8] =   Ge[:, 1, 7]
        Ge[:, 2, 10] = -Ge[:, 1, 11]

        # Raio de giração polar e termos de torção (dNθx/dξ)
        # Ge[:, 3, 3] = -J * 0.5 * rp
        # Ge[:, 3, 9] =  J * 0.5 * rp

        # Matriz deformação-deslocamento não linear axial, {[Ge]{de}}^T @ [Ge]
        BbNL = (Ge @ de).transpose(0, 2, 1) @ Ge

        # Montar a matriz BNL completa
        BNL[:, 0:1, :] = BbNL # np.concatenate([BbNL, np.zeros((elements, 3, 12))], axis=1)

    return BL, BNL, Ge


@lru_cache(maxsize=128)
def get_gauss_points(n_points: int = 3):
    """Cache to store Gauss points (constants).

    Args:
        n_points (int, optional): Number of Gauss points. Defaults to 3.
    """
    return np.polynomial.legendre.leggauss(n_points)

def analise_elemento(de, estrutura, propriedades, NLG):
    # Initial data
    dofs = estrutura.dofs_per_element
    num_elements = estrutura.num_elements

    # Initialize matrices
    kt = np.zeros((num_elements, dofs, dofs))
    fe = np.zeros((num_elements, dofs, 1))

    # Dados geométricos e constitutivos
    L = propriedades['L']
    k = propriedades['k']
    EA = propriedades['E'] * propriedades['A']
    GA = propriedades['G'] * propriedades['A']
    EIy = propriedades['E'] * propriedades['Iy']
    EIz = propriedades['E'] * propriedades['Iz']
    GIt = propriedades['G'] * propriedades['It']
    rp = propriedades['rp']

    # Jacobian and its inverse
    detJ = (L / 2).reshape(-1, 1, 1)
    invJ = (2 / L).reshape(-1, 1)

    # Verify if the beam element is quadratic (Timoshenko) or cubic (Euler-Bernoulli)
    if estrutura.is_quadratic:
        # Get Gauss points and weights (full integration)
        points, weights = get_gauss_points(n_points=3)

        # Total constitutive matrix
        D = np.zeros((num_elements, 6, 6))
        D[:, 0, 0] = EA
        D[:, 1, 1] = EIz
        D[:, 2, 2] = EIy
        D[:, 3, 3] = GIt
        D[:, 4, 4] = k * GA
        D[:, 5, 5] = k * GA

        for xi, wi in zip(points, weights):
            # Bending kinematic matrices
            BL, BNL, Ge = timoshenko_kinematics(estrutura, xi, invJ, de, NLG, type='full')

            # Matriz deformação-deslocamento total, [B]
            B = BL + BNL

            # Matriz deformação-deslocamento incremental, [Bε]
            Bε = BL + 0.5 * BNL

            # Matriz de rigidez material, [kl]
            kl = np.einsum('eji,ejk,ekl->eil', B, D, B, optimize='optimal')

            # Tensões internas, [S]
            S = np.einsum('eij,ejk,ekl->eil', D, Bε, de, optimize='optimal')

            # Matriz de rigidez geométrica local, [kg]
            if NLG:
                kg = S[:, 0:1, :] * np.einsum('eji,ejk->eik', Ge, Ge, optimize='optimal')
            else:
                kg = 0

            # Matriz de rigidez tangente local, [kt]
            kt += wi * detJ * (kl + kg)

            # Vetor de forças internas, [fe]
            fe += (wi * detJ) * np.einsum('eji,ejk->eik', B, S, optimize='optimal')
        
        # Reduced integration for shear terms
        points, weights = get_gauss_points(n_points=2)

        for xi, wi in zip(points, weights):
            # Shear kinematic matrices
            B, *_ = timoshenko_kinematics(estrutura, xi, invJ, de, NLG, type='reduced')

            # Matriz de rigidez material, [kl]
            kl = np.einsum('eji,ejk,ekl->eil', B, D, B, optimize='optimal')

            # Tensões internas, [S]
            S = np.einsum('eij,ejk,ekl->eil', D, B, de, optimize='optimal')

            # Matriz de rigidez tangente local, [kt]
            kt += (wi * detJ) * kl

            # Vetor de forças internas, [fe]
            fe += (wi * detJ) * np.einsum('eji,ejk->eik', B, S, optimize='optimal')

    else:
        # Pontos e pesos de Gauss (comprimento)
        points, weights = get_gauss_points(n_points=3)

        # Matriz constitutiva
        D = np.zeros((num_elements, 4, 4))
        D[:, 0, 0] = EA
        D[:, 1, 1] = EIz
        D[:, 2, 2] = EIy
        D[:, 3, 3] = GIt

        # Pré-alocar buffers
        B = np.empty((num_elements, 4, 12))
        Bε = np.empty((num_elements, 4, 12))
        S = np.empty((num_elements, 4, 1))

        for ξ, wi in zip(points, weights):
            # Matrizes cinemáticas
            BL, BNL, Ge = matrizes_cinematicas(estrutura, ξ, L, rp, de, NLG)

            # Matriz deformação-deslocamento total, [B]
            np.add(BL, BNL, out=B) # BL + BNL

            # Matriz deformação-deslocamento incremental, [Bε]
            np.add(BL, 0.5 * BNL, out=Bε) # BL + 0.5 * BNL

            # Matriz de rigidez material, [kl]
            kl = np.einsum('eji,ejk,ekl->eil', B, D, B, optimize='optimal')

            # Tensões internas, [S]
            np.einsum('eij,ejk,ekl->eil', D, Bε, de, out=S, optimize='optimal') # S = np.einsum('eij,ejk,ekl->eil', D, Bε, de, optimize='optimal')

            # Matriz de rigidez geométrica local, [kg]
            if NLG:
                kg = S[:, 0:1, :] * np.einsum('eji,ejk->eik', Ge, Ge, optimize='optimal')
            else:
                kg = 0

            # Matriz de rigidez tangente local, [kt]
            kt += wi * detJ * (kl + kg)

            # Vetor de forças internas, [fe]
            fe += (wi * detJ) * np.einsum('eji,ejk->eik', B, S, optimize='optimal')

    # Apply static condensation if necessary
    kt, fe = static_condensation(estrutura, kt, fe)

    return kt, fe

def analise_global(d, estrutura, propriedades, numDOF, GLL, GLe, T, NLG):
    """
    Calcula as matrizes de rigidez e vetores de força internos para todos os elements
    e monta as matrizes globais e o vetor de forças internas.

    Args:
        d (array_like): Vetor de deslocamentos globais.
        estrutura (objeto): Instância da classe Estrutura.
        propriedades (dict): Dicionário com as propriedades dos elements.
        numDOF (int): Número total de graus de liberdade.
        GLL (array_like): Nós globais.
        GLe (array_like): Elementos globais.
        T (array_like): Matrizes de transformação local-global.
        NLG (bool): Se True, calcula a matriz de rigidez geométrica.

    Returns:
        KG_reduzida (array_like): Matriz de rigidez global reduzida.
        Fint (array_like): Vetor de forças internas global.
    """
    # Inicialização do vetor de forças globais, {Fe}
    F = np.zeros((numDOF, 1))

    # Criação dos deslocamentos locais do elemento
    de = atribuir_deslocamentos(numDOF, GLL, GLe, T, d)

    # Calcula as matrizes de rigidez e vetores de força internos para todos os elements
    ke, fe = analise_elemento(de, estrutura, propriedades, NLG)
    
    # Converter para o sistema global
    ke_global = np.einsum('eji,ejk,ekl->eil', T, ke, T, optimize='optimal')
    fe_global = np.einsum('eji,ejk->eik', T, fe, optimize='optimal')

    # Montar a matriz de rigidez global e o vetor de forças
    np.add.at(F, GLe, fe_global)

    # Montagem da matriz de rigidez tangente global, [KG]
    KG = assemble_sparse_matrix(estrutura, ke_global, numDOF, GLe)

    # Aplicação das condições de contorno
    KG_reduzida = KG[np.ix_(GLL, GLL)]
    Fint = F[GLL]
    Ra = F[~GLL]

    return KG_reduzida, Fint, Ra, fe, de

def calcular_estrutura(
        estrutura,
        propriedades,
        coords,
        numDOF,
        GLL,
        GLe,
        T,
        Ke,
        ke,
        Fe,
        fq
) -> tuple[Dict, Dict, Dict, Dict]:
    """
    Função para calcular a estrutura.

    Args:
        estrutura (Estrutura): Instância da classe Estrutura.
        propriedades (np.ndarray): Matriz de propriedades dos elements.
        dofs_per_node (int): Graus de liberdade por nó.
        numDOF (int): Número de graus de liberdade.
        resDOF (np.ndarray): Vetor de graus de liberdade restritos.
        GLe (np.ndarray): Vetor de graus de liberdade associados a cada elemento.
        T (np.ndarray): Matriz de transformação.
        Ke (np.ndarray): Matriz de rigidez dos elements (global).
        ke (np.ndarray): Matriz de rigidez dos elements (local).
        Fe (np.ndarray): Vetor de forças externas globais.
        f (np.ndarray): Vetor de forças distribuídas.

    Returns:
        deslocamentos (dict): Dicionário dos deslocamentos (globais e locais).
        esforcos (dict): Dicionário dos esforços (globais e locais).
    """
    # Dados iniciais
    analysis = estrutura.analysis

    # Apply boundary conditions to global force vector
    F = Fe[GLL]

    # Encontra os índices (DOFs) com rigidez abaixo do limite
    diagonal_Ke = Ke.diagonal()
    dofs_instaveis = np.where(np.abs(diagonal_Ke) < 1e-9)[0]

    if dofs_instaveis.size > 0:
        # A ESTRUTURA É INSTÁVEL!
        raise ValueError(
            f"ERRO: Estrutura instável (mecanismo detectado).\n"
            f"DOFs com rigidez nula: {dofs_instaveis}\n"
            f"Verifique condições de contorno e member_releases."
        )

    # Resolução do sistema de equações para os deslocamentos globais
    d = spsolve(Ke, F).reshape(-1, 1)

    if np.isnan(d).any():
        # A ESTRUTURA É INSTÁVEL!
        raise ValueError(
            "ERRO: Estrutura instável (Instabilidade de corpo rígido).\n"
            "Matriz resultante é singular.\n"
            "Verifique condições de contorno e member_releases."
        )

    # Vetor dos deslocamentos locais, {dl}
    dl = atribuir_deslocamentos(numDOF, GLL, GLe, T, d)

    # Vetor das forças locais, {fl}
    fl = ke @ dl - T @ fq

    # Pré-alocar estruturas de retorno
    deslocamentos = {}
    esforcos = {}
    d_total = np.zeros((numDOF, 1))

    if analysis == 'linear':
        # Obter os deslocamentos e esforços lineares
        # d, dl, fl, f_vs_d, Ra, convergence_data = newton_raphson(F, KE, estrutura, propriedades, numDOF, GLL, GLe, T, NLG=False)
        d, dl, fl, f_vs_d, Ra, convergence_data = run_nonlinear_solver(
            AnalysisType.NEWTON_RAPHSON,
            F, Ke, estrutura, propriedades, numDOF, GLL, GLe, T, NLG=False,
            num_passos_inicial=20, abs_tol=1e-8, rel_tol=1e-8)
        
        # Atribuir os deslocamentos
        d_total[GLL] = d
        deslocamentos['d'] = d_total
        deslocamentos['de'] = dl

        # Atribuir os esforços
        esforcos['F'] = F
        esforcos['R'] = Ra - Fe[~GLL]
        esforcos['fe'] = fl - T @ fq

    elif analysis == 'nonlinear':
        # Obter os deslocamentos e esforços não-lineares
        # d, dnl, fnl, f_vs_d, Ra, convergence_data = run_nonlinear_solver(
        #     AnalysisType.ARC_LENGTH,
        #     F, Ke, estrutura, propriedades, numDOF, GLL, GLe, T, NLG=True,
        #     lmbda0=0.01, allow_lambda_exceed=False, max_lambda=1.5 , psi=0.001, abs_tol=1e-6, rel_tol=1e-6)

        d, dnl, fnl, f_vs_d, Ra, convergence_data = run_nonlinear_solver(
            AnalysisType.NEWTON_RAPHSON,
            F, Ke, estrutura, propriedades, numDOF, GLL, GLe, T, NLG=True,
            num_passos_inicial=100, abs_tol=1e-8, rel_tol=1e-8)
        
        # Atribuir os deslocamentos
        d_total[GLL] = d
        deslocamentos['d'] = d_total
        deslocamentos['de'] = dnl

        # Atribuir os esforços
        esforcos['F'] = F
        esforcos['R'] = Ra - Fe[~GLL]
        esforcos['fe'] = fnl

    elif analysis == 'buckling':
        # Obter os autovalores e autovetores
        num_modos, autovalores, autovetores = eigen_analysis(estrutura, propriedades, numDOF, GLe, GLL, T, Ke, fl)

        # Atribuir os autovalores e desconsiderar os termos de análise não linear
        esforcos['autovalores'] = autovalores
        f_vs_d = None
        convergence_data = None

        # Atribuir os autovetores
        deslocamentos['d'] = autovetores
        deslocamentos['de'] = T @ autovetores[:, GLe]

    return deslocamentos, esforcos, f_vs_d, convergence_data
