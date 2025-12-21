# Third-party libraries
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class MeshData:
    """Armazena dados de malhas."""
    indeformada: Optional[Any] = None
    deformada: Optional[Any] = None
    secoes: Optional[Any] = None
    secoes_indices: Optional[Any] = None
    coords: Optional[np.ndarray] = None
    coords_deformadas: Optional[np.ndarray] = None


@dataclass
class ResultsData:
    """Armazena dados de resultados da análise."""
    esforcos_int: Optional[Dict] = None
    deslocamentos: Optional[Dict] = None
    reacoes_nos_apoios: Optional[Dict] = None
    autovalores: Optional[np.ndarray] = None
    convergence_data: Optional[Dict] = None
    f_vs_d: Optional[list] = None
    MT: Optional[np.ndarray] = None


@dataclass
class AnalysisConfig:
    """Configurações da análise."""
    tipo: str = 'Linear'
    subdivisions: int = 1
    num_modos: int = 0
    modo_flambagem_atual: int = 0


@dataclass
class VisualizationOptions:
    """Opções de visualização."""
    escala: float = 1.0
    plotar_secao: bool = True
    visualizacao_tipo: str = 'Colormap'
    modo_escuro: bool = False


class StateManager:
    """Gerenciador de estado da aplicação."""
    def __init__(self):
        self.estrutura = None
        self.malhas = MeshData()
        self.resultados = ResultsData()
        self.config = AnalysisConfig()
        self.viz_options = VisualizationOptions()
        self._observers = []
    
    def reset_results(self):
        """Reseta apenas os dados de resultados, mantendo a estrutura."""
        estrutura_cache = self.estrutura
        self.malhas = MeshData()
        self.resultados = ResultsData()
        self.config = AnalysisConfig()
        self.estrutura = estrutura_cache
        self._notify_observers('results_reset')
    
    def reset_all(self):
        """Reseta todo o estado da aplicação."""
        self.__init__()
        self._notify_observers('state_reset')
    
    def set_structure(self, structure):
        """Define a estrutura carregada."""
        self.estrutura = structure
        self._notify_observers('structure_loaded')
    
    def set_analysis_results(self, **kwargs):
        """Atualiza os resultados da análise."""
        # Atualizar estrutura se fornecida
        if 'estrutura' in kwargs:
            self.estrutura = kwargs['estrutura']
        
        # Atualizar malhas
        self.malhas.indeformada = kwargs.get('malha_indeformada')
        self.malhas.deformada = kwargs.get('malha_deformada')
        self.malhas.coords = kwargs.get('coords')
        self.malhas.coords_deformadas = kwargs.get('coords_deformadas')
        
        # Atualizar resultados
        self.resultados.esforcos_int = kwargs.get('esforcos_int')
        self.resultados.deslocamentos = kwargs.get('deslocamentos')
        self.resultados.reacoes_nos_apoios = kwargs.get('reacoes_nos_apoios')
        self.resultados.autovalores = kwargs.get('autovalores')
        self.resultados.convergence_data = kwargs.get('convergence_data')
        self.resultados.f_vs_d = kwargs.get('f_vs_d')
        self.resultados.MT = kwargs.get('MT')
        
        # Atualizar configuração
        if self.estrutura is not None:
            self.config.elementos = self.estrutura.num_elements
            self.config.num_modos = (self.resultados.autovalores.size 
                                    if self.resultados.autovalores is not None else 0)

            analise_map = {
                'linear': 'Linear',
                'nonlinear': 'Não Linear',
                'buckling': 'Flambagem'
            }
            self.config.tipo = analise_map.get(self.config.tipo, 'Linear')

        self._notify_observers('results_updated')
    
    def has_structure(self) -> bool:
        """Verifica se há uma estrutura carregada."""
        return self.estrutura is not None
    
    def has_results(self) -> bool:
        """Verifica se há resultados disponíveis."""
        return (self.resultados.deslocamentos is not None or 
                self.resultados.esforcos_int is not None)
    
    def is_buckling_analysis(self) -> bool:
        """Verifica se é análise de flambagem."""
        return self.config.tipo == 'Flambagem'
    
    def add_observer(self, callback):
        """Adiciona um observador para mudanças de estado."""
        if callback not in self._observers:
            self._observers.append(callback)
    
    def remove_observer(self, callback):
        """Remove um observador."""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def _notify_observers(self, event_type: str):
        """Notifica observadores sobre mudanças."""
        for observer in self._observers:
            try:
                observer(event_type, self)
            except Exception as e:
                print(f"Erro ao notificar observador: {e}")
    
    def get_current_buckling_mode(self) -> int:
        """Retorna o modo de flambagem atual."""
        return self.config.modo_flambagem_atual
    
    def set_buckling_mode(self, mode: int):
        """Define o modo de flambagem atual."""
        if 0 <= mode < self.config.num_modos:
            self.config.modo_flambagem_atual = mode
            self._notify_observers('buckling_mode_changed')
    
    def get_visualization_options(self) -> Dict[str, Any]:
        """Retorna as opções de visualização como dicionário."""
        return {
            'escala': self.viz_options.escala,
            'plotar_secao': self.viz_options.plotar_secao,
            'grid_secao': self.malhas.secoes,
            'MT': self.resultados.MT,
            'modo': self.config.modo_flambagem_atual
        }
