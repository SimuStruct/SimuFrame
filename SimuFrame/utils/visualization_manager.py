# Third-party libraries
import numpy as np
import pyvista as pv
from functools import partial

# Internal modules
from SimuFrame.post_processing.visualization import (
    plot_structure, 
    plotar_deslocamentos, 
    plotar_esforcos, 
    plotar_reacoes, 
    adicionar_escalares, 
    criar_secoes_base, 
    generate_mesh
)
from SimuFrame.utils.helpers import extract_element_data, orientation_vector

class VisualizationManager:
    """
    Gerencia a lógica de visualização e chamadas PyVista.
    """
    def __init__(self, plotter):
        self.plotter = plotter
        self.base_scale_size = 1.0  # Tamanho de referência
        self.plot_handlers = {}     # Mapa de strings -> funções de plotagem
        self._setup_handlers()      # Inicializa o mapa

    def calculate_adaptive_scale(self, estrutura, propriedades):
        """
        Calcula um fator de escala visual baseado na geometry_type.
        """
        # Obter coordenadas
        coords = np.array([no.coord for no in estrutura.original_nodes.values()])
        if len(coords) == 0:
            return 1.0
        
        # Bounding Box global
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        bbox_diag = np.linalg.norm(bbox_max - bbox_min)
        if bbox_diag < 1e-6:
            bbox_diag = 1.0

        # Comprimento médio dos elements
        lengths = propriedades.L
        avg_len = np.mean(lengths)

        # Escala base adaptativa
        base_scale = avg_len * 0.40
        
        # Clamping (Limites globais baseados no tamanho total da estrutura)
        lower_limit = bbox_diag * 0.005 
        upper_limit = bbox_diag * 0.05
        
        visual_size = np.clip(base_scale, lower_limit, upper_limit)
        
        return visual_size

    def generate_base_mesh(self, context):
        """
        Gera a malha indeformada (base) da estrutura.

        Args:
            context (context): Objeto da classe ProjectContext.
        """
        structure = context.structure

        # Extração dos dados
        coords, initial_coords, _, properties = extract_element_data(structure)
        self.ref_vector = orientation_vector(structure, coords, initial_coords)

        # Criação das seções
        context.secoes_base, secoes_indices = criar_secoes_base(structure)

        # Criação da malha
        context.malha_indeformada = generate_mesh(
            structure, context.secoes_base, secoes_indices, coords, self.ref_vector, geometry='undeformed'
        )

        # Calcular e armazenar a escala visual
        context.base_scale_size = self.calculate_adaptive_scale(structure, properties)
    
    def plot_structure(self, context):
        """
        Gerar a malha deformada da estrutura, finalizada a análise.

        Args:
            structure (Structure): Objeto da classe Structure.
        """
        # Extração dos dados
        structure = context.structure
        results = context.results
            
