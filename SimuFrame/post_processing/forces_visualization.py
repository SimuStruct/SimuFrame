# Built-in libraries
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Any

# Third-party libraries
import numpy as np
import pyvista as pv
import numpy.typing as npt

# Local libraries
from .visualization import adicionar_escalares

class VisualizationType(Enum):
    """Tipos de visualização disponíveis."""
    DIAGRAM = 'Diagrama'
    COLORMAP = 'Colormap'


class ForceType(Enum):
    """Tipos de esforços e suas configurações."""
    FX = ('Fx', 'b', 1, 2, 'Fx (kN)')
    FY = ('Fy', 'g', 1, 1, 'Fy (kN)')
    FZ = ('Fz', 'r', 1, 2, 'Fz (kN)')
    MX = ('Mx', 'peru', -1, 2, 'Mx (kN.m)')
    MY = ('My', 'purple', -1, 2, 'My (kN.m)')
    MZ = ('Mz', 'slateblue', 1, 1, 'Mz (kN.m)')

    def __init__(self, key, color, direction, axis, label):
        self.key = key
        self.color = color
        self.direction = direction
        self.axis = axis
        self.label = label

    @classmethod
    def from_key(cls, key: str):
        """Obtém ForceType a partir da chave."""
        for ft in cls:
            if ft.key == key:
                return ft
        raise ValueError(f"Tipo de esforço desconhecido: {key}")


@dataclass
class ForceData:
    """Dados de esforços processados."""
    valores: np.ndarray
    valores_corrigidos: np.ndarray
    magnitude: float
    title: str
    force_type: ForceType


@dataclass
class PlotConfig:
    """Configuração de plotagem."""
    escala: float = 1.0
    modo: int = 0
    force: str = 'Fx'
    visualizacao: str = 'Colormap'
    MT: Optional[Any] = None
    grid_secao: Optional[Any] = None


class ForceDataExtractor:
    """Extrai e processa dados de esforços."""

    @staticmethod
    def extract(config, esforcos_int, force_key: str, modo: int) -> ForceData:
        """
        Extrai dados de esforços da estrutura.

        Args:
            estrutura: Objeto Structure
            esforcos_int: Dicionário com esforços internos
            force_key: Chave do tipo de esforço ('Fx', 'My', etc)
            modo: Modo de flambagem (se aplicável)

        Returns:
            ForceData com valores processados
        """
        is_buckling = config.analysis == 'buckling'
        force_type = ForceType.from_key(force_key)

        # Extrair valores
        if is_buckling:
            valores = esforcos_int[force_key][modo]
        else:
            valores = esforcos_int[force_key]

        # Inverter sinal para Fx
        if force_key == 'Fx':
            valores = -valores

        # Corrigir componentes (inverter coluna 1)
        valores_corrigidos = valores.copy()
        valores_corrigidos[:, 1] *= -1

        # Calcular magnitude
        max_valor = np.max(np.abs(valores_corrigidos))
        magnitude = 1.0 / max_valor if max_valor > 1e-9 else 1.0

        return ForceData(
            valores=valores,
            valores_corrigidos=valores_corrigidos,
            magnitude=magnitude,
            title=force_type.label,
            force_type=force_type
        )


class DiagramPlotter:
    """Plota diagramas de esforços."""

    def __init__(self, widget, estrutura, coords):
        self.widget = widget
        self.estrutura = estrutura
        self.coords = coords

    def plot(self, force_data: ForceData, config: PlotConfig):
        """
        Plota diagrama de esforços.

        Args:
            force_data: Dados de esforços processados
            config: Configuração de plotagem
        """
        # Ocultar malha deformada
        self._hide_deformed_mesh()

        # Calcular pontos deformados
        pontos_deformados = self._compute_deformed_points(
            force_data, config
        )

        # Criar malha do diagrama
        malha_diagrama = self._create_diagram_mesh(pontos_deformados)

        # Criar legendas
        legendas_pontos, legendas_textos = self._create_legends(
            force_data.valores_corrigidos, pontos_deformados
        )

        # Adicionar ao plotter
        self._add_to_plotter(
            malha_diagrama,
            legendas_pontos,
            legendas_textos,
            force_data
        )

    def _hide_deformed_mesh(self):
        """Oculta atores da malha deformada."""
        if 'malha_deformada_tubos' in self.widget.actors:
            self.widget.actors['malha_deformada_tubos'].visibility = False
        if 'malha_deformada_secoes' in self.widget.actors:
            self.widget.actors['malha_deformada_secoes'].visibility = False

        # Ocultar barra de escalares
        try:
            self.widget.scalar_bar.SetVisibility(False)
        except StopIteration:
            pass

    def _compute_deformed_points(self, force_data: ForceData,
                                 config: PlotConfig) -> np.ndarray:
        """Calcula pontos deformados pelo diagrama de esforços."""
        # Escalar esforços
        f_escalado = (force_data.magnitude *
                      force_data.valores_corrigidos * config.escala)

        # Criar matriz de forças 3D
        f = np.zeros((self.estrutura.num_elements, 2, 3))
        axis = force_data.force_type.axis
        direction = force_data.force_type.direction
        f[:, :, axis] = direction * f_escalado

        # Transformar para sistema global
        f_global = np.einsum('eji,enj->eni', config.MT, f)

        # Retornar coordenadas deformadas
        return self.coords + config.escala * f_global

    def _create_diagram_mesh(self, pontos_deformados: np.ndarray) -> pv.PolyData:
        """Cria malha poligonal do diagrama."""
        pontos_poligonos = []
        faces_poligonos = []
        offset = 0

        for i in range(self.estrutura.num_elements):
            # Pontos da base e topo
            p_base_1 = self.coords[i, 0]
            p_base_2 = self.coords[i, 1]
            p_topo_1 = pontos_deformados[i, 0]
            p_topo_2 = pontos_deformados[i, 1]

            # Adicionar pontos
            pontos_poligonos.extend([p_base_1, p_topo_1, p_topo_2, p_base_2])

            # Adicionar face
            faces_poligonos.extend([4, offset, offset + 1, offset + 2, offset + 3])
            offset += 4

        return pv.PolyData(np.array(pontos_poligonos), faces=faces_poligonos)

    def _create_legends(self, valores_corrigidos: np.ndarray,
                        pontos_deformados: np.ndarray) -> Tuple[list, list]:
        """Cria legendas para os valores extremos."""
        pontos_legenda = []
        textos_legenda = []

        n_sub = self.estrutura.subdivisions
        num_membros = len(self.estrutura.original_members)

        for i in range(num_membros):
            start_idx = i * n_sub
            end_idx = start_idx + n_sub - 1

            # Valor no início
            valor_inicio = valores_corrigidos[start_idx, 0]
            if np.abs(valor_inicio) > 1e-9:
                pontos_legenda.append(pontos_deformados[start_idx, 0])
                textos_legenda.append(f"{valor_inicio:.3f}")

            # Valor no fim
            valor_fim = valores_corrigidos[end_idx, 1]
            if np.abs(valor_fim) > 1e-9:
                pontos_legenda.append(pontos_deformados[end_idx, 1])
                textos_legenda.append(f"{valor_fim:.3f}")

        return pontos_legenda, textos_legenda

    def _add_to_plotter(self, malha_diagrama, pontos_legenda,
                        textos_legenda, force_data: ForceData):
        """Adiciona malha e legendas ao plotter."""
        # Adicionar diagrama
        self.widget.add_mesh(
            malha_diagrama,
            color=force_data.force_type.color,
            opacity=0.7,
            name='diagrama_esforcos'
        )

        # Adicionar título
        self.widget.add_text(
            force_data.title,
            position='upper_left',
            font_size=10,
            name='rotulos_esforcos'
        )

        # Adicionar legendas
        if pontos_legenda:
            self.widget.add_point_labels(
                pontos_legenda,
                textos_legenda,
                name='legendas_esforcos',
                font_size=18,
                text_color='black',
                shape=None,
                always_visible=True
            )


class ColormapPlotter:
    """Plota mapa de cores de esforços."""

    def __init__(self, widget, estrutura, malha_deformada):
        self.widget = widget
        self.estrutura = estrutura
        self.malha_deformada = malha_deformada

    def plot(self, force_data: ForceData, config: PlotConfig):
        """
        Plota mapa de cores de esforços.

        Args:
            force_data: Dados de esforços processados
            config: Configuração de plotagem
        """
        needs_recalculation = self._needs_mesh_recalculation(config.modo)

        if needs_recalculation:
            self._plot_new_mesh(force_data, config)
        else:
            self._update_existing_mesh(force_data, config)

        # Garantir que a barra de escalares esteja visível
        self._ensure_scalar_bar_visible()

    def _ensure_scalar_bar_visible(self):
        """Garante que a barra de escalares esteja visível."""
        if hasattr(self.widget, 'scalar_bar') and self.widget.scalar_bar is not None:
            self.widget.scalar_bar.SetVisibility(True)

    def _needs_mesh_recalculation(self, modo: int) -> bool:
        """Verifica se precisa recalcular a malha."""
        if not self.estrutura.is_buckling:
            return not self._mesh_exists()

        # Para flambagem, verificar mudança de modo
        if not hasattr(self.widget, 'estado_anterior'):
            return True

        return self.widget.estado_anterior.get('modo') != modo

    def _mesh_exists(self) -> bool:
        """Verifica se a malha já existe."""
        return ('malha_deformada_tubos' in self.widget.actors and
                'malha_deformada_secoes' in self.widget.actors)

    def _plot_new_mesh(self, force_data: ForceData, config: PlotConfig):
        """Cria e plota nova malha."""
        # Limpar atores antigos (exceto estrutura indeformada)
        self._clear_old_actors()

        # Extrair malhas
        tubos, secoes = self._extract_meshes(config.modo)

        # Combinar malhas
        grid_tubos = tubos.combine()
        grid_secao = secoes.combine()

        # Adicionar escalares
        valores_iniciais = force_data.valores_corrigidos[:, 0]
        valores_finais = force_data.valores_corrigidos[:, 1]
        adicionar_escalares(secoes, grid_secao, valores_iniciais, valores_finais)

        # Calcular limites
        vmin, vmax = force_data.valores_corrigidos.min(), force_data.valores_corrigidos.max()

        # Configurar barra de escalares
        scalar_bar_args = {
            'title': force_data.title,
            'title_font_size': 20,
            'label_font_size': 16,
            'n_labels': 10,
            'vertical': True,
            'fmt': '%.3e'
        }

        # Adicionar ao plotter
        self.widget.add_mesh(
            grid_tubos,
            color='sienna',
            opacity=1.0,
            name='malha_deformada_tubos'
        )
        self.widget.add_mesh(
            grid_secao,
            scalars='scalars',
            cmap="turbo",
            clim=(vmin, vmax),
            scalar_bar_args=scalar_bar_args,
            name='malha_deformada_secoes'
        )

        # Salvar estado
        if not self.estrutura.is_buckling:
            self.widget.estado_anterior = {'modo': config.modo}

    def _update_existing_mesh(self, force_data: ForceData, config: PlotConfig):
        """Atualiza malha existente com novos dados."""
        # Garantir visibilidade
        if 'malha_deformada_tubos' in self.widget.actors:
            self.widget.actors['malha_deformada_tubos'].visibility = True
        if 'malha_deformada_secoes' in self.widget.actors:
            self.widget.actors['malha_deformada_secoes'].visibility = True

        # Obter ator e dataset
        secoes = self.malha_deformada['section'] if not self.estrutura.is_buckling else self.malha_deformada['section'][config.modo]
        secao_actor = self.widget.actors['malha_deformada_secoes']
        grid_secao = secao_actor.mapper.dataset

        # Limites
        valores_iniciais = force_data.valores_corrigidos[:, 0]
        valores_finais = force_data.valores_corrigidos[:, 1]
        vmin, vmax = force_data.valores_corrigidos.min(), force_data.valores_corrigidos.max()

        # Atualizar escalares
        adicionar_escalares(secoes, grid_secao, valores_iniciais, valores_finais)

        # Atualizar mapper
        secao_actor.mapper.scalar_range = (vmin, vmax)

        # Atualizar barra de escalares
        self.widget.update_scalar_bar_range([vmin, vmax])

        if self.widget.scalar_bars:
            self.widget.scalar_bar.SetTitle(force_data.title)
        else:
            self._add_scalar_bar(force_data.title)

    def _clear_old_actors(self):
        """Remove atores antigos, exceto estrutura indeformada."""
        actors_to_keep = [
            'estrutura_indeformada_secoes', 'estrutura_indeformada_nos',
            'apoios_engaste', 'apoios_rotula', 'apoios_translacao', 'apoios_rotacao'
        ]

        for name in list(self.widget.actors.keys()):
            if name not in actors_to_keep:
                self.widget.remove_actor(name)

    def _extract_meshes(self, modo: int):
        """Extrai malhas tubos e seções."""
        if not self.estrutura.is_buckling:
            tubos = self.malha_deformada['tubos']
            secoes = self.malha_deformada['section']
        else:
            tubos = self.malha_deformada['tubos'][modo]
            secoes = self.malha_deformada['section'][modo]

        return tubos, secoes

    def _add_scalar_bar(self, title: str):
        """Adiciona barra de escalares."""
        self.widget.add_scalar_bar(
            title=title,
            title_font_size=20,
            label_font_size=16,
            n_labels=10,
            vertical=True,
            fmt='%.3e'
        )


class CameraManager:
    """Gerencia estado da câmera."""

    def __init__(self, widget):
        self.widget = widget
        self.saved_state = None

    def save(self):
        """Salva estado atual da câmera."""
        self.saved_state = self.widget.camera.copy()

    def restore(self):
        """Restaura estado salvo da câmera."""
        if self.saved_state is not None:
            self.widget.camera = self.saved_state

    def __enter__(self):
        """Context manager: salva ao entrar."""
        self.save()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: restaura ao sair."""
        self.restore()


def plotar_esforcos(estrutura, malha_deformada, coords,
                    esforcos_int, widget, **kwargs):
    """
    Plota esforços internos na estrutura.

    Args:
        estrutura: Objeto Structure
        malha_deformada: Malha deformada da estrutura
        coords: Coordenadas dos pontos
        esforcos_int: Dicionário com esforços internos
        widget: Widget PyVista para plotagem
        **kwargs: Argumentos opcionais (escala, modo, force, visualizacao, MT, grid_secao)

    Raises:
        ValueError: Se tipo de esforço ou visualização for inválido
    """
    # Criar configuração
    config = PlotConfig(
        escala=kwargs.get('escala', 1.0),
        modo=kwargs.get('modo', 0),
        force=kwargs.get('force', 'Fx'),
        visualizacao=kwargs.get('visualizacao', 'Colormap'),
        MT=kwargs.get('MT', []),
        grid_secao=kwargs.get('grid_secao')
    )

    # Gerenciar câmera (salvar e restaurar automaticamente)
    with CameraManager(widget):
        # Extrair dados de esforços
        force_data = ForceDataExtractor.extract(
            estrutura, esforcos_int, config.force, config.modo
        )

        # Plotar de acordo com o tipo de visualização
        if config.visualizacao == VisualizationType.DIAGRAM.value:
            plotter = DiagramPlotter(widget, estrutura, coords)
            plotter.plot(force_data, config)

        elif config.visualizacao == VisualizationType.COLORMAP.value:
            plotter = ColormapPlotter(widget, estrutura, malha_deformada)
            plotter.plot(force_data, config)

        else:
            raise ValueError(f"Tipo de visualização inválido: {config.visualizacao}")

        # Limpar atores auxiliares
        _cleanup_auxiliary_actors(widget)


def _cleanup_auxiliary_actors(widget):
    """Remove atores auxiliares antigos."""
    actors_to_remove = [
        'esfera_max_deslocamento',
        'rotulo_max_deslocamento'
    ]

    for name in actors_to_remove:
        try:
            widget.remove_actor(name)
        except KeyError:
            pass
