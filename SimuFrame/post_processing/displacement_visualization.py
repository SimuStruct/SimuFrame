# Third-party libraries
import numpy as np
import pyvista as pv

# Local libraries
from .visualization import adicionar_escalares


def plotar_deslocamentos(estrutura, malha_deformada, coords_deformadas,
                         deformacao_global, widget, **kwargs):
    """
    Plota os deslocamentos da estrutura com colormap.

    Args:
        estrutura: Objeto Structure
        malha_deformada: Dicionário com malhas {'tubos': ..., 'section': ...}
        coords_deformadas: Coordenadas dos pontos deformados
        deformacao_global: Array com deslocamentos [elements, nós, DOFs]
        widget: Plotter PyVista
        **kwargs: modo, componente, grid_secao
    """
    # Extrair parâmetros
    modo = kwargs.get('modo', 0)
    componente = kwargs.get('componente', 'u')
    # secoes_mb = kwargs.get('grid_secao', None)

    # Salvar câmera
    camera_state = widget.camera.copy()

    # Mapa de componentes
    componentes_disponiveis = {
        'u': (slice(None), 'U (m)', 'Deslocamento Total'),
        'x': (0, 'UX (m)', 'Deslocamento em X'),
        'y': (1, 'UY (m)', 'Deslocamento em Y'),
        'z': (2, 'UZ (m)', 'Deslocamento em Z'),
        'θx': (3, 'Rx (rad)', 'Rotação em X'),
        'θy': (4, 'Ry (rad)', 'Rotação em Y'),
        'θz': (5, 'Rz (rad)', 'Rotação em Z')
    }

    if componente not in componentes_disponiveis:
        raise ValueError(f"Componente '{componente}' não suportado. Use: {list(componentes_disponiveis.keys())}")

    eixo_idx, title_label, _ = componentes_disponiveis[componente]

    # Verificar se precisa recalcular malha (apenas para flambagem)
    needs_recalc = _needs_mesh_recalculation(widget, estrutura, modo)

    if needs_recalc:
        _plot_new_displacement_mesh(
            widget, estrutura, malha_deformada, deformacao_global,
            coords_deformadas, componente, eixo_idx, title_label, modo
        )
    else:
        _update_existing_displacement_mesh(
            widget, deformacao_global, malha_deformada, componente,
            eixo_idx, title_label, modo, estrutura
        )

    # Adicionar marcador de deslocamento máximo
    _add_max_displacement_marker(
        widget, deformacao_global, coords_deformadas,
        componente, eixo_idx, modo, estrutura
    )

    # Restaurar câmera
    widget.camera = camera_state


def _needs_mesh_recalculation(widget, estrutura, modo):
    """Verifica se precisa recalcular a malha."""
    if not estrutura.is_buckling:
        return not _mesh_exists(widget)

    # Para flambagem, verificar mudança de modo
    if not hasattr(widget, 'estado_anterior'):
        return True

    return widget.estado_anterior.get('modo') != modo


def _mesh_exists(widget):
    """Verifica se a malha deformada já existe."""
    return ('malha_deformada_tubos' in widget.actors and
            'malha_deformada_secoes' in widget.actors)


def _plot_new_displacement_mesh(widget, estrutura, malha_deformada, deformacao_global,
                                coords_deformadas, componente, eixo_idx, title_label, modo):
    """Cria e plota nova malha de deslocamentos."""

    # Limpar apenas atores de resultados
    _clear_result_actors(widget)

    # Extrair valores de deslocamento
    valores = _extract_displacement_values(
        deformacao_global, componente, eixo_idx, modo, estrutura
    )

    # Limites do colormap
    valores_iniciais = valores[:, 0]
    valores_finais = valores[:, -1]
    vmin, vmax = valores.min(), valores.max()

    # Extrair malhas
    tubos = malha_deformada['tubos'] if not estrutura.is_buckling else malha_deformada['tubos'][modo]
    secoes = malha_deformada['section'] if not estrutura.is_buckling else malha_deformada['section'][modo]

    # Combinar malhas
    grid_tubos = tubos.combine()
    grid_secao = secoes.combine()

    # Adicionar escalares
    adicionar_escalares(secoes, grid_secao, valores_iniciais, valores_finais)

    # Configurar barra de escalares
    scalar_bar_args = {
        'title': title_label,
        'title_font_size': 20,
        'label_font_size': 16,
        'n_labels': 10,
        'vertical': True,
        'fmt': '%.3e'
    }

    # Adicionar ao plotter
    widget.add_mesh(
        grid_tubos,
        color='sienna',
        opacity=1.0,
        name='malha_deformada_tubos'
    )
    widget.add_mesh(
        grid_secao,
        scalars='scalars',
        cmap="turbo",
        clim=(vmin, vmax),
        scalar_bar_args=scalar_bar_args,
        name='malha_deformada_secoes'
    )

    # Salvar estado (para flambagem)
    if estrutura.is_buckling:
        widget.estado_anterior = {'modo': modo}


def _update_existing_displacement_mesh(widget, deformacao_global, malha_deformada,
                                       componente, eixo_idx, title_label, modo, estrutura):
    """Atualiza malha existente com novos dados."""
    # Garantir visibilidade
    widget.actors['malha_deformada_tubos'].visibility = True
    widget.actors['malha_deformada_secoes'].visibility = True

    # Extrair valores
    valores = _extract_displacement_values(
        deformacao_global, componente, eixo_idx, modo, estrutura
    )

    # Limites
    valores_iniciais = valores[:, 0]
    valores_finais = valores[:, -1]
    vmin, vmax = valores.min(), valores.max()

    # Obter ator e dataset
    secoes = malha_deformada['section'] if not estrutura.is_buckling else malha_deformada['section'][modo]
    secao_actor = widget.actors['malha_deformada_secoes']
    grid_secao = secao_actor.mapper.dataset

    # Atualizar escalares
    adicionar_escalares(secoes, grid_secao, valores_iniciais, valores_finais)

    # Atualizar mapper
    secao_actor.mapper.scalar_range = (vmin, vmax)

    # Atualizar barra de escalares
    widget.update_scalar_bar_range([vmin, vmax])

    if widget.scalar_bars:
        widget.scalar_bar.SetTitle(title_label)
    else:
        widget.add_scalar_bar(
            title=title_label,
            title_font_size=20,
            label_font_size=16,
            n_labels=10,
            vertical=True,
            fmt='%.3e'
        )


def _extract_displacement_values(deformacao_global, componente, eixo_idx, modo, estrutura):
    """
    Extrai valores de deslocamento baseado no componente.

    Returns:
        np.ndarray: Valores de deslocamento [elements, nós]
    """
    if componente == 'u':
        # Deslocamento total (norma das translações)
        if estrutura.is_buckling:
            desl_trans = deformacao_global[modo, :, :, :3]
        else:
            desl_trans = deformacao_global[:, :, :3]

        valores = np.linalg.norm(desl_trans, axis=2)
    else:
        # Componente específica
        if estrutura.is_buckling:
            valores = deformacao_global[modo, :, :, eixo_idx]
        else:
            valores = deformacao_global[:, :, eixo_idx]

    return valores


def _add_max_displacement_marker(widget, deformacao_global, coords_deformadas,
                                 componente, eixo_idx, modo, estrutura):
    """Adiciona marcador visual no ponto de deslocamento máximo."""

    # Remover marcadores antigos
    for actor_name in ['esfera_max_deslocamento', 'rotulo_max_deslocamento']:
        try:
            widget.remove_actor(actor_name)
        except KeyError:
            pass

    # Extrair valores
    valores = _extract_displacement_values(
        deformacao_global, componente, eixo_idx, modo, estrutura
    )

    # Encontrar máximo (em valor absoluto)
    idx_max = np.unravel_index(np.abs(valores).argmax(), valores.shape)
    valor_max = valores[idx_max]

    # Obter posição
    if estrutura.is_buckling:
        pos_max = coords_deformadas[modo][idx_max[0], idx_max[1]]
    else:
        pos_max = coords_deformadas[idx_max[0], idx_max[1]]

    # Criar esfera no ponto máximo
    esfera = pv.Sphere(radius=0.05, center=pos_max)
    widget.add_mesh(
        esfera,
        color='black',
        opacity=0.8,
        name='esfera_max_deslocamento'
    )

    # Adicionar rótulo
    widget.add_point_labels(
        pos_max,
        [f'Max: {valor_max:.3e}'],
        font_size=28,
        text_color='red',
        shape_color='white',
        shape_opacity=0.8,
        margin=5,
        always_visible=True,
        name='rotulo_max_deslocamento'
    )


def _clear_result_actors(widget):
    """Remove atores antigos, exceto estrutura indeformada."""
    actors_to_keep = [
        'estrutura_indeformada_secoes', 'estrutura_indeformada_nos',
        'apoios_engaste', 'apoios_rotula', 'apoios_translacao', 'apoios_rotacao'
    ]
    actors_to_remove = [
        name for name in widget.actors.keys()
        if name not in actors_to_keep
    ]

    for name in actors_to_remove:
        widget.remove_actor(name)