# Third-party libraries
import numpy as np
import pyvista as pv
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Local libraries
from .support_visualization import plot_supports


def plot_nodal_loads(plotter, estrutura, magnitude=1.0, num_legendas=8):
    """
    Plota as cargas concentradas nodes nós da estrutura.

    Args:
        plotter (pyvista.Plotter): Objeto Plotter do PyVista.
        estrutura (Estrutura): Instância da classe Structure.
        magnitude (float): Magnitude das flechas.
        num_legendas (int): Número de legendas para as flechas.
    """
    # Listas para armazenar as cargas concentradas
    pontos_inicio = []
    valores_cargas = []
    vetores_direcao = []

    # Iterar sobre cada objeto 'Node' na estrutura
    for i, load in estrutura.nodal_loads.items():
        for j in range(estrutura.dofs_per_node): # Itera sobre os 6 graus de liberdade
            node = estrutura.nodes[i]
            load_component = load[j]

            # Se o componente for não-nulo
            if abs(load_component) > 1e-9:
                # Adiciona o ponto de início da flecha (a coordenada do nó)
                pontos_inicio.append(node.coord)

                # Adiciona o valor escalar desta componente de carga
                valores_cargas.append(load_component)
                
                # Cria o vetor de direção para esta flecha
                if j < 3: # Se for Fx, Fy, ou Fz
                    vetor_direcao = np.zeros(3)
                    vetor_direcao[j] = np.sign(load_component)
                    vetores_direcao.append(vetor_direcao)
                else: # Se for Mx, My, ou Mz
                    vetor_direcao = np.zeros(3)
                    vetor_direcao[j-3] = np.sign(load_component)
                    vetores_direcao.append(vetor_direcao)

    if not pontos_inicio:
        return

    # Transformar as listas em arrays NumPy
    pontos_inicio = np.array(pontos_inicio)
    valores_cargas = np.array(valores_cargas)
    vetores_direcao = np.array(vetores_direcao)

    # Vetores de direção para cada flecha
    magnitudes_cargas = np.abs(valores_cargas)

    # Encontrar valores únicos e arredondados para agrupar
    valores_unicos = np.unique(np.round(magnitudes_cargas, 5))
    
    # Limitar o número de entradas na legenda para não poluir a tela
    if len(valores_unicos) > num_legendas:
        valores_unicos = np.linspace(valores_unicos.min(), valores_unicos.max(), num_legendas)

    colors = plt.cm.get_cmap("turbo", len(valores_unicos))
    legend_entries = []
    
    # Plotar um glifo para cada categoria de cor
    flecha_template = pv.Arrow()
    for i, value in enumerate(valores_unicos):
        color = colors(i)
        
        # Encontrar quais flechas pertencem a esta categoria
        if len(valores_unicos) == num_legendas and num_legendas > 1:
            lim_inf = value - (valores_unicos[1] - valores_unicos[0]) / 2
            lim_sup = value + (valores_unicos[1] - valores_unicos[0]) / 2
            mask = (magnitudes_cargas >= lim_inf) & (magnitudes_cargas < lim_sup)
        else:
            mask = np.round(magnitudes_cargas, 5) == value
        
        if not np.any(mask):
            continue

        # Criar um PolyData apenas com os dados desta categoria
        malha_pontos = pv.PolyData(pontos_inicio[mask])
        malha_pontos['vetores'] = vetores_direcao[mask]
        malha_pontos['magnitude_carga'] = magnitudes_cargas[mask]

        # Criar o glifo para este grupo
        malha_flechas = malha_pontos.glyph(
            orient='vetores',
            scale=False,
            factor=magnitude,
            geom=flecha_template
        )
        
        # Adicionar ao plotter
        plotter.add_mesh(malha_flechas, color=color, name=f"cargas_concentradas_{value:.3f}")
        legend_entries.append([f"{value:.3f} kN", color])

    # Adicionar a legenda
    if legend_entries:
        plotter.add_legend(legend_entries, bcolor=(0.9, 0.9, 0.9), loc='lower right', size=(0.15, 0.15), border=True)


def plot_distributed_loads(plotter, estrutura, magnitude=1.0, num_flechas=5, num_legendas=8):
    """
    Plotar as cargas distribuídas (PyVista)

    Args:
        plotter (pyvista.Plotter): Objeto Plotter do PyVista.
        estrutura (Estrutura): Estrutura a ser plotada.
        P (np.ndarray): Matriz de cargas concentradas.
        magnitude (float): Magnitude das flechas.
    """
    # Listas para armazenar os dados das flechas
    pontos = []
    vetores = []
    cargas_escalares = []

    # Coletar os dados de todas as flechas
    for elemento in estrutura.original_members.values():
        if elemento.get('distributed_load') is not None:
            n1, n2 = elemento['nodes'][0], elemento['nodes'][-1]
            q = elemento['distributed_load']

            # Pular elements sem nenhuma carga distribuída
            if not np.any(q):
                continue
            
            # Pontos iniciais e finais do elemento
            pt1 = n1.coord
            pt2 = n2.coord
            delta = pt2 - pt1

            # Processar cada eixo (x, y, z)
            for k in range(3): # k=0 (x), k=1 (y), k=2 (z)
                q1, q2 = q[0][k], q[1][k]
                if q1 == 0 and q2 == 0:
                    continue

                # Gerar pontos e cargas para as flechas deste elemento
                segmentos = np.linspace(0, 1, num_flechas)
                pontos_no_elemento = pt1 + np.outer(segmentos, delta)
                cargas_no_elemento = q1 + segmentos * (q2 - q1)

                # Vetor de direção base para este eixo
                vetor_direcao_base = np.zeros(3)
                vetor_direcao_base[k] = 1.0

                # Criar os vetores de direção e magnitude para cada flecha
                sinais = np.sign(cargas_no_elemento)
                vetores_flechas = np.outer(sinais, vetor_direcao_base)

                # Adicionar os dados gerados às listas principais
                pontos.append(pontos_no_elemento)
                vetores.append(vetores_flechas)
                cargas_escalares.append(np.abs(cargas_no_elemento))

    # Se nenhuma carga foi encontrada, encerrar a função
    if not pontos:
        return

    # Criar um único objeto PolyData com todos os pontos e vetores
    pontos_finais = np.vstack(pontos)
    vetores_finais = np.vstack(vetores)
    magnitudes_cargas  = np.concatenate(cargas_escalares)

    # Criar categorias de cores e a legenda
    valores_unicos = np.unique(np.round(magnitudes_cargas, 5))
    if len(valores_unicos) > num_legendas:
        valores_unicos = np.linspace(valores_unicos.min(), valores_unicos.max(), num_legendas)

    colors = plt.cm.get_cmap("jet", len(valores_unicos))
    legend_entries = []
    
    # Plotar um glifo para cada categoria de cor
    flecha_template = pv.Arrow()
    for i, value in enumerate(valores_unicos):
        color = colors(i)
        
        if len(valores_unicos) == num_legendas and num_legendas > 1:
            lim_inf = value - (valores_unicos[1] - valores_unicos[0]) / 2
            lim_sup = value + (valores_unicos[1] - valores_unicos[0]) / 2
            mask = (magnitudes_cargas >= lim_inf) & (magnitudes_cargas < lim_sup)
        else:
            mask = np.round(magnitudes_cargas, 5) == value
        
        if not np.any(mask):
            continue

        malha_pontos = pv.PolyData(pontos_finais[mask])
        malha_pontos['vetores'] = vetores_finais[mask]

        malha_flechas = malha_pontos.glyph(
            orient='vetores', scale=False, factor=magnitude, geom=flecha_template
        )
        
        plotter.add_mesh(malha_flechas, color=color, name=f"cargas_distribuidas_{value:.3f}")
        legend_entries.append([f"{value:.3f} kN/m", color])

    # Adicionar a legenda manual
    if legend_entries:
        plotter.add_legend(legend_entries, bcolor=(0.9, 0.9, 0.9), loc='lower right', size=(0.15, 0.15), border=True)


def plot_structure(plotter, estrutura, malha, 
                   transparencia=1.0, 
                   plotar_secao=True, 
                   plotar_cargas=True, 
                   plotar_nos=True, 
                   name=None,
                   arrow_scale_factor: float = 0.50,
                   node_scale_factor: float = 0.05,
                   support_scale_factor: float = 0.40,
                   min_font_size: int = 12,
                   max_font_size: int = 24):
    """
    Função para plotar a estrutura indeformed (PyVista)

    Args:
        plotter (QtInteractor): Objeto Plotter do PyVista.
        estrutura (Estrutura): Estrutura a ser plotada.
        malha (dict): Dicionário contendo as malhas da estrutura.
        transparencia (float): Transparencia das seções transversais.
        plotar_secao (bool): Flag para plotar a seção transversal.
        plotar_cargas (bool): Flag para plotar as cargas.
        plotar_nos (bool): Flag para plotar os nós.
        name (str): Nome da malha.
    """
    # Calcular o tamanho característico da estrutura
    coords = np.array([no.coord for no in estrutura.original_nodes.values()])
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    bbox_diag = np.linalg.norm(bbox_max - bbox_min)

    # Evitar estruturas de tamanho zero
    if bbox_diag < 1e-6:
        bbox_diag = 1.0

    elementos = list(estrutura.original_members.values())
    lengths = []
    for elemento in elementos:
        n1, n2 = elemento['nodes'][0], elemento['nodes'][-1]
        lengths.append(np.linalg.norm(n2.coord - n1.coord))

    mediana = np.percentile(lengths, 50)

    base_value = 0.40 * mediana
    lower_limit = 0.020 * bbox_diag
    upper_limit = 0.25 * bbox_diag

    scale = np.clip(base_value, lower_limit, upper_limit)

    # Calcular escalas proporcionais
    arrow_scale = scale * arrow_scale_factor
    node_radius = scale * node_scale_factor
    support_scale = scale * support_scale_factor

    # Desempacotar os resultados em listas separadas
    malhas = malha['section']

    # Adicionar as malhas à estrutura
    # plotter.add_mesh(tubos.combine(), color='gray', opacity=transparencia, name=f'{name}_tubos')

    # Plotar a seção transversal
    if plotar_secao:
        plotter.add_mesh(
            malhas.combine(),
            color='lightblue',
            edge_color='gray',
            opacity=transparencia,
            name=f'{name}_secoes'
        )
    
    # Plotar os nós
    if plotar_nos:
        coord_nos = np.array([no.coord for no in estrutura.original_nodes.values()])

        # Representar os nós por esferas
        pontos_nos = pv.PolyData(coord_nos)

        # Definir a geometry_type do glyph (uma esfera)
        glyph_esfera = pv.Sphere(radius=float(node_radius), theta_resolution=12, phi_resolution=12)

        # Definir o glyph
        glyph = pontos_nos.glyph(scale=False, orient=False, geom=glyph_esfera)

        # Adicionar o glyph à cena
        plotter.add_mesh(
            glyph,
            style='surface',
            color='yellow',
            name=f'{name}_glyphs'
        )

        # Adicionar os rótulos dos nós
        plotter.add_point_labels(
            coord_nos,
            [str(i + 1) for i in estrutura.original_nodes.keys()],
            font_size=20,
            text_color='black',
            shape=None,
            always_visible=True,
            name=f'{name}_nos'
        )

    # Plotar os apoios
    plot_supports(plotter, estrutura, support_scale)

    # Plotar as cargas (caso seja solicitado)
    if plotar_cargas:
        plot_nodal_loads(plotter, estrutura, magnitude=float(arrow_scale))
        plot_distributed_loads(plotter, estrutura, magnitude=float(arrow_scale))
    
    plotter.add_axes()


def plotar_reacoes(plotter, reacoes_nos_apoios, reacoes_a_plotar=None, magnitude_escala=1.00, offset_dist=1.00):
    """
    Plota as reações de apoio (forças e momentos).

    Args:
        plotter (pyvista.Plotter): Objeto Plotter do PyVista.
        reacoes_nos_apoios (dict): Dicionário retornado pela função `reacoes_apoios`.
        magnitude_escala (float): Fator de escala global para o tamanho das setas.
        offset_dist (float): Distância para afastar a base da seta do nó de apoio.
    """
    # Remove os atores de reação antes de qualquer outra lógica
    plotter.remove_actor('reacoes_forcas', render=False)
    plotter.remove_actor('reacoes_momentos', render=False)

    # Se nenhuma reação foi solicitada após a limpeza, apenas renderiza a cena vazia e sai
    if not reacoes_a_plotar:
        plotter.render()
        return
    
    # Salvar o estado atual da câmera
    camera_state = plotter.camera.copy()

    # Obter os vetores de reação
    Ra = reacoes_nos_apoios['R']
    coords = reacoes_nos_apoios['coords']

    # Vetores completos de forças e momentos
    forcas = Ra[:, :3, 0]
    momentos = Ra[:, 3:, 0]

    # Máscaras para filtrar os componentes desejados
    mascara_forcas = np.array(['rx', 'ry', 'rz'])
    mascara_momentos = np.array(['rmx', 'rmy', 'rmz'])
    sel_forcas = np.isin(mascara_forcas, reacoes_a_plotar)
    sel_momentos = np.isin(mascara_momentos, reacoes_a_plotar)

    # Filtrar os vetores
    forcas_filtradas = forcas * sel_forcas
    momentos_filtrados = momentos * sel_momentos

    # Plotar Forças (Setas Simples)
    _plotar_vetores_nodais(
        plotter, coords, forcas_filtradas, magnitude_escala, offset_dist,
        template_glyph=pv.Arrow(shaft_radius=0.02, tip_radius=0.04, scale=0.8),
        unidade_legenda='kN',
        nome_ator='reacoes_forcas',
        color='crimson'
    )

    # Plotar Momentos (Setas Duplas)
    arrow1 = pv.Arrow(shaft_radius=0.02, tip_radius=0.04, scale=0.8)
    arrow2 = pv.Arrow(shaft_radius=0.02, tip_radius=0.04, scale=0.8).translate((0.15, 0, 0), inplace=False)
    template_momento = arrow1 + arrow2

    _plotar_vetores_nodais(
        plotter, coords, momentos_filtrados, magnitude_escala, 2 * offset_dist,
        template_glyph=template_momento,
        unidade_legenda='kN.m',
        nome_ator='reacoes_momentos',
        color='dodgerblue'
    )

    # Restaurar o estado da câmera
    plotter.camera = camera_state


def adicionar_escalares(grid, grid_combinado, valores_iniciais=None, valores_finais=None):
    """Calcular todos os escalares de uma única vez e depois os atribui."""
    if grid is None or grid.n_blocks == 0:
        return

    # Adicionar valores fictícios à malha, caso os valores fornecidos sejam None
    if valores_iniciais is None:
        valores_iniciais = np.zeros(grid.n_blocks)
    if valores_finais is None:
        valores_finais = np.ones(grid.n_blocks)

    # Coletar o número de células de cada seção
    n_cells_por_secao = np.array([secao.n_cells for secao in grid])

    # Cria um array de "rampas" (sequências de 0 a 1) para todas as seções
    ramps = np.r_[tuple(np.linspace(0, 1, n) for n in n_cells_por_secao)]

    # Repete os valores iniciais e finais para corresponder ao tamanho total de 'rampas'
    starts_repetidos = np.repeat(valores_iniciais, n_cells_por_secao)
    fins_repetidos = np.repeat(valores_finais, n_cells_por_secao)

    # Calcular todos os escalares
    todos_os_escalares = starts_repetidos + (fins_repetidos - starts_repetidos) * ramps

    grid_combinado.cell_data['scalars'] = todos_os_escalares

def _plotar_vetores_nodais(plotter, coords, valores, magnitude, offset,
                          template_glyph, unidade_legenda, nome_ator, color):
    """
    Função auxiliar genérica para plotar vetores (forças ou momentos) nodes nós.
    """
    # Definir uma tolerância para evitar plotagem nula
    tolerance = 1e-6

    # Encontra os índices de todos os valores com magnitude maior que a tolerância
    indices_nos, indices_eixos = np.where(np.abs(valores) > tolerance)
    
    if indices_nos.size == 0:
        return

    # Coleta os dados necessários
    pontos_apoio = coords[indices_nos]
    valores_vetores = valores[indices_nos, indices_eixos]

    # Obtem os vetores de direção
    num_vetores = len(indices_nos)
    vetores_direcao = np.zeros((num_vetores, 3))
    vetores_direcao[np.arange(num_vetores), indices_eixos] = np.sign(valores_vetores)
    
    # Afasta a base da seta do nó na direção oposta ao vetor
    pontos_inicio = pontos_apoio - offset * vetores_direcao

    # Criar um único PolyData com todos os pontos e dados
    malha_pontos = pv.PolyData(pontos_inicio)
    malha_pontos['vetores'] = vetores_direcao

    # Criar um único glifo para todas as setas
    malha_flechas = malha_pontos.glyph(
        orient='vetores',
        scale=False,
        factor=magnitude,
        geom=template_glyph
    )
    
    # Adicionar a malha de setas com uma cor sólida
    plotter.add_mesh(malha_flechas, color=color, name=nome_ator)

    # Adicionar os rótulos de texto com os valores
    labels = [f"{np.abs(val):.2f} {unidade_legenda}" for val in valores_vetores]
    plotter.add_point_labels(
        pontos_inicio,
        labels,
        font_size=20,
        text_color=color,
        shape=None,
        always_visible=True,
        name=nome_ator
    )

def criar_secoes_base(estrutura):
    """
    Itera sobre os elements da estrutura para identificar seções únicas e
    criar um mapa de índices para cada elemento.

    Args:
        estrutura (Structure): Instância da classe Structure.

    Returns:
        tuple: Uma tupla contendo:
            - secoes_unicas (list): Uma lista de objetos pv.PolyData.
            - secoes_indices (np.ndarray): Um array onde o índice 'i' armazena o
                                          índice da seção única para o elemento 'i'.
    """
    secoes_unicas = []
    secoes_indices = np.full(estrutura.num_elements, -1, dtype=int)

    # Dicionário para mapear a "impressão digital" da geometry_type a um índice
    fingerprint_para_indice = {}
    
    # Iterar sobre os objetos 'Element'
    for elem_id, elemento in estrutura.elements.items():
        if elem_id >= len(secoes_indices):
            continue

        if not elemento.section:
            continue

        # Gerar a geometry_type do elemento
        polydata = elemento.section.generate_polydata()
        
        # Criar uma "impressão digital" única para esta geometry_type
        fingerprint = polydata.points.tobytes() + polydata.faces.tobytes()

        # Verificar se a impressão digital já foi armazenada
        if fingerprint in fingerprint_para_indice:
            indice = fingerprint_para_indice[fingerprint]
        else:
            # Se a impressão digital ainda não foi armazenada, cria um novo índice
            indice = len(secoes_unicas)

            # Armazena o novo índice no mapa
            fingerprint_para_indice[fingerprint] = indice

            # Adiciona o novo objeto PolyData à lista de únicos
            secoes_unicas.append(polydata)
        
        # Associar o índice da seção única ao elemento atual
        secoes_indices[elem_id] = indice
        
    return secoes_unicas, secoes_indices


def criar_secao_elemento(section, start_coord, extrude_vec, rot_matrix):
    """
    Função para criar a seção transversal da estrutura com o PyVista.
    Válida tanto para estrutura deformed como indeformed.
    """
    if section is None:
        return None
    
    # Rotate the section and translate to the start coordinate
    points_transformed = section.points @ rot_matrix.T + start_coord
    
    # Create new PolyData from the transformed points
    section_transformed = pv.PolyData(points_transformed, section.faces)
    
    # Extrude the section along the extrusion vector
    try:
        mesh = section_transformed.extrude(extrude_vec, capping=True)
    except Exception:
        return None
    
    return mesh


def matriz_rotacao_malha(L_vec, ref_vector):
    """
    Função para obter a matriz de transformação para plotagem
    """
    # Normalize the extrusion vector, L_vec
    L_norm = np.linalg.norm(L_vec)
    if L_norm < 1e-10:
        return np.eye(3)
    
    # 
    x_ = L_vec / L_norm
    
    # Calcular y_ com verificação de colinearidade
    y_ = np.cross(ref_vector, x_)
    y_norm = np.linalg.norm(y_)
    
    if y_norm < 1e-10:
        # Vetores colineares, usar vetor alternativo
        alt_vector = np.array([0, 0, 1]) if abs(x_[2]) < 0.9 else np.array([1, 0, 0])
        y_ = np.cross(alt_vector, x_)
        y_norm = np.linalg.norm(y_)
    
    y_ /= y_norm
    z_ = np.cross(x_, y_)
    
    return np.stack((y_, z_, x_), axis=-1)


def precalcular_matrizes_rotacao(coords, ref_vector, elementos):
    """
    Pré-calcula todas as matrizes de rotação usando operações vetorizadas.
    
    Args:
        coords: Lista ou array de coordenadas [coord1, coord2] para cada elemento
        ref_vector: Vetor de referência para cada elemento
        elementos: Número de elements
    
    Returns:
        matrizes: Lista de matrizes de rotação
    """
    rot_matrices = []
    
    for i in range(elementos):
        # Get the first and last coordinates
        coord1 = coords[i][0]
        coord2 = coords[i][-1]

        # Calculate the vector between the first and last coordinates
        L_vec = coord2 - coord1
        ref_vec = ref_vector[i]
        
        # Calculate the rotation matrix
        matriz = matriz_rotacao_malha(L_vec, ref_vec)
        rot_matrices.append(matriz)
    
    return rot_matrices

def processar_elemento(args):
    """
    Processa um único elemento de forma otimizada.
    
    Otimizações:
    - Usa matriz de rotação pré-calculada
    - Reutiliza geometry_type de seções idênticas
    - Cria tubos com resolução reduzida
    """
    # Get element data
    coords, base_section, rot_matrices = args
    num_points = coords.shape[0]
    
    # Generate the line connecting the nodes of the element
    if num_points == 3:
        # Create a spline with 3 points
        line = pv.Spline(coords, n_points=20)
    else:
        # Create a normal line with 2 points
        line = pv.Line(pointa=coords[0], pointb=coords[-1])
    
    # Create a tube from the line
    tube = line.tube(radius=0.01, n_sides=8)

    # Extrude the base section
    if base_section is None:
        return tube, None
    mesh = None 

    if num_points == 2:
        # Extrude from initial coord to final coord
        L_vec = coords[-1] - coords[0]
        mesh = criar_secao_elemento(base_section, coords[0], L_vec, rot_matrices)

    else:
        # Segmented extrusion from spline
        blocks = pv.MultiBlock()

        # Iterate over the three points
        for i in range(num_points - 1):
            p_start = coords[i]
            p_end = coords[i + 1]
            p_vec = p_end - p_start

            # Extrude this segment
            seg_mesh = criar_secao_elemento(base_section, p_start, p_vec, rot_matrices)

            if seg_mesh is not None:
                blocks.append(seg_mesh)

        # Merge the segments
        mesh = blocks.combine()
    
    return tube, mesh

def generate_mesh(
        structure, 
        sections, 
        sections_idx, 
        coords, 
        ref_vector, 
        geometry, 
        **kwargs):
    """
    Função para criar a malha da estrutura deformed (PyVista).

    Args:
        estrutura (Structure): Instância da classe Structure
        secoes ():
        coords (np.ndarray): Coordenadas da estrutura
        ref_vector: Vetor de referência para a estrutura
        geometria (str): Geometria da estrutura (deformed ou indeformed)

    Returns:
        tubos_def: Dicionário com os tubos da estrutura deformed, separados por tipo de análise
        secao_def: Dicionário com as malhas da estrutura deformed, separadas por tipo de análise
    """
    # Dados iniciais
    num_elementos = structure.num_elements if geometry == 'deformed' else len(structure.original_members)
    autovalores = kwargs.get('autovalores', np.array([]))
    num_modos = autovalores.shape[0] if autovalores.size > 0 else 0
    num_indices = len(sections_idx)
    k_vec = ref_vector[geometry]

    # Alinhamento da malha undeformed
    if num_indices != num_elementos:
        if num_indices > num_elementos:
            ratio = num_indices / num_elementos
            if ratio.is_integer():
                step = int(ratio)
                sections_idx = sections_idx[::step]
            else:
                sections_idx = sections_idx[:num_elementos]

    # Definir os "trabalhos" a serem processados
    jobs = []

    # Verify if the analysis is buckling
    if structure.is_buckling and geometry == 'deformed' and num_modos > 0:
        for modo in range(num_modos):
            jobs.append({
                'coords': coords[modo],
                'matrizes_rotacao': precalcular_matrizes_rotacao(coords[modo], k_vec, num_elementos),
                'label': f"Buckling mode {modo + 1}"
            })
    else:
        jobs.append({
            'coords': coords,
            'matrizes_rotacao': precalcular_matrizes_rotacao(coords, k_vec, num_elementos),
            'label': f"{geometry.capitalize()} mesh"
        })

    # Inicializar estrutura de dados
    resultados_finais_tubos = []
    resultados_finais_secoes = []

    for job in jobs:
        # Prepara a lista de tarefas para o Parallel
        tasks = [
            (job['coords'][i], 
             sections[sections_idx[i]] if sections_idx[i] >= 0 else None,
             job['matrizes_rotacao'][i])
            for i in range(num_elementos)
        ]
        
        # Chama a paralelização usando o backend de multiprocessamento (n_jobs=-1)
        resultados_paralelos = Parallel(n_jobs=-1)(
            delayed(processar_elemento)(task) 
            for task in tqdm(tasks, desc=f"Processing {job['label']}")
        )
        
        # Desempacota os resultados
        tubos_job = []
        secoes_job = []

        for res in resultados_paralelos:
            if res[0] is not None:
                tubos_job.append(res[0])
            if res[1] is not None:
                secoes_job.append(res[1])
        
        resultados_finais_tubos.append(tubos_job)
        resultados_finais_secoes.append(secoes_job)
        
    # Se a lista de resultados tem apenas um item (não é flambagem), simplifica a saída
    if len(resultados_finais_tubos) == 1:
        resultados_finais_tubos = pv.MultiBlock(resultados_finais_tubos[0])
        resultados_finais_secoes = pv.MultiBlock(resultados_finais_secoes[0])
    else:
        for modo in range(num_modos):
            resultados_finais_tubos[modo] = pv.MultiBlock(resultados_finais_tubos[modo])
            resultados_finais_secoes[modo] = pv.MultiBlock(resultados_finais_secoes[modo])
        
    return {
        'tubos': resultados_finais_tubos,
        'section': resultados_finais_secoes
    }
