"""
Visualização refinada de apoios estruturais.
Simples, elegante e funcional.
"""
import numpy as np
import pyvista as pv


def plot_supports(plotter, estrutura, support_scale_factor):
    """
    Plota os apoios da estrutura com símbolos visuais distintos.

    Tipos reconhecidos:
    - Engaste: Cubo sólido (vermelho)
    - Rótula: Esfera (verde)
    - Restrições parciais: Cones e placas
    """
    # Coletar malhas por tipo
    meshes_engaste = []
    meshes_pino = []
    meshes_transla = []
    meshes_rotacao = []

    # Padrões de apoio conhecidos
    SET_ENCASTRE = {0, 1, 2, 3, 4, 5}
    SET_PINNED = {0, 1, 2}

    # Definir escala dos elements
    s = support_scale_factor
    size_block = s * 0.4
    size_joint = s * 0.3
    size_arrow = s * 0.35
    offset_base = s * 0.3

    # Direções dos eixos cartesianos
    axes = {
        0: np.array([1, 0, 0]),  # X
        1: np.array([0, 1, 0]),  # Y
        2: np.array([0, 0, 1])   # Z
    }

    # Processar cada nó da estrutura
    for idx in range(estrutura.num_nodes):
        no = estrutura.nodes[idx]
        center = np.array(no.coord)
        bc_set = no.boundary_conditions

        # Caso especial: Engaste completo
        if bc_set == SET_ENCASTRE:
            cube = pv.Cube(
                center=center,
                x_length=size_block,
                y_length=size_block,
                z_length=size_block
            )
            meshes_engaste.append(cube)
            continue

        # Caso especial: Rótula (apoio simples)
        if bc_set == SET_PINNED:
            sphere = pv.Sphere(
                radius=size_joint * 0.7,
                center=center,
                theta_resolution=16,
                phi_resolution=16
            )
            meshes_pino.append(sphere)
            continue

        # Restrições de translação (DOFs 0, 1, 2)
        for dof in range(3):
            if dof not in bc_set:
                continue

            axis_vec = axes[dof]
            offset_pos = center - axis_vec * offset_base

            # Cone apontando para o nó
            cone = pv.Cone(
                center=offset_pos,
                direction=axis_vec,
                height=size_arrow,
                radius=size_arrow * 0.25,
                resolution=8
            )
            meshes_transla.append(cone)

        # Restrições de rotação (DOFs 3, 4, 5)
        for dof in range(3, 6):
            if dof not in bc_set:
                continue

            axis_vec = axes[dof - 3]

            # Criar disco fino perpendicular ao eixo
            # O disco representa o plano bloqueado pela restrição
            offset_pos = center + axis_vec * (offset_base * 0.8)

            disc = pv.Disc(
                center=offset_pos,
                inner=0,
                outer=size_arrow * 0.45,
                normal=axis_vec,
                r_res=1,
                c_res=16
            )
            meshes_rotacao.append(disc)

    # Adicionar malhas ao plotter por categoria
    if meshes_engaste:
        combined = pv.MultiBlock(meshes_engaste).combine()
        plotter.add_mesh(
            combined,
            color='#dc2626',
            opacity=0.9,
            label='Engaste',
            name='apoios_engaste'
        )

    if meshes_pino:
        combined = pv.MultiBlock(meshes_pino).combine()
        plotter.add_mesh(
            combined,
            color='#16a34a',
            opacity=0.9,
            label='Rótula',
            name='apoios_rotula'
        )

    if meshes_transla:
        combined = pv.MultiBlock(meshes_transla).combine()
        plotter.add_mesh(
            combined,
            color='#2563eb',
            opacity=0.85,
            label='Restrição de Translação',
            name='apoios_translacao'
        )

    if meshes_rotacao:
        combined = pv.MultiBlock(meshes_rotacao).combine()
        plotter.add_mesh(
            combined,
            color='#ea580c',
            opacity=0.75,
            label='Restrição de Rotação',
            name='apoios_rotacao'
        )


def criar_apoio_roller(center, direction, size):
    """
    Cria um apoio móvel (roller) - livre para deslizar em uma direção.

    Retorna um cilindro pequeno que simboliza o rolamento.
    """
    # Normalizar direção
    direction = direction / np.linalg.norm(direction)

    # Cilindro pequeno representando o rolo
    cylinder = pv.Cylinder(
        center=center - np.array([0, 0, size * 0.3]),
        direction=[0, 1, 0],
        radius=size * 0.15,
        height=size * 0.4
    )

    return cylinder


def criar_apoio_elastico(center, stiffness_ratio, size):
    """
    Cria símbolo de apoio elástico (mola simplificada).

    Args:
        center: Posição do nó
        stiffness_ratio: Razão da rigidez (0-1, onde 1 é rígido)
        size: Escala do símbolo
    """
    # Criar zigzag simples representando mola
    n_segments = 4
    points = []

    z_start = center[2] - size * 0.4
    z_end = center[2]

    for i in range(n_segments + 1):
        t = i / n_segments
        z = z_start + (z_end - z_start) * t

        # Zigzag lateral
        x_offset = (size * 0.15) * (1 if i % 2 == 0 else -1)

        point = center + np.array([x_offset, 0, z - center[2]])
        points.append(point)

    # Criar linha conectando os pontos
    spline = pv.Spline(points, n_points=len(points) * 3)

    # Adicionar espessura à linha
    tube = spline.tube(radius=size * 0.02)

    return tube
