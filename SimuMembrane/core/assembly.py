import numpy as np
import gmsh
import pyvista as pv
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

def shape_functions(element_type, xi, eta):
    """
    Funções de forma do elemento Q8.

    Args:
        element_type (str): Tipo do elemento (CST, QUAD, QUAD9, QUAD8).
        xi, eta: Coordenadas naturais (-1 a 1)

    Returns:
        N: Vetor (8,) com valores das funções de forma
    """
    if element_type == 'QUAD8':
        # Nós de canto
        N1 = -0.25 * (xi - 1) * (eta - 1) * (xi + eta + 1)
        N2 =  0.25 * (xi + 1) * (eta - 1) * (-xi + eta + 1)
        N3 =  0.25 * (xi + 1) * (eta + 1) * (xi + eta - 1)
        N4 = -0.25 * (xi - 1) * (eta + 1) * (-xi + eta - 1)

        # Nós intermediários
        N5 =  0.5 * (xi**2 - 1) * (eta - 1)
        N6 = -0.5 * (xi + 1) * (eta**2 - 1)
        N7 = -0.5 * (xi**2 - 1) * (eta + 1)
        N8 =  0.5 * (xi - 1) * (eta**2 - 1)

        return np.array([N1, N2, N3, N4, N5, N6, N7, N8])

    elif element_type == 'QUAD9':
        # Nós de canto
        N1 = xi * eta * (xi - 1) * (eta - 1)
        N2 = xi * eta * (xi + 1) * (eta - 1)
        N3 = xi * eta * (xi + 1) * (eta + 1)
        N4 = xi * eta * (xi - 1) * (eta + 1)

        # Nós intermediários
        N5 = -0.5 * eta * (eta - 1) * (xi**2 - 1)
        N6 =  0.5 * xi * (xi + 1) * (eta**2 - 1)
        N7 = -0.5 * eta * (eta + 1) * (xi**2 - 1)
        N8 =  0.5 * xi * (xi - 1) * (eta**2 - 1)

        # Nó central
        N9 = (xi**2 - 1) * (eta**2 - 1)

        return np.array([N1, N2, N3, N4, N5, N6, N7, N8, N9])

    else:
        raise ValueError(f"Element type {element_type} not supported.")


def shape_derivatives(element_type, xi, eta):
    """
    Derivadas das funções de forma.

    Returns:
        dN_dxi: Array (8,) - derivadas em relação a xi
        dN_deta: Array (8,) - derivadas em relação a eta
    """
    if element_type == 'QUAD':
        # Criar matriz de derivadas
        dN = np.zeros((2, 4))

        # Derivadas em relação a xi
        dN[0, 0] =  0.25 * (eta - 1)
        dN[0, 1] = -0.25 * (eta - 1)
        dN[0, 2] =  0.25 * (eta + 1)
        dN[0, 3] = -0.25 * (eta + 1)

        # Derivadas em relación a eta
        dN[1, 0] =  0.25 * (xi - 1)
        dN[1, 1] = -0.25 * (xi + 1)
        dN[1, 2] =  0.25 * (xi + 1)
        dN[1, 3] = -0.25 * (xi - 1)

    elif element_type == 'QUAD8':
        # Criar matriz de derivadas
        dN = np.zeros((2, 8))

        # Derivadas em relação a xi (dN/dξ)
        dN[0, 0] = -0.25 * (eta - 1) * (2 * xi + eta)
        dN[0, 1] = -0.25 * (eta - 1) * (2 * xi - eta)
        dN[0, 2] =  0.25 * (eta + 1) * (2 * xi + eta)
        dN[0, 3] =  0.25 * (eta + 1) * (2 * xi - eta)
        dN[0, 4] =  xi * (eta - 1)
        dN[0, 5] = -0.5 * (eta**2 - 1)
        dN[0, 6] = -xi * (eta + 1)
        dN[0, 7] =  0.5 * (eta**2 - 1)

        # Derivadas em relação a eta (dN/dη)
        dN[1, 0] = -0.25 * (xi - 1) * (xi + 2 * eta)
        dN[1, 1] =  0.25 * (xi + 1) * (-xi + 2 * eta)
        dN[1, 2] =  0.25 * (xi + 1) * (xi + 2 * eta)
        dN[1, 3] = -0.25 * (xi - 1) * (-xi + 2 * eta)
        dN[1, 4] =  0.5 * (xi**2 - 1)
        dN[1, 5] = -eta * (xi + 1)
        dN[1, 6] = -0.5 * (xi**2 - 1)
        dN[1, 7] =  eta * (xi - 1)
    
    elif element_type == 'QUAD9':
        # Criar matriz de derivadas
        dN = np.zeros((2, 9))

        # Derivadas em relação a xi (dN/dξ)
        dN[0, 0] =  0.25 * eta * (eta - 1) * (2 * xi - 1)
        dN[0, 1] =  0.25 * eta * (eta - 1) * (2 * xi + 1)
        dN[0, 2] =  0.25 * eta * (eta + 1) * (2 * xi + 1)
        dN[0, 3] =  0.25 * eta * (eta + 1) * (2 * xi - 1)
        dN[0, 4] =  xi * eta * (1 - eta)
        dN[0, 5] =  0.5 * (eta - 1) * (eta + 1) * (-2 * xi - 1)
        dN[0, 6] = -xi * eta * (eta + 1)
        dN[0, 7] =  0.5 * (eta - 1) * (eta + 1) * (1 - 2 * xi)
        dN[0, 8] =  2 * xi * (eta**2 - 1)

        # Derivadas em relação a eta (dN/dη)
        dN[1, 0] =  0.25 * xi * (xi - 1) * (2 * eta - 1)
        dN[1, 1] =  0.25 * xi * (xi + 1) * (2 * eta - 1)
        dN[1, 2] =  0.25 * xi * (xi + 1) * (2 * eta + 1)
        dN[1, 3] =  0.25 * xi * (xi - 1) * (2 * eta + 1)
        dN[1, 4] =  0.5 * (xi - 1) * (xi + 1) * (1 - 2 * eta)
        dN[1, 5] =  -xi * eta * (xi + 1)
        dN[1, 6] =  0.5 * (xi - 1) * (xi + 1) * (-2 * eta - 1)
        dN[1, 7] =  xi * eta * (1 - xi)
        dN[1, 8] =  2 * eta * (xi**2 - 1)

    else: 
        raise ValueError(f"Element type {element_type} not supported")

    return dN


def constitutive_matrix(num_elements, E, nu):
    """Matriz constitutiva para estado plano de tensão."""
    C = np.zeros((num_elements, 3, 3))

    # Matriz constitutiva para estado plano de tensão
    factor = E / (1 - nu ** 2)
    C_bulk = factor * np.array([[1, nu, 0],
                                [nu, 1, 0],
                                [0, 0, (1 - nu) / 2]])
    C[:] = C_bulk

    return C


def gauss_quadrature(element_type):
    """Retorna pontos e pesos de Gauss para integração 3x3."""
    gauss_map = {
        'QUAD': 2,
        'QUAD8': 3,
        'QUAD9': 3
    }

    # Number of Gauss points
    num_points = gauss_map[element_type]

    # Pontos e pesos de Gauss
    g, w = np.polynomial.legendre.leggauss(num_points)

    # Gera produto cartesiano (ξ, η)
    xi, eta = np.meshgrid(g, g)
    wi, wj = np.meshgrid(w, w)

    # Bidimensional Gauss Quadrature
    points = np.vstack([xi.ravel(), eta.ravel()]).T
    weights = (wi * wj).ravel()

    return points, weights


def jacobian(dN, coords):
    """
    Calcula a matriz Jacobiana.

    Returns:
        J: Matriz (2x2) Jacobiana
        detJ: Determinante da Jacobiana
    """
    # Calculate Jacobian
    J = dN @ coords

    # Calculate determinant
    detJ = np.linalg.det(J)

    return J, detJ


def B_matrix(element_type, num_elements, coords, xi=None, eta=None):
    """
    Matriz de deformação-deslocamento B.

    Returns:
        B: Matriz (3x16)
    """
    # Inicializar área do elemento
    area = np.zeros((num_elements, 1, 1))

    if element_type == 'CST':
        if coords.shape != (num_elements, 3, 2):
            raise ValueError(f"Coordenadas para CST devem ter shape (3, 2), mas têm {coords.shape}")

        # Extrair as coordenadas
        x1 = coords[:, 0, 0]
        y1 = coords[:, 0, 1]
        x2 = coords[:, 1, 0]
        y2 = coords[:, 1, 1]
        x3 = coords[:, 2, 0]
        y3 = coords[:, 2, 1]

        # Coeficientes da matriz [B]
        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2

        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1

        # Calcular a área do elemento
        area = 0.5 * (x1 * b1 + x2 * b2 + x3 * b3)

        if np.any(area <= 0):
            # Nós podem estar em sentido horário ou colineares
            raise ValueError("Área do elemento T3 é zero ou negativa. Verifique a ordem dos nós.")

        # Fator de multiplicação da matriz [B]
        fator = (1.0 / (2.0 * area))[:, np.newaxis]

        # Montagem da matriz [B]
        B = np.zeros((num_elements, 3, 6))
        B[:, 0, [0, 2, 4]] = fator * np.stack((b1, b2, b3), axis=1)
        B[:, 1, [1, 3, 5]] = fator * np.stack((c1, c2, c3), axis=1)
        B[:, 2, [0, 1, 2, 3, 4, 5]] = fator * np.stack((c1, b1, c2, b2, c3, b3), axis=1)

        # Determinante do Jacobiano (constante)
        detJ = 2.0 * area

    elif element_type == 'QUAD':
        if xi is None or eta is None:
            raise ValueError("xi e eta são necessários para QUAD8")
        
        # Derivadas das funções de forma (2, 4)
        dN_local = shape_derivatives(element_type, xi, eta)

        # Expandir para todos os elements (num_elements, 2, 4)
        dN = np.tile(dN_local, (num_elements, 1, 1))

        # Calculate Jacobian
        J, detJ = jacobian(dN, coords)
        inv_J = np.linalg.inv(J)
        J2 = np.kron(np.eye(2), inv_J)

        # Montar matriz [DN]
        DN = np.zeros((num_elements, 4, 8))
        DN[:, 0, 0::2] = dN[:, 0, :]  # ∂N/∂ξ em u
        DN[:, 1, 0::2] = dN[:, 1, :]  # ∂N/∂η em u
        DN[:, 2, 1::2] = dN[:, 0, :]  # ∂N/∂ξ em v
        DN[:, 3, 1::2] = dN[:, 1, :]  # ∂N/∂η em v

        # Montar matriz [H]
        M = np.array([[1, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 1, 1, 0]])
        H = np.tile(M, (num_elements, 1, 1))

        # Montar matriz [B]
        B = H @ J2 @ DN

    elif element_type == 'QUAD8':
        if xi is None or eta is None:
            raise ValueError("xi e eta são necessários para QUAD8")

        # Derivadas das funções de forma
        dN_local = shape_derivatives(element_type, xi, eta)

        # Expandir para todos os elements
        dN = np.tile(dN_local, (num_elements, 1, 1))

        # Calculate Jacobian
        J, detJ = jacobian(dN, coords)
        inv_J = np.linalg.inv(J)
        J2 = np.kron(np.eye(2), inv_J)

        DN = np.zeros((num_elements, 4, 16))
        DN[:, 0, 0::2] = dN[:, 0, :]  # ∂N/∂ξ em u
        DN[:, 1, 0::2] = dN[:, 1, :]  # ∂N/∂η em u
        DN[:, 2, 1::2] = dN[:, 0, :]  # ∂N/∂ξ em v
        DN[:, 3, 1::2] = dN[:, 1, :]  # ∂N/∂η em v

        # Montar matriz [H] 
        M = np.array([[1, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 1, 1, 0]])
        H = np.tile(M, (num_elements, 1, 1))

        # Montar matriz [B]
        B = H @ J2 @ DN
    
    elif element_type == 'QUAD9':
        if xi is None or eta is None:
            raise ValueError("xi e eta são necessários para QUAD9")

        # Derivadas das funções de forma (2, 9)
        dN_local = shape_derivatives(element_type, xi, eta)

        # Expandir para todos os elements (num_elements, 2, 9)
        dN = np.tile(dN_local, (num_elements, 1, 1))

        # Calculate Jacobian
        J, detJ = jacobian(dN, coords)
        inv_J = np.linalg.inv(J)
        J2 = np.kron(np.eye(2), inv_J)

        DN = np.zeros((num_elements, 4, 18))
        DN[:, 0, 0::2] = dN[:, 0, :]  # ∂N/∂ξ em u
        DN[:, 1, 0::2] = dN[:, 1, :]  # ∂N/∂η em u
        DN[:, 2, 1::2] = dN[:, 0, :]  # ∂N/∂ξ em v
        DN[:, 3, 1::2] = dN[:, 1, :]  # ∂N/∂η em v

        # Montar matriz [H] 
        M = np.array([[1, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 1, 1, 0]])
        H = np.tile(M, (num_elements, 1, 1))

        # Montar matriz [B]
        B = H @ J2 @ DN
    
    else:
        raise ValueError(f"Element type '{element_type}' is not supported.")

    return B, area, detJ


def stress_strain_at_point(element_type, num_elements, coords, C, u, xi=None, eta=None):
    """
    Calcula tensões e deformações em um ponto do elemento.

    Args:
        xi, eta: Coordenadas naturais
        u: Vetor de deslocamentos nodais (16,)

    Returns:
        strain: Vetor de deformações [εxx, εyy, γxy]
        stress: Vetor de tensões [σxx, σyy, τxy]
    """
    # Obter a matriz [B]
    B, *_ = B_matrix(element_type, num_elements, coords, xi, eta)
    
    # Deformações (ε = B @ u)
    strain = B @ u

    # Tensões (σ = C @ ε)
    stress = C @ strain

    return stress, strain


def compute_nodal_stresses(element_type, num_elements, coords, C, u):
    """
    Calcula tensões nodes nós do elemento (extrapolação de Gauss).

    Args:
        u: Vetor de deslocamentos (16,)

    Returns:
        nodal_stresses: Array (8x3) com [σxx, σyy, τxy] em cada nó
    """
    # Determinar o tipo de elemento analisado
    if element_type == 'CST':
        # Tensão no elemento (consntante)
        stress, strain = stress_strain_at_point(element_type, num_elements, coords, C, u)

        # Transpor o array
        stress_b = stress.transpose(0, 2, 1)
        strain_b = strain.transpose(0, 2, 1)

        # Repetir para todos os elements
        nodal_stresses = np.broadcast_to(stress_b, (num_elements, 3, 3))
        nodal_strains = np.broadcast_to(strain_b, (num_elements, 3, 3))
    
    elif element_type == 'QUAD':
        # Tensão no elemento (extrapolação de Gauss)
        g = 1 / np.sqrt(3)
        gauss_3x3 = [(-g, -g),
                     (+g, -g),
                     (+g, +g),
                     (-g, +g)]

        # Calcular tensões nodes 4 pontos de Gauss
        gauss_stresses = np.zeros((num_elements, 4, 3, 1))
        gauss_strains = np.zeros((num_elements, 4, 3, 1))

        # Calcular tensão e deformação em cada ponto
        for i, (xi, eta) in enumerate(gauss_3x3):
            stress, strain = stress_strain_at_point(element_type, num_elements, coords, C, u, xi, eta)
            gauss_stresses[:, i] = stress
            gauss_strains[:, i] = strain

        # Extrapolação para os nós
        nodal_stresses = np.zeros((num_elements, 4, 3, 1))
        nodal_strains = np.zeros((num_elements, 4, 3, 1))

        # Arrays de tensão e deformação
        s0 = gauss_stresses[:, 0]
        s1 = gauss_stresses[:, 1]
        s2 = gauss_stresses[:, 2]
        s3 = gauss_stresses[:, 3]

        e0 = gauss_strains[:, 0]
        e1 = gauss_strains[:, 1]
        e2 = gauss_strains[:, 2]
        e3 = gauss_strains[:, 3]

        # Nós de canto (extrapolação direta)
        c1 = 1 + np.sqrt(3) / 2
        c2 = -0.5
        c3 = 1 - np.sqrt(3) / 2

        # Nós de canto
        nodal_stresses[:, 0] = c1 * s0 + c2 * s1 + c3 * s2 + c2 * s3  # Nó 1 (-1,-1)
        nodal_stresses[:, 1] = c2 * s0 + c1 * s1 + c2 * s2 + c3 * s3  # Nó 2 ( 1,-1)
        nodal_stresses[:, 2] = c3 * s0 + c2 * s1 + c1 * s2 + c2 * s3  # Nó 3 ( 1, 1)
        nodal_stresses[:, 3] = c2 * s0 + c3 * s1 + c2 * s2 + c1 * s3  # Nó 4 (-1, 1)
        nodal_strains[:, 0] = c1 * e0 + c2 * e1 + c3 * e2 + c2 * e3  # Nó 1 (-1,-1)
        nodal_strains[:, 1] = c2 * e0 + c1 * e1 + c2 * e2 + c3 * e3  # Nó 2 ( 1,-1)
        nodal_strains[:, 2] = c3 * e0 + c2 * e1 + c1 * e2 + c2 * e3  # Nó 3 ( 1, 1)
        nodal_strains[:, 3] = c2 * e0 + c3 * e1 + c2 * e2 + c1 * e3  # Nó 4 (-1, 1)

    elif element_type == 'QUAD8':
        # Tensão no elemento (extrapolação de Gauss)
        g = 1 / np.sqrt(3)
        gauss_3x3 = [(-g, -g),
                     (+g, -g),
                     (+g, +g),
                     (-g, +g)]

        # Calcular tensões nodes 4 pontos de Gauss
        gauss_stresses = np.zeros((num_elements, 4, 3, 1))
        gauss_strains = np.zeros((num_elements, 4, 3, 1))

        # Calcular tensão e deformação em cada ponto
        for i, (xi, eta) in enumerate(gauss_3x3):
            stress, strain = stress_strain_at_point(element_type, num_elements, coords, C, u, xi, eta)
            gauss_stresses[:, i] = stress
            gauss_strains[:, i] = strain

        # Extrapolação para os nós
        nodal_stresses = np.zeros((num_elements, 8, 3, 1))
        nodal_strains = np.zeros((num_elements, 8, 3, 1))

        # Arrays de tensão e deformação
        s0 = gauss_stresses[:, 0]
        s1 = gauss_stresses[:, 1]
        s2 = gauss_stresses[:, 2]
        s3 = gauss_stresses[:, 3]

        e0 = gauss_strains[:, 0]
        e1 = gauss_strains[:, 1]
        e2 = gauss_strains[:, 2]
        e3 = gauss_strains[:, 3]

        # Nós de canto (extrapolação direta)
        c1 = 1 + np.sqrt(3) / 2
        c2 = -0.5
        c3 = 1 - np.sqrt(3) / 2

        # Nós de canto
        nodal_stresses[:, 0] = c1 * s0 + c2 * s1 + c3 * s2 + c2 * s3  # Nó 1 (-1,-1)
        nodal_stresses[:, 1] = c2 * s0 + c1 * s1 + c2 * s2 + c3 * s3  # Nó 2 ( 1,-1)
        nodal_stresses[:, 2] = c3 * s0 + c2 * s1 + c1 * s2 + c2 * s3  # Nó 3 ( 1, 1)
        nodal_stresses[:, 3] = c2 * s0 + c3 * s1 + c2 * s2 + c1 * s3  # Nó 4 (-1, 1)
        nodal_strains[:, 0] = c1 * e0 + c2 * e1 + c3 * e2 + c2 * e3  # Nó 1 (-1,-1)
        nodal_strains[:, 1] = c2 * e0 + c1 * e1 + c2 * e2 + c3 * e3  # Nó 2 ( 1,-1)
        nodal_strains[:, 2] = c3 * e0 + c2 * e1 + c1 * e2 + c2 * e3  # Nó 3 ( 1, 1)
        nodal_strains[:, 3] = c2 * e0 + c3 * e1 + c2 * e2 + c1 * e3  # Nó 4 (-1, 1)

        # Nós intermediários (interpolação linear)
        nodal_stresses[:, 4] = 0.5 * (nodal_stresses[:, 0] + nodal_stresses[:, 1])  # Entre 1-2
        nodal_stresses[:, 5] = 0.5 * (nodal_stresses[:, 1] + nodal_stresses[:, 2])  # Entre 2-3
        nodal_stresses[:, 6] = 0.5 * (nodal_stresses[:, 2] + nodal_stresses[:, 3])  # Entre 3-4
        nodal_stresses[:, 7] = 0.5 * (nodal_stresses[:, 3] + nodal_stresses[:, 0])  # Entre 4-1
        nodal_strains[:, 4] = 0.5 * (nodal_strains[:, 0] + nodal_strains[:, 1])  # Entre 1-2
        nodal_strains[:, 5] = 0.5 * (nodal_strains[:, 1] + nodal_strains[:, 2])  # Entre 2-3
        nodal_strains[:, 6] = 0.5 * (nodal_strains[:, 2] + nodal_strains[:, 3])  # Entre 3-4
        nodal_strains[:, 7] = 0.5 * (nodal_strains[:, 3] + nodal_strains[:, 0])  # Entre 4-1
    
    elif element_type == 'QUAD9':
        # Tensão no elemento (extrapolação de Gauss)
        g = np.sqrt(3 / 5)
        gauss_3x3 = [(-g, -g), (+g, -g), (+g, +g), (-g, +g),                                      
                     ( 0, -g), (+g,  0), ( 0, +g), (-g,  0), (0, 0)]

        # Calcular tensões nodes 9 pontos de Gauss
        gauss_stresses = np.zeros((num_elements, 9, 3))
        gauss_strains = np.zeros((num_elements, 9, 3))

        # Calcular tensão e deformação em cada ponto
        for i, (xi, eta) in enumerate(gauss_3x3):
            stress, strain = stress_strain_at_point(element_type, num_elements, coords, C, u, xi, eta)
            gauss_stresses[:, i, :] = stress.squeeze(axis=-1)
            gauss_strains[:, i, :] = strain.squeeze(axis=-1)
        
        # Extrapolação para os nós
        nodal_stresses = np.zeros((num_elements, 9, 3, 1))
        nodal_strains = np.zeros((num_elements, 9, 3, 1))

        # Nós de canto
        c1 = 5 * (np.sqrt(15) + 4) / 18
        c2 = 5 / 18
        c3 = -5 * (np.sqrt(15) - 4) / 18
        c4 = -(np.sqrt(15) + 5) / 9
        c5 = (-5 + np.sqrt(15)) / 9
        c6 = 4 / 9

        # Nós intermediários
        c0 = 0
        c7 = (np.sqrt(15) + 5) / 6
        c8 = -(-5 + np.sqrt(15)) / 6
        c9 = -2 / 3

        # Matriz de extrapolação
        ext_matrix = np.array([
            [c1, c2, c3, c2, c4, c5, c5, c4, c6],
            [c2, c1, c2, c3, c4, c4, c5, c5, c6],
            [c3, c2, c1, c2, c5, c4, c4, c5, c6],
            [c2, c3, c2, c1, c5, c5, c4, c4, c6],
            [c0, c0, c0, c0, c7, c0, c8, c0, c9],
            [c0, c0, c0, c0, c0, c7, c0, c8, c9],
            [c0, c0, c0, c0, c8, c0, c7, c0, c9],
            [c0, c0, c0, c0, c0, c8, c0, c7, c9],
            [c0, c0, c0, c0, c0, c0, c0, c0, +1]
        ])

        # Extrapolação para os nós
        nodal_stresses = np.einsum('ij, njk -> nik', ext_matrix, gauss_stresses)
        nodal_strains = np.einsum('ij, njk -> nik', ext_matrix, gauss_strains)

        # Ajustar dimensões
        nodal_stresses = nodal_stresses[..., np.newaxis]
        nodal_strains = nodal_strains[..., np.newaxis]

    else:
        raise ValueError(f"Element type '{element_type}' is not supported.")

    return nodal_stresses, nodal_strains


def local_stiffness_matrix(element_type, num_elements, C, t, coords):
    """
    Calcula a matriz de rigidez do elemento por integração de Gauss.

    Returns:
        K: Matriz de rigidez elástica de cada elemento da estrutura
    """
    if element_type == 'CST':
        # Matriz deformação-deslocamento, [B]
        B, area, _ = B_matrix(element_type, num_elements, coords)

        # Obter K = t * area * [B].T @ [C] @ [B]
        te = t[:, np.newaxis]
        area = area[:, np.newaxis, np.newaxis]
        K = te * area * (B.transpose(0, 2, 1) @ C @ B)

    elif element_type == 'QUAD':
        # Initialize K matrix
        K = np.zeros((num_elements, 8, 8))

        # Pontos e pesos de Gauss
        points, weights = gauss_quadrature(element_type)

        # Integração numérica, [B].T @ [C] @ [B]
        te = t[:, np.newaxis]
        for (xi, eta), weight in zip(points, weights):
            B, _, detJ = B_matrix(element_type, num_elements, coords, xi, eta)
            detJ = detJ[:, np.newaxis, np.newaxis]
            K += weight * te * detJ * (B.transpose(0, 2, 1) @ C @ B)
        
    elif element_type == 'QUAD8':
        # Initialize K matrix
        K = np.zeros((num_elements, 16, 16))

        # Pontos e pesos de Gauss
        points, weights = gauss_quadrature(element_type)

        # Integração numérica, [B].T @ [C] @ [B]
        te = t[:, np.newaxis]
        for (xi, eta), weight in zip(points, weights):
            B, _, detJ = B_matrix(element_type, num_elements, coords, xi, eta)
            detJ = detJ[:, np.newaxis, np.newaxis]
            K += weight * te * detJ * (B.transpose(0, 2, 1) @ C @ B)
        
    elif element_type == 'QUAD9':
        # Initialize K matrix
        K = np.zeros((num_elements, 18, 18))

        # Pontos e pesos de Gauss
        points, weights = gauss_quadrature(element_type)

        # Integração numérica, [B].T @ [C] @ [B]
        te = t[:, np.newaxis]
        for (xi, eta), weight in zip(points, weights):
            B, _, detJ = B_matrix(element_type, num_elements, coords, xi, eta)
            detJ = detJ[:, np.newaxis, np.newaxis]
            K += weight * te * detJ * (B.transpose(0, 2, 1) @ C @ B)
    
    else:
        raise ValueError(f"Element type '{element_type}' is not supported.")

    return K


def sparse_matrix(num_elements, ke, DOF, numDOF, GLe):
    """
    Monta a matriz de rigidez global esparsa a partir das matrizes
    de rigidez dos elements.

    Parâmetros:
        elements (int): Número de elements.
        ke (np.ndarray): Matrizes de rigidez dos elements (shape: (elements, 2*dofs_per_node, 2*dofs_per_node)).
        dofs_per_node (int): Número de graus de liberdade por nó.
        numDOF (int): Número total de graus de liberdade.
        GLe (np.ndarray): Vetor de graus de liberdade dos elements (shape: (elements, 2*dofs_per_node)).

    Retorna:
        scipy.sparse.csc_matrix: Matriz de rigidez global esparsa.
    """
    # Obter todos os valores de 'metadata'
    data = ke.flatten()

    # Gerar os índices de 'rows'
    rows_temp = np.broadcast_to(GLe[:, :, None], (num_elements, DOF, DOF))
    rows = rows_temp.flatten()

    # Gerar os índices de 'cols' 
    cols_temp = np.broadcast_to(GLe[:, None, :], (num_elements, DOF, DOF))
    cols = cols_temp.flatten()

    # Converter listas para matriz esparsa no formato COO -> CSC
    KG = coo_matrix((data, (rows, cols)), shape=(numDOF, numDOF)).tocsc()

    return KG


def global_stiffness_matrix(element_type, coords, connectivity, properties, numDOF, GLL, GLe, F):
    """
    Orquestra a análise de elements finitos para uma malha completa.

    Args:
        element_type (str): Tipo do elemento (CST, QUAD, QUAD9, QUAD8).
        coords (np.array): Coordenadas de todos os nós da malha.
        connectivity (np.array): Matriz de conectividade (lista de nós por elemento).
        E, nu, t: Propriedades do material e espessura.
        fixed_nodes (list): Lista de índices de nós que estão engastados (ux=uy=0).
        applied_forces (dict): Dicionário {node_idx: (fx, fy)} com as forças aplicadas.
    """
    # Propriedades
    E = properties['E']
    nu = properties['nu']
    t = properties['t']

    # Number of elements and constitutive matrix of each surface
    C = []
    ts = []
    conec = []
    num_elements = 0

    # Loop over surfaces
    for surface in connectivity:
        # Surface connectivity
        surface_connectivity = connectivity[surface]
        conec.extend(surface_connectivity)

        # Surface elements
        surface_elements = len(connectivity[surface])
        num_elements += surface_elements

        # Surface thickness
        t_s = t[surface]
        t_s = np.ones((surface_elements, 1)) * t_s
        ts.extend(t_s)

        # Constitutive matrix
        C_s = constitutive_matrix(surface_elements, E[surface], nu[surface])
        C.extend(C_s)
    
    # Convert to array
    C = np.array(C)
    ts = np.array(ts)
    conec = np.array(conec)

    # Inicialização
    KG = np.zeros((numDOF, numDOF))
    node_map = {
        'CST': (3, 6),
        'QUAD': (4, 8),
        'QUAD8': (8, 16),
        'QUAD9': (9, 18),
    }

    # Coordinates of each element
    element_coords = coords[conec]
    nodes_per_element, DOF = node_map[element_type]

    # Calculate local stiffness matrix
    ke = local_stiffness_matrix(element_type, num_elements, C, ts, element_coords)

    # Build global stiffness matrix
    KG = sparse_matrix(num_elements, ke, DOF, numDOF, GLe)
    
    # Aplicar condições de contorno
    K_red = KG[np.ix_(GLL, GLL)]
    F_red = F[GLL]

    # Resolver o sistema
    U_red = spsolve(K_red, F_red).reshape(-1, 1)

    # Reconstruir o vetor de deslocamentos global completo
    U = np.zeros((numDOF, 1))
    U[GLL, 0] = U_red[:, 0]

    # Obter os deslocamentos locais por elemento
    u_local = U[GLe]

    print(f"Solução encontrada. Número de elements: {num_elements}. Deslocamento máximo: {np.max(np.abs(U)):.4f} m")

    # PÓS-PROCESSAMENTO (Cálculo de Tensões)
    stress, strain = compute_nodal_stresses(element_type, num_elements, element_coords, C, u_local)

    return KG, U, stress, strain


def _extrair_conectividade_por_dim(dim, grupos_fisicos, tipo_elemento_alvo, nos_por_elemento, node_map):
    """
    Função auxiliar genérica para extrair a conectividade de elements
    de uma determinada dimensão (1D ou 2D).
    """
    elementos_por_dominio = {}
    for dim_grupo, tag_grupo in grupos_fisicos:
        nome = gmsh.model.getPhysicalName(dim_grupo, tag_grupo)
        entidades = gmsh.model.getEntitiesForPhysicalGroup(dim_grupo, tag_grupo)

        conectividades_do_grupo = []
        for tag_entidade in entidades:
            # Extrai todos os tipos de elements da entidade
            tipos_elementos, _, tags_nos_por_elem = gmsh.model.mesh.getElements(dim, tag_entidade)

            # Verifica se o tipo de elemento que queremos existe
            if tipo_elemento_alvo in tipos_elementos:
                # Encontra o índice correto na lista
                try:
                    # Usar list.index é mais Pythonic que np.where[0][0]
                    idx_tipo = list(tipos_elementos).index(tipo_elemento_alvo)
                except ValueError:
                    continue # Tipo de elemento não encontrado

                tags_nos = np.array(tags_nos_por_elem[idx_tipo])
                if tags_nos.size > 0:
                    conec_entidade = tags_nos.reshape(-1, nos_por_elemento)
                    conectividades_do_grupo.append(conec_entidade)

        if conectividades_do_grupo:
            conectividade_total = np.vstack(conectividades_do_grupo)
            # Converte as tags do Gmsh para índices 0-based
            conectividade_idx = np.array(
                list(map(node_map.get, conectividade_total.flat))
            ).reshape(conectividade_total.shape)
            elementos_por_dominio[nome] = conectividade_idx

    return elementos_por_dominio


def _extrair_nos_por_dim(dims, node_map):
    """
    Função auxiliar genérica para extrair os NÓS de grupos físicos
    de dimensões 0D (pontos) ou 1D (linhas).
    """
    nos_por_grupo = {}
    
    # Coleta todos os grupos físicos das dimensões solicitadas
    grupos_fisicos = []
    for dim in dims:
        grupos_fisicos.extend(gmsh.model.getPhysicalGroups(dim))

    for dim, tag in grupos_fisicos:
        nome = gmsh.model.getPhysicalName(dim, tag)
        
        # getNodesForPhysicalGroup é a forma mais direta de obter os nós
        nos_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)
        
        if len(nos_tags):
            # Converte tags para índices e remove duplicatas
            node_indices = np.array(list(map(node_map.get, nos_tags)))
            nos_por_grupo[nome] = np.unique(node_indices)
            
    return nos_por_grupo


def carregar_estrutura_gmsh(caminho, tipo_elemento):
    """
    Carrega um arquivo .geo_unrolled do Gmsh, gera a malha e extrai
    as coordenadas e conectividades por grupo físico.

    Args:
        caminho (str): O caminho para o arquivo .geo_unrolled.
        tipo_elemento (str): Tipo do elemento ('CST', 'QUAD', 'QUAD8', 'QUAD9').

    Returns:
        coords (np.array): Array global de coordenadas de nós.
        conec_2d (dict): Conectividades 2D por grupo físico.
        nos_por_grupo (dict): Nós em grupos 0D (pontos) e 1D (linhas).
        conec_1d (dict): Elementos 1D nas bordas para aplicação de cargas.
    """
    # Mapeamento de tipos de elemento 2D (Tipo Gmsh, Qtd. Nós)
    mapa_elemento_2d = {
        'CST': (2, 3),    # Tipo 2 = Triângulo de 3 nós
        'QUAD': (3, 4),   # Tipo 3 = Quadrilátero de 4 nós
        'QUAD8': (16, 8), # Tipo 16 = Quadrilátero de 8 nós
        'QUAD9': (10, 9), # Tipo 10 = Quadrilátero de 9 nós
    }
    # Mapeamento de elements 1D (Bordas)
    mapa_elemento_1d = {
        'CST': (1, 2),    # Tipo 1 = Linha de 2 nós
        'QUAD': (1, 2),   # Tipo 1 = Linha de 2 nós
        'QUAD8': (8, 3),  # Tipo 8 = Linha de 3 nós
        'QUAD9': (8, 3),  # Tipo 8 = Linha de 3 nós
    }

    try:
        tipo_2d, nos_2d = mapa_elemento_2d[tipo_elemento]
        tipo_1d, nos_1d = mapa_elemento_1d[tipo_elemento]
    except KeyError:
        raise ValueError(f"Tipo de elemento '{tipo_elemento}' não suportado.")

    gmsh.initialize()
    try:
        gmsh.open(caminho)
        gmsh.model.geo.synchronize()

        # Extrair Nós
        node_tags, coords_raw, _ = gmsh.model.mesh.getNodes()
        coords = np.array(coords_raw).reshape(-1, 3)[:, :2]
        node_map = {tag: i for i, tag in enumerate(node_tags)}

        # Extrair Elementos 2D (Superfícies)
        grupos_fisicos_2d = gmsh.model.getPhysicalGroups(2)
        conec_2d = _extrair_conectividade_por_dim(
            2, grupos_fisicos_2d, tipo_2d, nos_2d, node_map
        )

        # Extrair Elementos 1D (Bordas)
        grupos_fisicos_1d = gmsh.model.getPhysicalGroups(1)
        conec_1d = _extrair_conectividade_por_dim(
            1, grupos_fisicos_1d, tipo_1d, nos_1d, node_map
        )
        
        # Extrair Nós (0D e 1D)
        nos_por_grupo = _extrair_nos_por_dim([0, 1], node_map)

    finally:
        gmsh.finalize()

    return coords, conec_2d, nos_por_grupo, conec_1d


# Definir o tipo do elemento
# element_type = 'CST'
# element_type = 'QUAD'
element_type = 'QUAD8'
# element_type = 'QUAD9'

path = r"C:\Users\volpi\Documents\Python Projects\SimuStruct\SimuMembrane\steel_bar.msh"
# path = r"C:\Users\volpi\Documents\Python Projects\SimuStruct\SimuMembrane\Lista 3 - Viga Parede.msh"
coords, connectivity, boundary_nodes, boundary_elements = carregar_estrutura_gmsh(path, element_type)

# Material properties (define accordingly to the materials used in the mesh)
# properties = {
#     'E': {
#         'VigaParede': 2e6,
#         'Support_left': 2e12,
#         'Support_right': 2e12
#     },
#     'nu': {
#         'VigaParede': 0.2,
#         'Support_left': 0.2,
#         'Support_right': 0.2
#     },
#     't': {
#         'VigaParede': 0.4,
#         'Support_left': 0.01,
#         'Support_right': 0.01
#     }
# }

# Material properties
properties = {
    'E': {
        'Steel': 20000,
    },
    'nu': {
        'Steel': 0.3,
    },
    't': {
        'Steel': 1.0,
    }
}

def degrees_of_freedom(element_type, coords, connectivity, boundary_nodes):
    """
    Define os graus de liberdade e restrições da estrutura.

    Args:
        element_type (str): Tipo do elemento.
        coords (np.array): Coordenadas dos nós.
        connectivity (dict): Conectividades por superfície.
        boundary_nodes (dict): Nós de contorno e suas restrições.

    Returns:
        GLL (np.array): Máscara booleana dos graus de liberdade livres.
        GLe (np.array): Graus de liberdade locais de cada elemento.
        numDOF (int): Número total de graus de liberdade.
    """
    DOF = 2
    num_nos = coords.shape[0]
    numDOF = DOF * num_nos

    # Mapeamento de restrições
    CONSTRAINTS_MAP = {
        'ENGASTE':  range(DOF),
        'FIXO_XY':  [0, 1],
        'ROLETE_X': [1],
        'ROLETE_Y': [0]
    }

    # Mapeamento de tipos de célula
    CELL_TYPE_MAP = {
        'CST': (3, 6),
        'QUAD': (4, 8),
        'QUAD8': (8, 16),
        'QUAD9': (9, 18)
    }

    nodes_per_cell, dof_per_cell = CELL_TYPE_MAP[element_type]

    # Criar set de graus de liberdade restritos
    glr_set = set()
    for support, nodes in boundary_nodes.items():
        if support in CONSTRAINTS_MAP:
            for idx in nodes:
                idx_support = 2 * idx
                for dof_local in CONSTRAINTS_MAP[support]:
                    glr_set.add(idx_support + dof_local)

    GLR = np.array(sorted(list(glr_set)), dtype=int)

    # Criar máscara de graus de liberdade livres
    GLL = np.ones(numDOF, dtype=bool)
    if GLR.size > 0:
        GLL[GLR] = False

    # Calcular graus de liberdade locais
    dof_offsets = np.arange(DOF)
    GLe = []
    
    for surface in connectivity:
        conec = connectivity[surface]
        elements = conec.shape[0]
        gle_base = DOF * conec
        GLe_temp = gle_base[..., np.newaxis] + dof_offsets
        GLe.extend(GLe_temp.reshape(elements, dof_per_cell))

    GLe = np.array(GLe)

    return GLL, GLe, numDOF


def global_force_vector(element_type, coords, numDOF, boundary_nodes=None, boundary_elements=None, F_total=None, direction=(0.0, -1.0)):
    """
    Cria um vetor de força global aplicando carga distribuída em elements 1D.

    Args:
        element_type (str): Tipo do elemento ('CST', 'QUAD', 'QUAD8', 'QUAD9').
        coords (np.array): Coordenadas dos nós.
        numDOF (int): Número total de graus de liberdade.
        elementos_1d (np.array): Conectividade dos elements 1D (N, 2 ou 3).
        F_total (float): Força total a ser aplicada.
        direction (tuple): Direção da força (dx, dy).

    Returns:
        np.array: Vetor de força global.
    """
    F = np.zeros(numDOF, dtype=float)
    
    # Verificar a existência de cargas concentradas
    if boundary_nodes is not None and 'CLoad' in boundary_nodes:
        nodes = boundary_nodes['CLoad']
        
        # Verificar formato da carga
        if F_total and nodes:
            for node in nodes:
                # Formato: (node_idx, Fx, Fy)
                Fx, Fy = F_total
                F[2 * node] += Fx
                F[2 * node + 1] += Fy
        else:
            print(f"Aviso: 'Load' deve ser uma lista/tupla, recebido: {type(nodes)}")
        
    if boundary_elements is None or len(boundary_elements) == 0:
        print("Aviso: Nenhum elemento 1D fornecido para aplicação de carga.")
        return F

    # Normalizar direção
    dir_vec = np.array(direction)
    norm = np.linalg.norm(dir_vec)
    if norm == 0:
        return F
    unit_dir = dir_vec / norm

    # Determinar tipo de interpolação
    interpolacao_linear = element_type in ['CST', 'QUAD']
    # nos_por_elemento_1d = 2 if interpolacao_linear else 3

    # Calcular comprimento total da linha
    comprimento_total = 0.0
    comprimentos_elementos = []
    
    for elem in boundary_elements:
        if interpolacao_linear:
            # Elementos lineares: 2 nós
            p1, p2 = coords[elem[0]], coords[elem[1]]
            L = np.linalg.norm(p2 - p1)
        else:
            # Elementos quadráticos: 3 nós (usar distância dos extremos)
            p1, p3 = coords[elem[0]], coords[elem[2]]
            L = np.linalg.norm(p3 - p1)
        
        comprimentos_elementos.append(L)
        comprimento_total += L
    
    # Intensidade da carga por unidade de comprimento
    if F_total:
        q = F_total / comprimento_total

        # Aplicar carga em cada elemento
        for i, elem in enumerate(boundary_elements):
            L = comprimentos_elementos[i]
            print(L)
            F_total_on_edge = q * L # Força total neste elemento de borda
            
            # Componentes X e Y da força total no elemento
            F_vec_x = F_total_on_edge * unit_dir[0]
            F_vec_y = F_total_on_edge * unit_dir[1]
            
            if interpolacao_linear:
                # LÓGICA LINEAR (CST, Q4)
                n1_idx, n2_idx = elem
                
                F_nodal_x = F_vec_x * 0.5
                F_nodal_y = F_vec_y * 0.5
                
                F[2 * n1_idx] += F_nodal_x
                F[2 * n1_idx + 1] += F_nodal_y
                F[2 * n2_idx] += F_nodal_x
                F[2 * n2_idx + 1] += F_nodal_y
                
            else:
                # LÓGICA QUADRÁTICA (Q8, Q9)
                n1_idx, n2_idx, n_mid_idx = elem
                
                # Aplicar pesos corretos aos nós corretos
                F_canto_x = F_vec_x * (1.0/6.0)
                F_canto_y = F_vec_y * (1.0/6.0)
                F_meio_x  = F_vec_x * (4.0/6.0)
                F_meio_y  = F_vec_y * (4.0/6.0)
                
                # Nó de Canto 1
                F[2 * n1_idx] += F_canto_x
                F[2 * n1_idx + 1] += F_canto_y
                
                # Nó de Canto 2
                F[2 * n2_idx] += F_canto_x
                F[2 * n2_idx + 1] += F_canto_y
                
                # Nó do Meio
                F[2 * n_mid_idx] += F_meio_x
                F[2 * n_mid_idx + 1] += F_meio_y
    else:
        raise ValueError("Vetor de forças externas não definido.")

    return F

# Associar graus de liberdade à estrutura
GLL, GLe, numDOF = degrees_of_freedom(element_type, coords, connectivity, boundary_nodes)

# Extract forces from source
F = np.zeros(numDOF, dtype=float)
if 'DLoad' in boundary_elements:
    F_total = 1.0
    direction = (0.0, -1.0)
    
    F = global_force_vector(
        element_type, coords, numDOF, boundary_elements=boundary_elements['DLoad'],
        F_total=F_total, direction=direction
    )
elif 'CLoad' in boundary_nodes:
    F_total = (0.0, -1.0)
    
    F = global_force_vector(
        element_type, coords, numDOF, boundary_nodes, F_total=F_total
    )
else:
    raise ValueError("You must define a valid external force.")

print(F)
quit()

# Calculate global stiffness matrix, [KE]
KE, U, nodal_stresses, nodal_strains = global_stiffness_matrix(element_type, coords, connectivity, properties, numDOF, GLL, GLe, F)


def visualize_results(element_type, coords, conec, U_global, stresses_per_element, boundary_nodes, save_plot=None, save_filename='plot.png'):
    """
    Visualiza os resultados de uma análise de múltiplos elements,
    usando os dados nodais pré-calculados para a plotagem de tensões.
    """
    # Inicialização dos dados
    node_map = {
        'CST': (3, pv.CellType.TRIANGLE),
        'QUAD': (4, pv.CellType.QUAD),
        'QUAD8': (4, pv.CellType.QUAD),
        'QUAD9': (9, pv.CellType.BIQUADRATIC_QUAD),
    }
    nodes_per_element, cell_type = node_map[element_type]

    # Preparar a malha de dados
    if coords.shape[1] == 2:
        coords_3d = np.hstack([coords, np.zeros((len(coords), 1))])
    else:
        coords_3d = coords

    # Fator de escala para visualização de deslocamentos
    deslocamentos_2d = U_global.reshape(-1, 2)
    deslocamentos_3d = np.hstack([deslocamentos_2d, np.zeros((len(deslocamentos_2d), 1))])
    deslocamentos_mag =  np.abs(deslocamentos_2d[:, 1]) # np.linalg.norm(deslocamentos_3d, axis=1)
    scale_factor = 1.0 / np.max(deslocamentos_mag)
    coords_def = coords_3d + scale_factor * deslocamentos_3d

    # Construção da malha unificada
    all_cells = []
    all_cell_types = []

    # Iterar sobre as superfícies criadas
    for surface in conec:
        num_elem = len(conec[surface])
        surface_cells = np.hstack((np.full((num_elem, 1), nodes_per_element), conec[surface][:, :4])).flatten()
        surface_cell_types = np.full(num_elem, cell_type)
        all_cells.append(surface_cells)
        all_cell_types.append(surface_cell_types)

    # Concatena todas as superfícies num único array
    all_cells = np.hstack(all_cells)
    all_cell_types = np.hstack(all_cell_types)

    # Cria a malha global
    mesh = pv.UnstructuredGrid(all_cells, all_cell_types, coords_3d)
    mesh.point_data['Deslocamento'] = deslocamentos_mag

    # Cálculo das tensões nodes nós
    num_total_nodes = len(coords)
    global_sigma_xx = np.zeros(num_total_nodes, dtype=float)
    global_sigma_yy = np.zeros(num_total_nodes, dtype=float)
    global_tau_xy = np.zeros(num_total_nodes, dtype=float)
    node_contribution_count = np.zeros(num_total_nodes, dtype=int)

    # Itera sobre cada elemento para calcular e acumular as tensões
    for surface in conec:
        num_elementos = len(conec[surface])
        for i in range(num_elementos):
            element_conec = conec[surface][i]
            element_stresses = stresses_per_element[i]  # Tensões [σxx, σyy, τxy]

            # Pega os valores de tensão para cada nó
            σxx = element_stresses[:, 0].flatten()
            σyy = element_stresses[:, 1].flatten()
            τxy = element_stresses[:, 2].flatten()

            # Acumula os valores nodes arrays globais
            global_sigma_xx[element_conec] += σxx
            global_sigma_yy[element_conec] += σyy
            global_tau_xy[element_conec] += τxy

            # Acumula o contador de contribuições para cada nó
            node_contribution_count[element_conec] += 1

    # Realiza a média, evitando divisão por zero
    valid_nodes = node_contribution_count > 0
    node_contribution_count[~valid_nodes] = 1 # Evita divisão por zero

    # Cálculo das tensões médias
    avg_sigma_xx = global_sigma_xx / node_contribution_count
    avg_sigma_yy = global_sigma_yy / node_contribution_count
    avg_tau_xy = global_tau_xy / node_contribution_count
    avg_vm_stresses = np.sqrt(
            avg_sigma_xx ** 2 - avg_sigma_xx * avg_sigma_yy + avg_sigma_yy ** 2 + 3 * avg_tau_xy ** 2
        )
    
    # Adiciona os resultados de tensão como "point metadata" à malha
    mesh.point_data['von_Mises'] = avg_vm_stresses
    mesh.point_data['sigma_xx'] = avg_sigma_xx
    mesh.point_data['sigma_yy'] = avg_sigma_yy
    mesh.point_data['tau_xy'] = avg_tau_xy

    # Obter a tensão em um ponto
    if 'Stress_point' in boundary_nodes:
        stress_value = mesh.point_data['sigma_xx'][boundary_nodes['Stress_point']]
        print(stress_value)

    # Cria uma cópia da malha para a forma deformada
    deformed_mesh = mesh.copy()
    deformed_mesh.points = coords_def

    # Salvar um subplot específico
    if save_plot:
        print(f"Salvando plot '{save_plot}' em '{save_filename}'...")
        
        # Cria um plotter "off-screen" (invisível)
        single_plotter = pv.Plotter(off_screen=True)
        
        # Dicionário de plots
        plot_configs = {
            'malha': {
                'title': "Malha indeformada",
                'mesh': mesh.copy(),
                'add_original_wireframe': True
            },
            'deslocamento': {
                'title': "Deformada com Deslocamento",
                'mesh': deformed_mesh.copy(),
                'scalars': 'Deslocamento',
                'cmap': 'viridis',
                'sargs': dict(title='Desloc. (m)', color='black', position_x=0.2),
                'add_original_wireframe': True
            },
            'von_mises': {
                'title': "Tensão de von Mises (kPa)",
                'mesh': deformed_mesh.copy(),
                'scalars': 'von_Mises',
                'cmap': 'viridis',
                'sargs': dict(title='Mises (kPa)', color='black', position_x=0.2),
                'add_original_wireframe': False
            },
            'sigma_xx': {
                'title': "Tensão Normal X (S11)",
                'mesh': deformed_mesh.copy(),
                'scalars': 'sigma_xx',
                'cmap': 'jet',
                'sargs': dict(title='S11 (kPa)', color='black', fmt='%.2e', position_x=0.2),
                'add_original_wireframe': False
            },
            'sigma_yy': {
                'title': "Tensão Normal Y (S22)",
                'mesh': deformed_mesh.copy(),
                'scalars': 'sigma_yy',
                'cmap': 'viridis',
                'sargs': dict(title='S22 (kPa)', color='black', fmt='%.2e', position_x=0.2),
                'add_original_wireframe': False
            },
            'tau_xy': {
                'title': "Tensão de Cisalhamento (S12)",
                'mesh': deformed_mesh.copy(),
                'scalars': 'tau_xy',
                'cmap': 'viridis',
                'sargs': dict(title='S12 (kPa)', color='black', fmt='%.2e', position_x=0.2),
                'add_original_wireframe': False
            }
        }

        if save_plot in plot_configs:
            config = plot_configs[save_plot]
            single_plotter.add_text(config['title'], font_size=12, color='black')
            if save_plot != 'malha':
                single_plotter.add_mesh(config['mesh'], 
                                        scalars=config['scalars'], 
                                        show_edges=True, 
                                        cmap=config['cmap'], 
                                        scalar_bar_args=config['sargs'])
            
            if config.get('add_original_wireframe', False):
                single_plotter.add_mesh(mesh, style='wireframe', color='grey', opacity=0.5, line_width=1)

            single_plotter.view_xy()
            single_plotter.camera.zoom(1.2)
            single_plotter.screenshot(save_filename)
            print("... Salvo com sucesso.")
        else:
            print(f"Aviso: Tipo de plot para salvar '{save_plot}' não reconhecido. Ignorando.")

    # PLOTAGEM INTERATIVA (Grid 2x3)
    plotter = pv.Plotter(shape=(2, 3), border=False)

    # Subplot 1: Malha indeformada
    plotter.subplot(0, 0)
    plotter.add_text("Malha indeformada", font_size=10, color='black')
    plotter.add_mesh(mesh, style='wireframe', color='grey', opacity=1.0, line_width=1.5)

    # Subplot 2: Comparativo da malha indeformada e deformada
    plotter.subplot(0, 1)
    plotter.add_text("Comparação da Malha Indeformada e Deformada", font_size=10, color='black')
    plotter.add_mesh(deformed_mesh, style='wireframe', color='red', line_width=2)
    plotter.add_mesh(mesh, style='surface', color='grey', opacity=0.2)

    # Subplot 3: Deformada com colormap de Deslocamento
    plotter.subplot(0, 2)
    plotter.add_text("Deformada com Deslocamento", font_size=10, color='black')
    plotter.add_mesh(deformed_mesh.copy(), scalars='Deslocamento', show_edges=True,
                     cmap='viridis', scalar_bar_args={'title': 'Desloc. (m)', 'color': 'black', 'position_x': 0.2})
    plotter.add_mesh(mesh, style='wireframe', color='grey', opacity=0.5, line_width=1)

    # Subplot 4: Tensão de von Mises
    plotter.subplot(1, 0)
    plotter.add_text("Tensão de von Mises", font_size=10, color='black')
    plotter.add_mesh(deformed_mesh.copy(), scalars='von_Mises', show_edges=True, cmap='jet',
                     scalar_bar_args={'title': 'Mises (kPa)', 'color': 'black', 'position_x': 0.2})

    # Subplot 5: Tensão Normal X (σ_xx)
    plotter.subplot(1, 1)
    plotter.add_text("Tensão Normal X (S11)", font_size=10, color='black')
    plotter.add_mesh(deformed_mesh.copy(), scalars='sigma_xx', show_edges=True, cmap='jet',
                     scalar_bar_args={'title': 'S11 (kPa)', 'color': 'black', 'fmt': '%.1e', 'position_x': 0.2})

    # Subplot 6: Tensão Normal Y (σ_yy)
    plotter.subplot(1, 2)
    plotter.add_text("Tensão Normal Y (S22)", font_size=10, color='black')
    plotter.add_mesh(deformed_mesh.copy(), scalars='sigma_yy', show_edges=True, cmap='jet',
                     scalar_bar_args={'title': 'S22 (kPa)', 'color': 'black', 'fmt': '%.1e', 'position_x': 0.2})
    
    # Configurações gerais
    plotter.link_views()
    plotter.view_xy()
    plotter.camera.zoom(1.2)

    print("\nAbrindo visualização 3D...")
    print("Feche a janela para continuar.")
    plotter.show()


visualize_results(element_type, coords, connectivity, U, nodal_stresses, boundary_nodes, save_plot='sigma_xx', save_filename='sigma_xx.png')
# visualize_results(element_type, coords, connectivity, U, nodal_stresses, save_plot='sigma_yy', save_filename='sigma_yy.png')
# visualize_results(element_type, coords, connectivity, U, nodal_stresses, save_plot='tau_xy', save_filename='tau_xy.png')
# visualize_results(element_type, coords, connectivity, U, nodal_stresses, save_plot='von_mises', save_filename='von_Mises.png')
