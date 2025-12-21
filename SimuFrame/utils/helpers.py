# Third-party libraries
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_array, csc_array

# Local libraries
from SimuFrame.core.model import Structure


def orientation_vector(estrutura, coords, initial_coords):
    """
    Função para obter o vetor de orientação da seção transversal (k).

    Args:
        coords (np.ndarray): Coordenadas dos nós da estrutura.

    Returns:
        dict (np.ndarray): Vetor de referência (indeformada e deformada).
    """
    # Número de elements
    num_elementos = estrutura.num_elements
    num_membros_originais = len(initial_coords)

    # Inicializar vetor de referência
    ref_vector = {
        'undeformed': np.zeros((num_membros_originais, 3)),
        'deformed': np.zeros((num_elementos, 3))
    }

    # Dicionário das coordenadas
    coordenadas = {
        'undeformed': np.copy(initial_coords),
        'deformed': np.copy(coords)
    }

    # Loop ao longo da estrutura
    for tipo in coordenadas:
        # Coordenadas atuais
        coord = coordenadas[tipo]

        # Vetor x_ (direção local x)
        x_ = normalize(coord[:, -1] - coord[:, 0])

        # Criar máscara para determinar os elements alinhados ao eixo z
        mask = (np.abs(x_[:, 0]) < 1e-9) & (np.abs(x_[:, 1]) < 1e-9)

        # Gerar vetores de referência automaticamente
        ref_vector[tipo] = np.where(mask[:, None], np.array([1., 0., 0.]), np.array([0., 0., 1.]))

    return ref_vector


def atribuir_deslocamentos(numDOF, GLL, GLe, T, dr):
    """
    Função para atribuir os deslocamentos reduzidos (dr)
    aos graus de liberdade livres de cada elemento.

    Parâmetros:
        numDOF (int): Número total de graus de liberdade.
        dofs_per_node (int): Graus de liberdade por nó.
        GLL (np.ndarray): Array booleano indicando os graus de liberdade livres.
        GLe (np.ndarray): Array com os graus de liberdade dos elements.
        T (np.ndarray): Matriz de transformação.
        dr (np.ndarray): Array de deslocamentos reduzidos.

    Retorna:
        np.ndarray: Array de deslocamentos locais do elemento.
    """
    # Inicializar o array de deslocamentos
    d = np.zeros((numDOF, 1))

    # Atribuir os deslocamentos aos graus de liberdade livres
    d[GLL] = dr

    return T @ d[GLe]


def assemble_sparse_matrix(
        structure: Structure,
        global_element_stiffness: npt.NDArray[np.float64],
        total_dofs: int,
        element_dofs: npt.NDArray[np.integer]
) -> csc_array:
    """
    Assembles the global stiffness matrix in CSC format.

    Args:
        structure (Structure): Instance of the Structure class.
        global_element_stiffness (npt.NDArray[np.float64]): Assembled global stiffness matrix.
        total_dofs (int): Total number of degrees of freedom in the system.
        element_dofs (npt.NDArray[np.integer]): Array mapping element local DOFs to global indices.

    Returns:
        K (csc_array): Global stiffness matrix in CSC format.
    """
    # Tamanho da matriz de rigidez global
    tamanho_ke = structure.dofs_per_element
    num_elementos = structure.num_elements

    # Obter todos os valores de 'metadata'
    data = global_element_stiffness.flatten()

    # Gerar os índices de 'rows'
    rows_temp = np.broadcast_to(element_dofs[:, :, None], (num_elementos, tamanho_ke, tamanho_ke))
    rows = rows_temp.flatten()

    # Gerar os índices de 'cols' 
    cols_temp = np.broadcast_to(element_dofs[:, None, :], (num_elementos, tamanho_ke, tamanho_ke))
    cols = cols_temp.flatten()

    # Converter listas para array no formato COO -> CSC
    KG = coo_array((data, (rows, cols)), shape=(total_dofs, total_dofs)).tocsc()

    return KG


def extract_element_data(structure):
    """
    Calcula e retorna dados geométricos e constitutivos para elements estruturais.

    Args:
        estrutura (Classe): instância da classe Structure.

    Returns:
        tuple: Uma tupla contendo:
            - coord_ord (np.ndarray): Coordenadas ordenadas dos elements.
            - L (np.ndarray): Comprimentos dos elements.
            - A (np.ndarray): Área da seção transversal dos elements.
            - Ix (np.ndarray): Momento de inércia em relação ao eixo x.
            - Iy (np.ndarray): Momento de inércia em relação ao eixo y.
            - Iz (np.ndarray): Momento de inércia em relação ao eixo z.
            - E (np.ndarray): Módulo de elasticidade.
            - nu (np.ndarray): Coeficiente de Poisson.
            - G (np.ndarray): Módulo de cisalhamento.
    """
    # Retrieve raw object lists
    elements = list(structure.elements.values())
    members = list(structure.original_members.values())

    # Extract connectivity IDs
    conec = np.array([[node.id for node in e.conec] for e in elements], dtype=int)

    # Extract element coords
    coords = np.array([[node.coord for node in e.conec] for e in elements], dtype=float)

    # Calculate lengths
    L = np.array([e.get_element_length() for e in elements], dtype=float)

    # Extract original members coords (initial and final only)
    coords_members = np.array([[m['nodes'][0].coord, m['nodes'][-1].coord] for m in members], dtype=float)

    # Extract sections and materials
    sections = [e.section for e in elements]
    materials = [s.material for s in sections]

    # Extract geometric properties
    A = np.array([s.A for s in sections], dtype=float)
    k = np.array([s.k for s in sections], dtype=float)
    It = np.array([s.It for s in sections], dtype=float)
    Iy = np.array([s.Iy for s in sections], dtype=float)
    Iz = np.array([s.Iz for s in sections], dtype=float)
    rp = np.array([s.rp for s in sections], dtype=float)

    # Extract constitutive properties
    E = np.array([m.E for m in materials], dtype=float)
    nu = np.array([m.nu for m in materials], dtype=float)
    G = np.array([m.G for m in materials], dtype=float)

    # Pack data into the dictionary
    properties = {
        "L": L,
        "A": A,
        "k": k,
        "It": It,
        "Iy": Iy,
        "Iz": Iz,
        "rp": rp,
        "E": E,
        "nu": nu,
        "G": G
    }

    return coords, coords_members, conec, properties


def static_condensation(
        estrutura,
        k,
        f: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Condensação estática vetorizada para múltiplas matrizes de rigidez.

    Args:
        estrutura (Estrutura): Instância da classe Estrutura.
        k (np.array): Matrizes de rigidez globais dos elements.
        f (np.array): Vetor de forças distribuídas.

    Retorna:
        K_cond (np.array): Matrizes de rigidez condensadas.
    """
    # Criar uma lista apenas com elements a serem condensados
    elementos_para_condensar = [
        (i, list(elem.hinges[0] + [estrutura.dofs_per_node + dof for dof in elem.hinges[1]]))
        for i, elem in estrutura.elements.items()
        if elem.hinges != [[], []]
    ]

    # Pré-calcular os índices
    todos_dofs = np.arange(12)
    indices = {
        r: {
            'mantidos': np.setdiff1d(todos_dofs, elim),
            'eliminados': np.array(elim)
        }
        for r, elim in elementos_para_condensar
    }

    # Definir o vetor de forças equivalentes
    if f is None:
        f = np.zeros((estrutura.num_elements, 12, 1))

    # Itera sobre cada elemento da malha
    for idx, releases in indices.items():
        # Pega os índices pré-calculados do cache
        gl_mantidos, gl_eliminados = releases['mantidos'], releases['eliminados']

        # Dados do elemento atual            
        ke, fe = k[idx], f[idx]

        # Partição das matrizes de rigidez
        k_mm = ke[np.ix_(gl_mantidos, gl_mantidos)]
        k_me = ke[np.ix_(gl_mantidos, gl_eliminados)]
        k_em = ke[np.ix_(gl_eliminados, gl_mantidos)]
        k_ee = ke[np.ix_(gl_eliminados, gl_eliminados)]

        # Verificar se k_ee é singular
        try:
            k_ee_inv = np.linalg.inv(k_ee)
        except np.linalg.LinAlgError:
            k_ee_inv = np.linalg.inv(k_ee + np.eye(k_ee.shape[0]) * 1e-9)

        # Condensação
        k_cond = k_mm - k_me @ k_ee_inv @ k_em
        f_cond = fe[gl_mantidos] - k_me @ k_ee_inv @ fe[gl_eliminados]

        # Atualiza apenas os DOFs mantidos
        ke.fill(0.0)
        fe.fill(0.0)
        ke[np.ix_(gl_mantidos, gl_mantidos)] = k_cond
        fe[gl_mantidos] = f_cond

        # Evitar singularidade nodes graus de liberdade condensados
        ke[gl_eliminados, gl_eliminados] += 1e-6

    return k, f

def transformation_matrix(
        estrutura, 
        coords
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calcula as matrizes de transformação locais e globais para múltiplos elements,
    além de obter a deformação relativa de cada eixo local x.

    Args:
        estrutura (Estrutura): Objeto da classe Structure.
        coords (array_like): Coordenadas dos nós dos elements.

    Returns:
        T (array_like): Matriz de transformação global.
        MT (array_like): Matriz de transformação do elemento.
    """
    # Dados iniciais
    num_elementos = estrutura.num_elements

    # Vetor x_ (direção local x: do nó inicial para o final)
    x_ = normalize(coords[:, -1] - coords[:, 0])

    # Auxiliar vector (global y)
    aux = np.array([0., 1., 0.])

    # Create a mask for elements parallel to the global y axis
    mask = (np.abs(x_[:, 0]) < 1e-9) & (np.abs(x_[:, 2]) < 1e-9)

    # For non parallel elements, determine z_aux
    z_aux = normalize(np.cross(x_, aux))

    # For parallel elements, aux = [0, 0, 1]
    aux_alt = np.array([0., 0., 1.])

    # Determine z_ based on mask
    z_ = np.where(mask[:, None], aux_alt, z_aux)

    # z_ vector (complete the right triangle z_ = x_ x y_)
    y_ = normalize(np.cross(z_, x_))

    # Empilhar matrizes de rotação [x_, y_, z_]
    MT = np.stack([x_, y_, z_], axis=1)

    # Expandir para matriz 12x12 via kron
    num_diag = estrutura.nodes_per_element
    T = np.kron(np.eye(2 * num_diag), MT)
    # T = np.zeros((num_elementos, 12, 12)) # np.kron(np.eye(4), MT[:, None, :, :])
    # T[:, 0:3, 0:3] = MT
    # T[:, 3:6, 3:6] = MT
    # T[:, 6:9, 6:9] = MT
    # T[:, 9:12, 9:12] = MT

    # Validar ortogonalidade
    assert np.allclose(np.einsum("eij,ejk->eik", MT, MT.transpose(0, 2, 1)),
                       np.eye(3), atol=1e-8), "MT não é ortogonal"
    assert np.allclose(np.linalg.det(MT), 1, atol=1e-8), "Determinante de MT não é 1"

    return T, MT

def transformation_matrix_ul(estrutura, d, numDOF, GLL, GLe):
    """
    Calcula as matrizes de transformação locais e globais para múltiplos elements,
    além de obter a deformação relativa de cada eixo local x.

    Args:
        estrutura (Estrutura): Objeto da classe Structure.
        coords (array_like): Coordenadas dos nós dos elements.

    Returns:
        T (array_like): Matriz de transformação global.
        MT (array_like): Matriz de transformação do elemento.
    """
    # Dados iniciais
    num_elementos = estrutura.num_elements
    coords = estrutura.coordinates

    # Vetor x_ (direção local x: do nó inicial para o final)
    x1g = coords[:, 0]
    x2g = coords[:, 1]

    dg = np.zeros((numDOF, 1))
    dg[GLL] = d
    de = dg[GLe]
    x1d = de[:, 0:3, 0]
    x2d = de[:, 6:9, 0]

    x1 = x1g + x1d
    x2 = x2g + x2d

    L_atual = np.linalg.norm(x2 - x1, axis=1)

    x_ = normalize(x2 - x1)

    # Auxiliar vector (global y)
    aux = np.array([0., 1., 0.])

    # Create a mask for elements parallel to the global y axis
    mask = (np.abs(x_[:, 0]) < 1e-9) & (np.abs(x_[:, 2]) < 1e-9)

    # For non parallel elements, determine z_aux
    z_aux = normalize(np.cross(x_, aux))

    # For parallel elements, aux = [0, 0, 1]
    aux_alt = np.array([0., 0., 1.])

    # Determine z_ based on mask
    z_ = np.where(mask[:, None], aux_alt, z_aux)

    # z_ vector (complete the right triangle z_ = x_ x y_)
    y_ = normalize(np.cross(z_, x_))

    # # VERIFICAÇÃO ADICIONAL: Para elements verticais, garantir que z_ está no plano horizontal
    # # Se x_ = [0, 0, ±1] e y_ = [0, 1, 0], então z_ deve ser [±1, 0, 0]
    # if np.any(mask):
    #     y_parallel = np.where(mask[:, None], 
    #                           normalize(np.cross(aux_alt, x_)),
    #                           y_)
    #     y_ = y_parallel

    # Empilhar matrizes de rotação [x_, y_, z_]
    MT = np.stack([x_, y_, z_], axis=1)

    # Expandir para matriz 12x12 via kron
    T = np.zeros((num_elementos, 12, 12)) # np.kron(np.eye(4), MT[:, None, :, :])
    T[:, 0:3, 0:3] = MT
    T[:, 3:6, 3:6] = MT
    T[:, 6:9, 6:9] = MT
    T[:, 9:12, 9:12] = MT

    # Validar ortogonalidade
    assert np.allclose(np.einsum("eij,ejk->eik", MT, MT.transpose(0, 2, 1)),
                       np.eye(3), atol=1e-8), "MT não é ortogonal"
    assert np.allclose(np.linalg.det(MT), 1, atol=1e-8), "Determinante de MT não é 1"

    return T, L_atual

def normalize(v):
    """
    Normaliza um array de vetores ao longo de um eixo.

    Args:
        v (array_like): Vetor a ser normalizado.

    Returns:
        array_like: Vetor normalizado.
    """
    norm = np.linalg.norm(v, axis=1, keepdims=True)

    # Evita divisão por zero
    nonzero_norm = np.where(norm == 0, 1, norm)
    return v / nonzero_norm

def check_convergence(d, Δd, λF, R,
                      tolerancias={'force': 1e-6, 'displ': 1e-6, 'energy': 1e-8},
                      ) -> bool:
    """
    Verifica se o incremento de deslocamento converge com base nodes critérios de força e deslocamento.

    Args:
    d (numpy array): Vetor de deslocamentos atuais.
    Δd (numpy array): Vetor de incrementos de deslocamento.
    F (numpy array): Vetor de forças atuais.
    R (numpy array): Vetor de forças residuais.
    tol_forca (float, optional): Tolerância para o critério de força.
    tol_deslocamento (float, optional): Tolerância para o critério de deslocamento.
    epsilon (float, optional): Valor epsilon para evitar divisão por zero.

    Returns:
    tuple: Um tuple contendo (convergencia, erro). 'convergencia' é um booleano que indica se o incremento
           convergiu. 'erro' é o erro máximo entre os critérios de força e deslocamento.
    """
    # Convergence status
    converged = False

    # Evaluate norms
    norm_R = np.linalg.norm(R)
    norm_F = np.linalg.norm(λF)
    norm_Δd = np.linalg.norm(Δd)
    norm_d = np.linalg.norm(d)

    # Force criterion
    rel_force = norm_R / max(norm_F, 1e-12)

    # Displacement criterion
    rel_displ = norm_Δd / max(norm_d, 1e-12)

    # Energy criterion
    energy = abs(np.dot(Δd.T, R)[0, 0])
    energy_ref = abs(np.dot(d.T, λF)[0, 0])
    rel_energy = energy / max(energy_ref, 1e-12)

    # Verify if any of the criteria is met
    conv_force = rel_force < tolerancias.get('force', 1e-6)
    conv_displ = rel_displ < tolerancias.get('displ', 1e-6)
    conv_energy = rel_energy < tolerancias.get('energy', 1e-6)

    # print(f"Forces: {rel_force:.6e} | Displacements: {rel_displ:.6e} | Energy: {rel_energy:.6e}")

    # Verify if at least two criteria are met
    checks = [conv_force, conv_displ, conv_energy]
    n_satisfied = sum(checks)

    # If at least two criteria are met, return True
    if n_satisfied >= 2:
        converged = True

    return converged

def get_deformed_coords(estrutura, coords, deslocamentos, GLe):
    """
    Returns deformed coordinates for linear, non-linear, and buckling analyses.
    Handles both Euler-Bernoulli (2 nodes) and Timoshenko (3 nodes) elements.

    Args:
        estrutura (Structure): Structure object.
        coords (np.ndarray): Original element coordinates. Shape (num_elements, num_nodes, 3).
        deslocamentos (dict): Dictionary containing displacement vectors.
        GLe (np.ndarray): Element degree of freedom indices.

    Returns:
        np.ndarray: Deformed coordinates.
    """
    # Initial data
    num_elements = estrutura.num_elements
    num_nodes = coords.shape[1]

    # Extract global displacements mapped to elements
    if estrutura.is_buckling:
        num_modes = deslocamentos['d'].shape[0]
        de_global = deslocamentos['d'][:, GLe].squeeze(-1)
    else:
        num_modes = 0
        de_global = deslocamentos['d'][GLe].squeeze(-1)

    def get_deformed_nodes(d, coords, scale_factor=None):
        """
        Internal function to apply displacements to coordinates.
        d: (num_elements, dofs_per_element)
        """
        # Extract translational displacements based on node count
        if num_nodes == 3:
            # Timoshenko (3 nodes): DOFs 0-2 (Start), 6-8 (Mid), 12-14 (End)
            u_start = d[:, 0:3]
            u_mid   = d[:, 6:9]
            u_end   = d[:, 12:15]
            
            # Stack to shape (num_elements, 3, 3)
            dg = np.stack([u_start, u_mid, u_end], axis=1)
            
        else:
            # Euler-Bernoulli (2 nodes): DOFs 0-2 (Start), 6-8 (End)
            u_start = d[:, 0:3]
            u_end   = d[:, 6:9]
            
            # Stack to shape (num_elements, 2, 3)
            dg = np.stack([u_start, u_end], axis=1)

        # Calculate scale factor automatically if not provided
        if scale_factor is None:
            # Avoid division by zero
            d_max = np.max(np.abs(dg))

            if d_max > 1e-10:
                # Normalize so max displacement is visible (e.g., 10% of structure or unit)
                # You might want to adjust this logic to be relative to structure size L
                scale_factor = 1.0 / d_max 
            else:
                scale_factor = 1.0
        
        # Apply displacements with scaling        
        cdef = coords + dg

        return cdef

    # Compute deformed coordinates
    if not estrutura.is_buckling:
        coordinates = get_deformed_nodes(de_global, coords)

    else:
        # Initialize buckling coordinates array
        # Shape: (num_modes, num_elements, num_nodes, 3)
        coordinates = np.zeros((num_modes, num_elements, 2, 3))

        # Calculate for each mode
        for idx in range(num_modes):
            coordinates[idx, :] = get_deformed_nodes(de_global[idx], coords)

    return coordinates


def deslocamentos_globais(estrutura, d, GLe):
    """
    Calcula os deslocamentos para todos os elementos.
    
    Args:
        analise (str): Tipo de análise ('linear', 'nao-linear', 'flambagem').
        deslocamentos_nodais (dict): Dicionário com os deslocamentos nodais.
        MT (np.ndarray): Matriz de transformação.
        modo (int): Modo de flambagem (apenas para analise='flambagem').
    
    Returns:
        deslocamentos_globais (np.ndarray): Matriz de deslocamentos globais.
    """
    # Número de elements
    dofs = estrutura.dofs_per_node
    nodes = estrutura.nodes_per_element
    num_elementos = estrutura.num_elements

    if not estrutura.is_buckling:
        # Deslocamentos globais por elements
        dg = d[GLe]
        
        # Deslocamentos globais reordenados
        global_displacement = dg.reshape(num_elementos, nodes, dofs)

    else:
        # Número de modos de flambagem
        num_modos = d.shape[0]
        
        # Deslocamentos globais por elements
        dg = d[:, GLe]
        
        # Deslocamentos globais reordenados
        global_displacement = dg.reshape(num_modos, num_elementos, nodes, dofs)

    return global_displacement

