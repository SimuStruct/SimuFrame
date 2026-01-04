# Built-in libraries
from typing import Tuple

# Third-party libraries
import numpy as np
from tqdm import tqdm
import numpy.typing as npt
from scipy.sparse import csc_array
from scipy.linalg import eigh, qr
from scipy.sparse.linalg import eigsh, factorized

# Local libraries
from .model import Structure
from SimuFrame.core.assembly import assemble_geometric_stiffness_matrix

def subspace_iteration(
    KE: csc_array,
    KG: csc_array,
    num_modes: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 42,
) -> Tuple[npt.NDArray[np.float64], np.ndarray]:
    """
    Subspace method for generalized eigenvalue problem (KE - λ KG) ϕ = 0.

    Args:
        KE (csc_array): Elastic stiffness matrix.
        KG (csc_array): Geometric stiffness matrix.
        num_modes (int): Number of eigenvalues/modes to compute.
        max_iter (int): Maximum number of iterations. Defaults to 40
        tol (float): Convergence tolerance for load factors. Defaults to 1e-6.
        compression_only: If True, returns only compression modes (λ > 0).
                         If False, returns both compression and tension modes.

    Returns:
        load_factors: Critical load multipliers:
                     - Positive: compression/buckling loads
                     - Negative: tension instability loads
                     - np.inf: mode not found or unstable structure
        mode_shapes: Corresponding eigenvectors.
    """
    n = KE.shape[0]

    # Validate inputs
    if num_modes < 1 or num_modes > n:
        raise ValueError(
            f"Invalid number of modes: {num_modes}. Must be between 1 and {n}"
        )

    # Subspace expansion
    subspace_size = min(2 * num_modes, num_modes + 8)

    # Initialize subspace with orthonormal vectors
    if seed is not None:
        rng = np.random.default_rng(seed)
        Q = qr(rng.random((n, subspace_size)), mode="economic")[0]
    else:
        Q = qr(np.random.rand(n, subspace_size), mode="economic")[0]

    # Storage for previous load factors
    final_load_factors = np.zeros(num_modes)

    # Factorize elastic stiffness matrix
    try:
        solve_KE = factorized(KE)
    except Exception as e:
        raise RuntimeError(
            f"Failed to factorize stiffness matrix KE. "
            f"Check for singularity or numerical issues. Error: {e}"
        )

    # Subspace iteration loop
    for iteration in tqdm(
        range(max_iter), desc="Structural stability analysis", leave=False
    ):
        # Solve linear system: K @ Y = KG @ Q
        Y = solve_KE(KG @ Q)

        # Orthogonalize via QR decomposition
        Q = qr(Y, mode="economic")[0]

        # Rayleigh-Ritz projection onto subspace
        KE_reduced = Q.T @ KE @ Q
        KG_reduced = Q.T @ KG @ Q

        # Ensure symmetry
        KE_reduced = 0.5 * (KE_reduced + KE_reduced.T)
        KG_reduced = 0.5 * (KG_reduced + KG_reduced.T)

        # Solve reduced eigenvalue problem
        eigenvalues, eigenvectors_reduced = eigh(-KG_reduced, KE_reduced)

        # Filter and convert eigenvalues to load factors
        valid_mask = np.abs(eigenvalues) > 1e-12  # & (eigenvalues < 1e+3)
        current_load_factors = np.full_like(eigenvalues, np.inf)
        current_load_factors[valid_mask] = 1.0 / eigenvalues[valid_mask]

        # Sort from smallest to largest load factor (most critical modes first)
        idx_sorted = np.argsort(np.abs(current_load_factors))
        sorted_load_factors = current_load_factors[idx_sorted]
        sorted_eigenvectors = eigenvectors_reduced[:, idx_sorted]

        # Extract only the requested number of modes
        req_load_factors = sorted_load_factors[:num_modes]

        # Project eigenvectors back to full space
        Q = Q @ sorted_eigenvectors

        # Convergence check
        if np.allclose(req_load_factors, final_load_factors, rtol=tol, atol=1e-9):
            final_load_factors = req_load_factors
            break

        # Update the final load factors
        final_load_factors = req_load_factors

    # Extract final mode shapes (corresponding to the num_modes smallest load factors)
    mode_shapes = Q[:, :num_modes]

    return final_load_factors, mode_shapes


def buckling_analysis(
    structure: Structure,
    properties: dict,
    numDOF: int,
    GLe: npt.NDArray[np.integer],
    GLL: npt.NDArray[np.bool_],
    T: npt.NDArray[np.float64],
    Ke: csc_array,
    fl: npt.NDArray[np.float64],
    num_modes: int = 5,
    seed: int = 42,
) -> tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Linear buckling analysis using subspace iteration method.

    Args:
        structure (Structure): Instance of the Structure class.
        properties (dict): Propriedades dos elements.
        numDOF (int): Número de graus de liberdade.
        GLe (list): Vinculações de cada elemento.
        GLL (list): Graus de liberdade livres.
        T (list): Matrizes de transformação.
        Ke (np.array): Matriz de rigidez reduzida.
        fl (list): Vetor de forças internas na configuração deformada.
        num_modos (int): Número de modos de flambagem a serem calculados (padrão: 5).

    Retorna:
        num_modos (int): Número de modos de flambagem encontrados.
        autovalores (list): Autovalores encontrados.
        d_flamb (list): Vetores de deslocamento para cada modo de flambagem.
    """
    # Validate inputs
    if num_modes < 1:
        raise ValueError(f"'num_modes' must be greater than 0, got {num_modes}.")

    # Initial data
    num_nodes = structure.num_nodes
    dofs_per_node = structure.dofs_per_node

    # Assemble geometric stiffness matrix
    Kg = assemble_geometric_stiffness_matrix(
        structure, properties, numDOF, GLe, GLL, T, fl
    )

    # Adjust number of modes if matrix is too small
    num_modes = min(num_modes, Ke.shape[0])

    # Solve generalized eigenvalue problem: (Ke + λ Kg) φ = 0
    try:
        # eigenvalues, eigenvectors = subspace_iteration(Ke, Kg, num_modes=num_modes, seed=seed)
        vals, vecs = eigsh(A=-Kg, M=Ke, k=num_modes, which='LM', tol=1e-12)
    except Exception as e:
        raise RuntimeError(
            f"Failed to solve generalized eigenvalue problem. "
            f"Structure may not be subjected to normal forces (no buckling possible).\n"
            f"Error: {e}"
        )

    # Compute eigenvalues (1.0 / vals)
    valid_indices = np.abs(vals) > 1e-12
    load_factors = np.full_like(vals, np.inf)
    load_factors[valid_indices] = 1.0 / vals[valid_indices]

    # Sort from smallest to largest load factor (most critical modes first)
    idx_sorted = np.argsort(np.abs(load_factors))
    eigenvalues = load_factors[idx_sorted]
    eigenvectors = vecs[:, idx_sorted]

    # Filter out infinite load factors (stable modes)
    finite_mask = np.isfinite(eigenvalues)
    num_modes = np.sum(finite_mask)

    if num_modes == 0:
        # No buckling modes were found.
        return 0, np.array([np.inf]), np.zeros((1, numDOF, 1))

    # Extract valid modes
    valid_eigvals = eigenvalues[finite_mask]
    valid_eigvecs = eigenvectors[:, finite_mask]

    # Expand eigenvectors to full space
    mode_shapes = np.zeros((num_modes, numDOF, 1))

    # Assign eigenvectors to corresponding modes
    mode_shapes[:, GLL, 0] = valid_eigvecs.T

    # Normalize buckling shapes
    for i in range(num_modes):
        mode = mode_shapes[i, :]

        # Reshape to group DOFs by node
        dofs_by_node = mode.reshape(num_nodes, dofs_per_node)

        # Extract only translational components
        desloc_translacional_por_no = dofs_by_node[:, :3]

        # Compute magnitude of translational displacement at each node
        translation_mag = np.linalg.norm(desloc_translacional_por_no, axis=1)

        # Find maximum translational displacement
        max_translation = np.max(translation_mag)

        # Normalize by this factor
        mode_shapes[i, :] /= max_translation

    return num_modes, valid_eigvals, mode_shapes
