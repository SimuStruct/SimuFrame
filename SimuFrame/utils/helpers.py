# Built-in libraries
from typing import Tuple, Optional, Dict, List

# Third-party libraries
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_array, csc_array

# Local libraries
from SimuFrame.core.model import Structure


def orientation_vector(
    structure: Structure,
    coords: npt.NDArray[np.float64],
    initial_coords: npt.NDArray[np.float64]
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Computes the cross-section orientation vector (reference vector).

    Args:
        structure (Structure): Instance of the Structure class.
        coords (np.ndarray): Current coordinates of the structure nodes.
        initial_coords (np.ndarray): Initial coordinates of the structure nodes.

    Returns:
        Dictionary containing reference vectors for 'undeformed' and 'deformed' states.
    """
    num_elements = structure.num_elements
    num_members = len(initial_coords)

    # Initialize reference vector dictionary
    ref_vectors = {
        'undeformed': np.zeros((num_members, 3)),
        'deformed': np.zeros((num_elements, 3))
    }

    # Coordinate states to process
    coordinate_states = {
        'undeformed': np.copy(initial_coords),
        'deformed': np.copy(coords)
    }

    for state, coord in coordinate_states.items():
        # Local x-axis vector
        x_ = normalize(coord[:, -1] - coord[:, 0])

        # Check for elements aligned with the global Z-axis (x and y components near zero)
        mask = (np.abs(x_[:, 0]) < 1e-9) & (np.abs(x_[:, 1]) < 1e-9)

        # Assign reference vectors: [1, 0, 0] for vertical elements, [0, 0, 1] otherwise
        ref_vectors[state] = np.where(mask[:, None], np.array([1., 0., 0.]), np.array([0., 0., 1.]))

    return ref_vectors


def get_local_displacements(
    num_dofs: int,
    free_dofs_mask: npt.NDArray[np.bool_],
    element_dofs: npt.NDArray[np.integer],
    transformation_matrix: npt.NDArray[np.float64],
    displacements: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Maps reduced displacements to full local element displacements.

    Args:
        num_dofs (int): Total number of degrees of freedom.
        free_dofs_mask (bool): Boolean mask indicating free DOFs.
        element_dofs (np.ndarray): Array of element DOF indices.
        transformation_matrix (np.ndarray): Global transformation matrix.
        displacements (np.ndarray): Array of reduced displacements.

    Returns:
        Array of local element displacements.
    """
    # Initialize full displacement array
    d = np.zeros((num_dofs, 1))

    # Assign reduced displacements to free DOFs
    d[free_dofs_mask] = displacements

    # Transform global displacements to local system
    return transformation_matrix @ d[element_dofs]


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
        global_element_stiffness (np.ndarray): Assembled global stiffness matrix.
        total_dofs (int): Total number of degrees of freedom in the system.
        element_dofs (np.ndarray): Array mapping element local DOFs to global indices.

    Returns:
        K (csc_array): Global stiffness matrix in CSC format.
    """
    dofs_per_el = structure.dofs_per_element
    num_elements = structure.num_elements

    # Flatten stiffness data
    data = global_element_stiffness.flatten()

    # Generate row indices
    rows = np.broadcast_to(
        element_dofs[:, :, None],
        (num_elements, dofs_per_el, dofs_per_el)
    ).flatten()

    # Generate column indices
    cols = np.broadcast_to(
        element_dofs[:, None, :],
        (num_elements, dofs_per_el, dofs_per_el)
    ).flatten()

    # Create sparse matrix
    kg = coo_array((data, (rows, cols)), shape=(total_dofs, total_dofs)).tocsc()

    return kg


def extract_element_data(structure: Structure) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, Dict]:
    """
    Extracts geometric and constitutive properties from structural elements.

    Args:
        structure (Structure): Instance of the Structure class.

    Returns:
        Tuple containing:
            - Element coordinates
            - Member coordinates
            - Connectivity matrix
            - Dictionary of properties (L, A, E, etc.)
    """
    # Retrieve raw object lists
    elements = list(structure.elements.values())
    members = list(structure.original_members.values())

    # Extract connectivity IDs
    conec = np.array([[node.id for node in e.conec] for e in elements], dtype=int)

    # Extract element coords
    elem_coords = np.array([[node.coord for node in e.conec] for e in elements], dtype=float)

    # Calculate lengths
    lengths = np.array([e.get_element_length() for e in elements], dtype=float)

    # Extract original members coords (initial and final only)
    member_coords = np.array([[m['nodes'][0].coord, m['nodes'][-1].coord] for m in members], dtype=float)

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
        "L": lengths,
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

    return elem_coords, member_coords, conec, properties


def static_condensation(
        structure: Structure,
        k_local: npt.NDArray[np.float64],
        f_local: npt.NDArray[np.float64] | None = None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Performs static condensation for released degrees of freedom (hinges).

    Args:
        structure (Structure): Instance of the Structure class.
        k_local (np.array): Local element stiffness matrices.
        f_local (np.array): Local element equivalent force vectors.

    Returns:
        tuple:
            - k_cond (np.array): Condensed element stiffness matrices.
            - f_cond (np.array): Condensed element equivalent force vectors.
    """
    if not structure.condensation_data:
        # If no condensation needed, initialize f_local if missing and return
        if f_local is None:
            f_local = np.zeros((structure.num_elements, structure.dofs_per_element, 1))
        return k_local, f_local

    # Initialize force vector if None
    if f_local is None:
        f_local = np.zeros((structure.num_elements, structure.dofs_per_element, 1))

    # Create a copy of original arrays
    k_condensed = k_local.copy()
    f_condensed = f_local.copy()

    # Iterate over condened elements
    for element in structure.condensation_data:
        elem_idx = element['id']
        elim = element['elim_indices']
        kept = element['kept_indices']

        # Extract local matrices and forces arrays
        ke = k_condensed[elem_idx]
        fe = f_condensed[elem_idx]

        # Partition matrices
        # mm: retained-retained, ee: eliminated-eliminated, me/em: coupling
        k_mm = ke[np.ix_(kept, kept)]
        k_me = ke[np.ix_(kept, elim)]
        k_em = ke[np.ix_(elim, kept)]
        k_ee = ke[np.ix_(elim, elim)]

        # Fallback for singular matrices
        try:
            k_ee_inv = np.linalg.inv(k_ee)
        except np.linalg.LinAlgError:
            k_ee_inv = np.linalg.pinv(k_ee)

        # Store k_me @ k_ee_inv
        k_matrix = k_me @ k_ee_inv

        # Calculate condensed arrays
        k_cond = k_mm - k_matrix @ k_em

        # Condense forces only in linear analysis
        f_m = fe[kept]
        f_e = fe[elim]
        f_cond = f_m - k_matrix @ f_e

        # Clear original arrays
        ke.fill(0.0)
        fe.fill(0.0)

        # Update local matrices
        ke[np.ix_(kept, kept)] = k_cond
        fe[kept] = f_cond

        # Add a small value to the diagonal to avoid singular matrices
        max_diag = np.max(np.abs(np.diag(k_cond)))
        ke[(elim, elim)] = 1e-6 * max_diag

    return k_condensed, f_condensed

def transformation_matrix(
        structure: Structure,
        coords: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculates local and global transformation matrices for multiple elements.

    Args:
        structure (Structure): Instance of the Structure class.
        coords (np.ndarray): Coordinates of the nodes of the elements.

    Returns:
        Tuple containing:
            - T (np.ndarray): Global transformation matrix.
            - MT (np.ndarray): Element transformation matrix.
    """
    # Local x-axis normalized
    x_ = normalize(coords[:, -1] - coords[:, 0])

    # Auxiliary vector (Global Y)
    aux = np.array([0., 1., 0.])

    # Identify elements parallel to global Y
    mask = (np.abs(x_[:, 0]) < 1e-9) & (np.abs(x_[:, 2]) < 1e-9)

    # For non parallel elements, determine z_aux
    z_aux = normalize(np.cross(x_, aux))

    # For parallel elements, aux = [0, 0, 1]
    aux_alt = np.array([0., 0., 1.])

    # Determine local z-axis
    z_ = np.where(mask[:, None], aux_alt, z_aux)

    # Determine local y-axis (complete the triad)
    y_ = normalize(np.cross(z_, x_))

    # Stack rotation matrices [x, y, z]
    el_rot_matrix = np.stack([x_, y_, z_], axis=1, dtype=np.float64)

    # Expand to full element DOF size using Kronecker product
    num_diag = structure.nodes_per_element
    global_rot_matrix = np.kron(np.eye(2 * num_diag), el_rot_matrix)

    # Check orthogonality: R * R.T = I
    assert np.allclose(np.einsum("eij,ejk->eik", el_rot_matrix, el_rot_matrix.transpose(0, 2, 1)),
                       np.eye(3), atol=1e-8), "Transformation matrix is not orthogonal"
    assert np.allclose(np.linalg.det(el_rot_matrix), 1, atol=1e-8), "Determinant of transformation matrix is not 1"

    return global_rot_matrix, el_rot_matrix

def normalize(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Normalizes an array of vectors along the last axis.

    Args:
        v (np.ndarray): Input vector(s).

    Returns:
        Normalized vector(s).
    """
    norm = np.linalg.norm(v, axis=1, keepdims=True)

    # Avoid division by zero
    nonzero_norm = np.where(norm == 0, 1, norm)
    return v / nonzero_norm

def check_convergence(
    d: npt.NDArray[np.float64],
    Δd: npt.NDArray[np.float64],
    λF: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    arc_length_criteria: bool = False,
    tols={'force': 1e-8, 'displ': 1e-8, 'energy': 1e-10},
) -> Tuple[bool, float, float, float]:
    """
    Checks convergence based on force, displacement, and energy criteria.

    Args:
        d (np.ndarray): Current displacement vector.
        Δd (np.ndarray): Incremental displacement vector.
        F (np.ndarray): Current force vector.
        R (np.ndarray): Residual force vector.
        arc_length_criterion (bool): Arc-length criteria, if provided (for arc-length method only).
        tols (dict, optional): Tolerances for the force, displacement, and energy criteria.

    Returns:
        Tuple (converged, rel_force_error, rel_displ_error, rel_energy_error).
    """
    if tols is None:
        tols = {'force': 1e-8, 'displ': 1e-8, 'energy': 1e-10}

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
    conv_force = rel_force < tols.get('force', 1e-6)
    conv_displ = rel_displ < tols.get('displ', 1e-6)
    conv_energy = rel_energy < tols.get('energy', 1e-8)

    # Verify if at least two criteria are met
    checks = [conv_force, conv_displ, conv_energy, arc_length_criteria]
    converged = (sum(checks) >= 2)

    return bool(converged), float(rel_force), float(rel_displ), float(rel_energy)

def get_deformed_coords(
    structure: Structure,
    coords: npt.NDArray[np.float64],
    displacements: Dict[str, npt.NDArray[np.float64]],
    element_dofs: npt.NDArray[np.integer]):
    """
    Returns deformed coordinates for linear, non-linear, and buckling analyses.
    Handles both Euler-Bernoulli (2 nodes) and Timoshenko (3 nodes) elements.

    Args:
        structure (Structure): Structure object.
        coords (np.ndarray): Original element coordinates. Shape (num_elements, num_nodes, 3).
        displacements (dict): Dictionary containing displacement vectors.
        element_dofs (np.ndarray): Element degree of freedom indices.

    Returns:
        np.ndarray: Deformed coordinates arrays.
    """
    # Initial data
    num_nodes = coords.shape[1]
    num_elements = structure.num_elements

    # Extract global displacements mapped to elements
    if structure.is_buckling:
        num_modes = displacements['d'].shape[0]
        de_global = displacements['d'][:, element_dofs].squeeze(-1)
    else:
        num_modes = 0
        de_global = displacements['d'][element_dofs].squeeze(-1)

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
                # Normalize max displacement to unit length
                scale_factor = 1.0 / d_max
            else:
                scale_factor = 1.0

        # Apply displacements with scaling
        cdef = coords + dg

        return cdef

    # Compute deformed coordinates
    if not structure.is_buckling:
        coordinates = get_deformed_nodes(de_global, coords)

    else:
        # Initialize buckling coordinates array
        coordinates = np.zeros((num_modes, num_elements, num_nodes, 3))

        # Calculate for each mode
        for idx in range(num_modes):
            coordinates[idx, :] = get_deformed_nodes(de_global[idx], coords)

    return coordinates


def get_global_displacements(
    structure: Structure,
    displacements: npt.NDArray[np.float64],
    element_dofs: npt.NDArray[np.integer]):
    """
    Reshapes global displacement vector into element-wise format.

    Args:
        structure (Structure): Instance of the Structure class.
        displacements (np.ndarray): Global displacement vector.
        element_dofs (np.ndarray): Element degree of freedom indices.

    Returns:
        Array of global displacements organized by element.
    """
    # Initial data
    dofs = structure.dofs_per_node
    nodes_per_el = structure.nodes_per_element
    num_elements = structure.num_elements

    if not structure.is_buckling:
        # Extract and reshape for standard analysis
        elem_displ = displacements[element_dofs]
        global_displacement = elem_displ.reshape(num_elements, nodes_per_el, dofs)

    else:
        # Extract and reshape for buckling (multiple modes)
        num_modes = displacements.shape[0]
        elem_displ = displacements[:, element_dofs]
        global_displacement = elem_displ.reshape(num_modes, num_elements, nodes_per_el, dofs)

    return global_displacement
