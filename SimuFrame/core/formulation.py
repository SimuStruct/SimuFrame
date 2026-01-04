# Built-in libraries
from typing import Tuple
from functools import lru_cache

# Third-party libraries
import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_array

# Local libraries
from .model import Structure
from SimuFrame.utils.helpers import (
    assemble_sparse_matrix,
    get_local_displacements,
    static_condensation,
)


@lru_cache(maxsize=128)
def get_gauss_points(n_points: int = 3):
    """Cache to store Gauss points (constants).

    Args:
        n_points (int, optional): Number of Gauss points. Defaults to 3.
    """
    return np.polynomial.legendre.leggauss(n_points)

def shape_functions(
        structure: Structure,
        xi: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Computes the shape functions for each element.

    Args:
        structure (Structure): Instance of the Structure class.
        connectivity (np.ndarray): Integer array of shape (num_elements, num_nodes_per_element).

    Returns:
        np.ndarray: Array of shape functions for each element.
    """
    # Initial data
    num_elements = structure.num_elements

    # Shape functions for Timoshenko beams (B32)
    if structure.is_quadratic:
        # Quadratic shape functions
        Nq = np.array([0.5 * xi * (xi - 1), 1.0 - xi**2, 0.5 * xi * (xi + 1)])

        # Tile shape functions for each element
        Nq = np.tile(Nq, (num_elements, 1))

        # Create array of shape functions
        N = np.zeros((num_elements, 6, 18))
        N[:, 0, 0::6] = Nq
        N[:, 1, 1::6] = Nq
        N[:, 2, 2::6] = Nq
        N[:, 3, 3::6] = Nq
        N[:, 4, 4::6] = Nq
        N[:, 5, 5::6] = Nq

    # Shape functions for Euler-Bernoulli beams and trusses (B33, T3D)
    else:
        ...
    return Nq, N

def shape_derivatives(
        structure: Structure,
        xi: float
    ) -> npt.NDArray[np.float64]:
    # Shape functions for Timoshenko beams (B32)
    if structure.is_quadratic:
        # Derivatives of quadratic shape functions
        dN = np.array([xi - 0.5, -2.0 * xi, xi + 0.5])

        # Tile shape functions for each element
        dN = np.tile(dN, (structure.num_elements, 1))

    else:
        ...

    return dN

def timoshenko_kinematics(
    structure: Structure,
    xi: float,
    invJ: npt.NDArray[np.float64],
    element_displ: npt.NDArray[np.float64],
    nonlinear: bool,
    integration: str
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute linear and nonlinear kinematics for Timoshenko beams.

    Args:
        structure (Structure): Instance of the Structure class.
        xi (float): Gauss point location along the length.
        invJ (np.ndarray): Array with the inverse Jacobian matrix.
        element_displ (np.ndarray): Array with the nodal displacements.
        nonlinear (bool): If True, calculates the nonlinear matrix BNL.
        integration (str): Integration type ("full" or "reduced").
    Returns:
        tuple:
            - BL (np.ndarray): Linear kinematic matrix.
            - BNL (np.ndarray): Nonlinear kinematic matrix.
            - Ge (np.ndarray): Gradient of shape functions.
    """
    # Initial data
    num_elements = structure.num_elements
    dofs = structure.dofs_per_element

    # Initialize matrices: [BL], [BNL] and [Ge]
    BL = np.zeros((num_elements, 6, dofs))
    BNL = np.zeros((num_elements, 6, dofs))
    Ge = np.zeros((num_elements, 6, dofs))

    # Get shape derivatives
    dN = shape_derivatives(structure, xi)
    dN_dx = dN * invJ

    # Determine if the component is for bending or shear
    if integration == "full":
        # Assemble bending kinematic matrix
        BL[:, 0, 0::6] = dN_dx
        BL[:, 1, 5::6] = dN_dx
        BL[:, 2, 4::6] = dN_dx
        BL[:, 3, 3::6] = dN_dx

        # Assemble gradient and nonlinear kinematic matrices
        if nonlinear:
            # Displacement gradients
            # Ge[:, 0, 0::6] = dN_dx  # du/dx
            Ge[:, 1, 1::6] = dN_dx  # dv/dx
            Ge[:, 2, 2::6] = dN_dx  # dw/dx

            # Nonlinear kinematic matrix, {[Ge]{de}}^T @ [Ge]
            BbNL = (Ge @ element_displ).transpose(0, 2, 1) @ Ge

            # Assemble nonlinear kinematic matrix
            BNL[:, 0:1, :] = BbNL

    elif integration == "reduced":
        # Get shape functions
        N, _ = shape_functions(structure, xi)

        # Assemble shear kinematic matrix
        BL[:, 4, 1::6] = dN_dx
        BL[:, 4, 5::6] = -N
        BL[:, 5, 2::6] = dN_dx
        BL[:, 5, 4::6] = N

    return BL, BNL, Ge


def euler_kinematics(
    structure: Structure,
    xi: float,
    L: npt.NDArray[np.float64],
    element_displ: npt.NDArray[np.float64],
    nonlinear: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute linear and nonlinear kinematics for Euler-Bernoulli beams.

    Args:
        structure (Structure): Instance of the Structure class.
        xi (float): Gauss point location along the length.
        element_displ (np.ndarray): Array with the element nodal displacements.
        nonlinear (bool): Nonlinear analysis flag.
        integration (str): Integration type ("full" or "reduced").
    Returns:
        tuple:
            - BL (np.ndarray): Linear kinematic matrix.
            - BNL (np.ndarray): Nonlinear kinematic matrix.
            - Ge (np.ndarray): Gradient of shape functions.
    """
    # Initial data
    num_elements = structure.num_elements
    dofs = structure.dofs_per_element

    # Initialize matrices: [BL], [BNL] and [Ge]
    BL = np.zeros((num_elements, 4, dofs))
    BNL = np.zeros((num_elements, 4, dofs))
    Ge = np.zeros((num_elements, 4, dofs))

    # Inverse of Jacobian matrix and its square
    invJ = 2.0 / L
    invJ2 = invJ**2

    # Assemble linear kinematic matrix
    # Axial and torsional terms (dNu/dξ and dNθx/dξ)
    BL[:, 0, 0] = -invJ * 0.5
    BL[:, 0, 6] = invJ * 0.5
    BL[:, 3, 3] = -invJ * 0.5
    BL[:, 3, 9] = invJ * 0.5

    # Bending about z-axis (d²Nv/dξ² and d²Nθz/dξ²)
    BL[:, 1, 1] = invJ2 * (3 / 2) * xi
    BL[:, 1, 5] = invJ2 * (L / 4) * (3 * xi - 1)
    BL[:, 1, 7] = -BL[:, 1, 1]
    BL[:, 1, 11] = invJ2 * (L / 4) * (3 * xi + 1)

    # Bending about y-axis (d²Nw/dξ² and d²Nθy/dξ²)
    BL[:, 2, 2] = BL[:, 1, 1]
    BL[:, 2, 4] = -BL[:, 1, 5]
    BL[:, 2, 8] = BL[:, 1, 7]
    BL[:, 2, 10] = -BL[:, 1, 11]

    # Assemble gradient and nonlinear kinematic matrices
    if nonlinear:
        # Axial terms (dNu/dξ)
        Ge[:, 0, 0] = -invJ * 0.5
        Ge[:, 0, 6] = invJ * 0.5

        # Bending about z-axis (dNv/dξ and dNθz/dξ)
        Ge[:, 1, 1] = invJ * (3 / 4) * (xi**2 - 1)
        Ge[:, 1, 5] = invJ * (L / 8) * (3 * xi**2 - 2 * xi - 1)
        Ge[:, 1, 7] = -invJ * (3 / 4) * (xi**2 - 1)
        Ge[:, 1, 11] = invJ * (L / 8) * (3 * xi**2 + 2 * xi - 1)

        # Bending about y-axis (dNw/dξ and dNθy/dξ)
        Ge[:, 2, 2] = Ge[:, 1, 1]
        Ge[:, 2, 4] = -Ge[:, 1, 5]
        Ge[:, 2, 8] = Ge[:, 1, 7]
        Ge[:, 2, 10] = -Ge[:, 1, 11]

        # Torsional terms (dNθx/dξ)
        # Ge[:, 3, 3] = -J * 0.5 * rp
        # Ge[:, 3, 9] =  J * 0.5 * rp

        # Nonlinear kinematic matrix, {[Ge]{de}}^T @ [Ge]
        BbNL = (Ge @ element_displ).transpose(0, 2, 1) @ Ge

        # Assemble nonlinear kinematic matrix
        BNL[:, 0:1, :] = BbNL

    return BL, BNL, Ge


def compute_element_stiffness(
    de: npt.NDArray[np.float64],
    structure: Structure,
    properties: dict,
    nonlinear: bool
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the element stiffness matrix and force vector for a beam element.

    Args:
        de (np.ndarray): Element displacement vector.
        structure (Structure): Instance of the Structure class.
        properties (dict): Dictionary containing element properties.
        nonlinear (bool): Nonlinear analysis flag.

    Returns:
        tuple:
            - kt (np.ndarray): Element stiffness matrix.
            - fe (np.ndarray): Element force vector.
    """
    # Initial data
    dofs = structure.dofs_per_element
    num_elements = structure.num_elements

    # Initialize matrices
    kt = np.zeros((num_elements, dofs, dofs))
    fe = np.zeros((num_elements, dofs, 1))

    # Geometric and constitutive properties
    L = properties["L"]
    k = properties["k"]
    EA = properties["E"] * properties["A"]
    GA = properties["G"] * properties["A"]
    EIy = properties["E"] * properties["Iy"]
    EIz = properties["E"] * properties["Iz"]
    GIt = properties["G"] * properties["It"]

    # Jacobian matrix and its inverse
    detJ = (L / 2).reshape(-1, 1, 1)
    invJ = (2 / L).reshape(-1, 1)

    # Verify if the beam element is quadratic (Timoshenko) or cubic (Euler-Bernoulli)
    if structure.is_quadratic:
        # Get Gauss points and weights (full integration)
        points, weights = get_gauss_points(n_points=3)

        # Total constitutive matrix
        D = np.zeros((num_elements, 6, 6))
        D[:, 0, 0] = EA
        D[:, 1, 1] = EIz
        D[:, 2, 2] = EIy
        D[:, 3, 3] = GIt
        D[:, 4, 4] = k * GA
        D[:, 5, 5] = k * GA

        # Bending kinematic matrices
        for xi, wi in zip(points, weights):
            BL, BNL, Ge = timoshenko_kinematics(
                structure, xi, invJ, de, nonlinear, integration="full"
            )

            # Strain-displacement matrix, [B]
            B = BL + BNL

            # Incremental strain-displacement matrix, [Bε]
            Bε = BL + 0.5 * BNL

            # Large-displacements stiffness matrix, [kl]
            kl = np.einsum("eji,ejk,ekl->eil", B, D, B, optimize="optimal")

            # Internal forces, [S]
            S = np.einsum("eij,ejk,ekl->eil", D, Bε, de, optimize="optimal")

            # Geometric stiffness matrix, [kg]
            if nonlinear:
                kg = S[:, 0:1, :] * np.einsum(
                    "eji,ejk->eik", Ge, Ge, optimize="optimal"
                )
            else:
                kg = 0

            # Add bending stiffness to tangent matrix, [kt]
            kt += wi * detJ * (kl + kg)

            # Add bending forces to internal forces vector, [fe]
            fe += (wi * detJ) * np.einsum("eji,ejk->eik", B, S, optimize="optimal")

        # Reduced integration for shear terms
        points, weights = get_gauss_points(n_points=2)

        # Shear kinematic matrices
        for xi, wi in zip(points, weights):
            B = timoshenko_kinematics(structure, xi, invJ, de, nonlinear, integration="reduced")[0]

            # Large-displacements stiffness matrix, [kl]
            kl = np.einsum("eji,ejk,ekl->eil", B, D, B, optimize="optimal")

            # Internal forces, [S]
            S = np.einsum("eij,ejk,ekl->eil", D, B, de, optimize="optimal")

            # Add shear stiffness to tangent matrix, [kt]
            kt += (wi * detJ) * kl

            # Add shear forces to internal forces vector, [fe]
            fe += (wi * detJ) * np.einsum("eji,ejk->eik", B, S, optimize="optimal")

    else:
        # Gauss points and weights
        points, weights = get_gauss_points(n_points=3)

        # Constitutive matrix
        D = np.zeros((num_elements, 4, 4))
        D[:, 0, 0] = EA
        D[:, 1, 1] = EIz
        D[:, 2, 2] = EIy
        D[:, 3, 3] = GIt

        for xi, wi in zip(points, weights):
            BL, BNL, Ge = euler_kinematics(structure, xi, L, de, nonlinear)

            # Strain-displacement matrix, [B]
            B = BL + BNL

            # Incremental strain-displacement matrix, [Bε]
            Bε = BL + 0.5 * BNL

            # Large-displacements stiffness matrix, [kl]
            kl = np.einsum("eji,ejk,ekl->eil", B, D, B, optimize="optimal")

            # Internal forces, [S]
            S = np.einsum("eij,ejk,ekl->eil", D, Bε, de, optimize="optimal")

            # Geometric stiffness matrix, [kg]
            if nonlinear:
                kg = S[:, 0:1, :] * np.einsum(
                    "eji,ejk->eik", Ge, Ge, optimize="optimal"
                )
            else:
                kg = 0

            # Tangent stiffness matrix, [kt]
            kt += (wi * detJ) * (kl + kg)

            # Internal forces vector, [fe]
            fe += (wi * detJ) * np.einsum("eji,ejk->eik", B, S, optimize="optimal")

    # Apply static condensation if necessary
    kt, fe = static_condensation(structure, kt, fe)

    return kt, fe


def global_analysis(
    displacements: npt.NDArray[np.float64],
    structure: Structure,
    properties: dict,
    num_dofs: int,
    free_dofs_mask: npt.NDArray[np.bool_],
    element_dofs: npt.NDArray[np.integer],
    transformation_matrix: npt.NDArray[np.float64],
    nonlinear: bool
) -> tuple[csc_array, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the global stiffness matrix and internal force vector for all elements.
    Assembles the global stiffness matrix and the global internal force vector.

    Args:
        displacements (np.ndarray): Array of global displacements.
        structure (Structure): Instance of the Structure class.
        properties (dict): Dictionary with the properties of the elements.
        num_dofs (int): Total number of degrees of freedom.
        free_dofs_mask (np.ndarray): Mask of free degrees of freedom.
        element_dofs (np.ndarray): Maps element local DOFs to global indices.
        transformation_matrix (np.ndarray): Local-to-global transformation matrices.
        nonlinear (bool): If True, calculates the geometric stiffness matrix.

    Returns:
        tuple:
            - KG_reduced (np.ndarray): Reduced global stiffness matrix.
            - Fint (np.ndarray): Global internal force vector.
    """
    # Initialize global force vector, {F}
    F = np.zeros((num_dofs, 1))

    # Get local displacements
    de = get_local_displacements(num_dofs, free_dofs_mask, element_dofs, transformation_matrix, displacements)

    # Compute element stiffness matrix and force vector
    ke, fe = compute_element_stiffness(de, structure, properties, nonlinear)

    # Convert to global system
    ke_global = np.einsum("eji,ejk,ekl->eil", transformation_matrix, ke, transformation_matrix, optimize="optimal")
    fe_global = np.einsum("eji,ejk->eik", transformation_matrix, fe, optimize="optimal")

    # Assemble global force vector, {F}
    np.add.at(F, element_dofs, fe_global)

    # Assemble global stiffness matrix, [KG]
    KG = assemble_sparse_matrix(structure, ke_global, num_dofs, element_dofs)

    # Apply boundary conditions
    KG_reduced = KG[np.ix_(free_dofs_mask, free_dofs_mask)]
    Fint = F[free_dofs_mask]
    Ra = F[~free_dofs_mask]

    return KG_reduced, Fint, Ra, fe, de
