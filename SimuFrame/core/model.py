# Built-in libraries
from abc import ABC, abstractmethod
from typing import List, Dict, Sequence, Union, Any, Tuple

# Third-party libraries
import numpy as np
import pyvista as pv
import numpy.typing as npt


class Node:
    """
    Represents a structural node (joint) in 3D space.

    Attributes:
        id (int): Unique identifier for the node.
        coord (np.ndarray): [x, y, z] coord.
        boundary_conditions (list[int]): List of constrained local degrees of freedom indices
                                         (e.g., [0, 1, 5] for fixed translation X, Y and rotation Z).
    """
    def __init__(self, node_id: int, coord: Sequence[float]):
        self.id = node_id
        self.coord = np.array(coord, dtype=float)
        self.boundary_conditions: List[int] = []

class Element:
    """
    Represents a structural element (member) in 3D space.

    Attributes:
        id (int): Unique identifier for the element.
        conec (tuple[Node, Node]): Tuple of connected nodes (start, end).
        section (SectionProperties): Section properties object.
        hinges (tuple[list[int], list[int]]): Tuple of lists of hinges (start, end).
        distributed_load (np.ndarray): Distributed load array.
    """
    def __init__(self, element_id: int, conec, secao, hinge):
        self.id = element_id
        self.conec = conec
        self.section = secao
        self.hinges = hinge
        self.distributed_load = np.zeros((len(conec), 3), dtype=float)

    def get_element_length(self):
        return np.linalg.norm(self.conec[-1].coord - self.conec[0].coord)

class Material:
    """
    Isotropic linear elastic material properties.

    Attributes:
        E (float): Young's Modulus (Elastic Modulus).
        nu (float): Poisson's Ratio.
        G (float): Shear Modulus (derived automatically if not provided).
    """
    def __init__(self, elastic_modulus: float, poisson_ratio: float = 0.0):
        self.E = elastic_modulus

        # Verify Poisson's ratio
        if 0 <= poisson_ratio <= 0.5:
            self.nu = poisson_ratio
        else:
            raise ValueError("Poisson's ratio must be between 0 and 0.5.")

        # Shear modulus (G)
        self.G = self.E / (2 * (1 + self.nu))

class SectionProperties(ABC):
    """
    Abstract Base Class for cross-section geometric properties.

    Attributes:
        material (Material): Associated material object.
        A (float): Cross-sectional Area.
        Ay, Az (float): Shear Areas in local Y and Z axes.
        Iy, Iz (float): Moments of Inertia about local Y and Z axes.
        Ip (float): Polar Moment of Inertia.
        It (float): Torsional Constant (Saint-Venant).
        ry, rz (float): Radii of Gyration about local Y and Z axes.
        rp (float): Polar Radius of Gyration.
    """
    def __init__(self, material):
        self.material = material

        # Initialize geometric properties to zero
        self.A = 0.0
        self.Ay = 0.0
        self.Az = 0.0
        self.k = 0.0
        self.Iy = 0.0
        self.Iz = 0.0
        self.Ip = 0.0
        self.It = 0.0
        self.ry = 0.0
        self.rz = 0.0
        self.rp = 0.0

        # Compute geometric properties
        self.compute_properties()

    @abstractmethod
    def compute_properties(self):
        """Calculates geometric and constitutive properties based on dimensions."""
        pass

    @abstractmethod
    def generate_polydata(self) -> Union[pv.PolyData, pv.DataObject]:
        """Generate a PyVista PolyData object representing the cross-section."""
        pass

class Rectangular(SectionProperties):
    """
    Represents a solid rectangular cross-section.

    Attributes:
        b (float): Width of the section.
        h (float): Height of the section.
        geometry_type (str): Identifier for the section shape ('rectangular').
        name (str): Label for the section.
    """
    def __init__(self, material: Material, width: float, height: float) -> None:
        self.b = width
        self.h = height
        self.geometry_type = 'rectangular'
        self.name = f"Rectangular {self.b * 100:.2f} x {self.h * 100:.2f} [cm]"

        # Initialize parent class
        super().__init__(material)

    def compute_properties(self):
        try:
            # Base parameters
            b, h = self.b, self.h

            # Define the larger and smaller dimensions for Saint-Venant torsion
            hx, hy = min(b, h), max(b, h)

            # Sectional area and shear factor
            self.A = b * h
            self.k = 0.85

            # Second moment of area
            self.Iy = b * h**3 / 12
            self.Iz = h * b**3 / 12
            self.Ip = self.Iy + self.Iz

            # Radii of gyration
            self.ry = np.sqrt(self.Iy / self.A)
            self.rz = np.sqrt(self.Iz / self.A)
            self.rp = np.sqrt(self.ry**2 + self.rz**2)

            # Torsional constant (Roark's formula approximation)
            self.It = 0.0625 * hy * hx**3 * (16/3 - 3.36 * hx / hy * (1 - hx**4 / (12 * hy**4)))

        except Exception as e:
            raise ValueError(f"Error attempting to calculate section properties: {str(e)}")


    def generate_polydata(self) -> pv.PolyData:
        # Define vertices of the section (x, y, z)
        vertices = np.array([
            [-self.b/2, -self.h/2, 0.0],
            [ self.b/2, -self.h/2, 0.0],
            [ self.b/2,  self.h/2, 0.0],
            [-self.b/2,  self.h/2, 0.0]
        ], dtype=np.float64)

        # Define section face
        faces = np.array([[4, 0, 1, 2, 3]], dtype=int)

        # Return PolyData object
        return pv.PolyData(vertices, faces)

class RHS(SectionProperties):
    """
    Represents a rectangular hollow cross-section.

    Attributes:
        b (float): Width of the section.
        h (float): Height of the section.
        t (float): Thickness of the section.
        geometry_type (str): Identifier for the section shape ('rhs').
        name (str): Label for the section.
    """
    def __init__(self, material: Material, width: float, height: float, thickness: float):
        self.b = width
        self.h = height
        self.t = thickness
        self.geometry_type = 'rhs'
        self.name = f"RHS {self.b * 100:.2f} x {self.h * 100:.2f} / {self.t * 100:.2f} [cm]"

        # Initialize parent class
        super().__init__(material)

    def compute_properties(self):
        try:
            # Base parameters
            b, h, t = self.b, self.h, self.t
            bi, hi = b - 2*t, h - 2*t

            # Sectional area and shear factor
            self.A = (b * h) - (bi * hi)
            self.k = 0.44

            # Second moment of area
            self.Iy = ((b * h**3) - (bi * hi**3)) / 12
            self.Iz = ((h * b**3) - (hi * bi**3)) / 12
            self.Ip = self.Iy + self.Iz

            # Radii of gyration
            self.ry = np.sqrt(self.Iy / self.A)
            self.rz = np.sqrt(self.Iz / self.A)
            self.rp = np.sqrt(self.ry**2 + self.rz**2)

            # Torsional constant (Bredt's formula)
            Ap = (b - t) * (h - t)
            p = 2 * ((b - t) + (h - t))
            self.It = 4 * Ap**2 * t / p + p * t**3 / 3

        except Exception as e:
            raise ValueError(f"Error attempting to calculate section properties: {str(e)}")

    def generate_polydata(self) -> pv.PolyData:
        # Define vertices of the section
        pontos = np.array([
            # Outer rectangle
            [-self.b/2, -self.h/2, 0.0],
            [ self.b/2, -self.h/2, 0.0],
            [ self.b/2,  self.h/2, 0.0],
            [-self.b/2,  self.h/2, 0.0],

            # Inner rectangle
            [-self.b/2 + self.t, -self.h/2 + self.t, 0.0],
            [ self.b/2 - self.t, -self.h/2 + self.t, 0.0],
            [ self.b/2 - self.t,  self.h/2 - self.t, 0.0],
            [-self.b/2 + self.t,  self.h/2 - self.t, 0.0]
        ], dtype=np.float64)

        # Region between contours
        triangles = np.array([
            # Triângulos conectando externo e interno
            [0, 4, 5], [0, 5, 1],  # Lado inferior
            [1, 5, 6], [1, 6, 2],  # Lado direito
            [2, 6, 7], [2, 7, 3],  # Lado superior
            [3, 7, 4], [3, 4, 0]   # Lado esquerdo
        ], dtype=int)

        # Define section face
        faces = np.hstack([[3, *t] for t in triangles])

        # Return PolyData object
        return pv.PolyData(pontos, faces=faces)

class Circular(SectionProperties):
    """
    Represents a solid circular cross-section.

    Attributes:
        r (float): Radius of the section.
        geometry_type (str): Identifier for the section shape ('circular').
        name (str): Label for the section.
    """
    def __init__(self, material: Material, radius: float):
        self.r = radius
        self.geometry_type = 'circular'
        self.name = f"Circular {self.r * 100:.2f} [cm]"

        # Initialize parent class
        super().__init__(material)

    def compute_properties(self):
        try:
            # Base parameters
            r = self.r

            # Sectional area and shear factor
            self.A = np.pi * r**2
            self.k = 0.89

            # Second moment of area
            self.Iy = (np.pi * r**4) / 4
            self.Iz = self.Iy
            self.Ip = self.Iy + self.Iz

            # Radii of gyration
            self.ry = np.sqrt(self.Iy / self.A)
            self.rz = np.sqrt(self.Iz / self.A)
            self.rp = np.sqrt(self.ry**2 + self.rz**2)

            # Torsional constant
            self.It = self.Iy + self.Iz

        except Exception as e:
            raise ValueError(f"Error attempting to calculate section properties: {str(e)}")

    def generate_polydata(self) -> pv.PolyData:
        # Return PolyData object
        return pv.Circle(radius=self.r, resolution=50)

class CHS(SectionProperties):
    """
    Represents a circular hollow cross-section.

    Attributes:
        ro (float): Outer radius of the section.
        ri (float): Inner radius of the section.
        geometry_type (str): Identifier for the section shape ('chs').
        name (str): Label for the section.
    """
    def __init__(self, material: Material, outer_radius: float, thickness: float):
        self.ro = outer_radius
        self.ri = outer_radius - thickness
        self.t = thickness
        self.geometry_type = 'chs'
        self.name = f"CHS {self.ro * 100:.2f} / {self.t * 100:.2f} [cm]"

        # Initialize parent class
        super().__init__(material)

    def compute_properties(self):
        try:
            # Base parameters
            ro, ri = self.ro, self.ri

            # Sectional area and shear factor
            self.A = np.pi * (ro**2 - ri**2)
            self.k = 0.53

            # Second moment of area
            self.Iy = (np.pi / 4) * (ro**4 - ri**4)
            self.Iz = self.Iy
            self.Ip = self.Iy + self.Iz

            # Radii of gyration
            self.ry = np.sqrt(self.Iy / self.A)
            self.rz = np.sqrt(self.Iz / self.A)
            self.rp = np.sqrt(self.ry**2 + self.rz**2)

            # Torsional constant
            self.It = self.Iy + self.Iz

        except Exception as e:
            raise ValueError(f"Error attempting to calculate section properties: {str(e)}")

    def generate_polydata(self) -> pv.PolyData:
        # Return PolyData object
        return pv.Disc(inner=self.ri, outer=self.ro, r_res=25, c_res=25)

class IProfile(SectionProperties):
    """
    Represents an I-profile cross-section.

    Attributes:
        b (float): Width of the section.
        h (float): Height of the section.
        tf (float): Flange thickness.
        tw (float): Web thickness.
        geometry_type (str): Identifier for the section shape ('rectangular').
        name (str): Label for the section.
    """
    def __init__(self, material: Material, width: float, height: float, tf: float, tw: float):
        self.b = width
        self.h = height
        self.tf = tf
        self.tw = tw
        self.geometry_type = 'I'
        self.name = f"I {self.b * 100} x {self.h * 100} x {self.tf * 100} x {self.tw * 100} [cm]"

        # Initialize parent class
        super().__init__(material)

    def compute_properties(self):
        try:
            # Base parameters
            b, h, tf, tw = self.b, self.h, self.tf, self.tw

            # Sectional area and shear factor
            self.A = 2 * b * tf + (h - 2 * tf) * tw
            self.k = 0.44

            # Second moment of area
            self.Iy = (b * h**3 - (b - tw) * (h - 2*tf)**3) / 12
            self.Iz = (2 * tf * b**3 / 12) + ((h - 2 * tf) * tw**3 / 12)
            self.Ip = self.Iy + self.Iz

            # Radii of gyration
            self.ry = np.sqrt(self.Iy / self.A)
            self.rz = np.sqrt(self.Iz / self.A)
            self.rp = np.sqrt(self.ry**2 + self.rz**2)

            # Torsional constant
            alpha = -0.042 + 0.2204 * (tw / tf) - 0.0725 * (tw / tf)**2
            D = (tf**2 + tw**2 / 4) / tf
            self.It = (2 * b * tf**3 + (h - 2 * tf) * tw**3) / 3 + 2 * alpha * D**4 - 4 * 0.105 * tf**4

        except Exception as e:
            raise ValueError(f"Error attempting to calculate section properties: {str(e)}")

    def generate_polydata(self) -> pv.PolyData:
        # Define vertices of the section
        vertices = np.array([
            [-self.b/2,  -self.h/2, 0],
            [ self.b/2,  -self.h/2, 0],
            [ self.b/2,  -self.h/2 + self.tf, 0],
            [ self.tw/2, -self.h/2 + self.tf, 0],
            [ self.tw/2,  self.h/2 - self.tf, 0],
            [ self.b/2,   self.h/2 - self.tf, 0],
            [ self.b/2,   self.h/2, 0],
            [-self.b/2,   self.h/2, 0],
            [-self.b / 2, self.h / 2 - self.tf, 0],
            [-self.tw / 2, self.h / 2 - self.tf, 0],
            [-self.tw / 2, -self.h / 2 + self.tf, 0],
            [-self.b/2,  -self.h/2 + self.tf, 0],
        ])

        # Define contour of the section
        contour = vertices[range(12)]

        # Define faces based on the number of vertices
        num_pontos = len(contour)
        faces = np.hstack([[num_pontos], np.arange(num_pontos)])

        # Return PolyData object
        return pv.PolyData(contour, faces=faces).triangulate()  # type: ignore[return-value]

class TProfile(SectionProperties):
    """
    Represents a T-profile cross-section.

    Attributes:
        b (float): Width of the section.
        h (float): Height of the section.
        tf (float): Flange thickness.
        tw (float): Web thickness.
        geometry_type (str): Identifier for the section shape ('T').
        name (str): Label for the section.
    """
    def __init__(self, material: Material, width: float, height: float, tf: float, tw:float):
        self.b = width
        self.h = height
        self.tf = tf
        self.tw = tw
        self.geometry_type = 'T'
        self.name = f"T {self.b * 100} x {self.h * 100} x {self.tf * 100} x {self.tw * 100} [cm]"

        # Initialize parent class
        super().__init__(material)

    def compute_properties(self):
        try:
            # Base parameters
            b, h, tf, tw = self.b, self.h, self.tf, self.tw

            # Sectional area and shear factor
            self.A = tf * (h - tw) + tw * b
            self.k = 0.44

            # Centroid calculation
            numerator = h ** 2 * tw + tf ** 2 * (b - tw)
            denominator = 2 * (b * tf + (h - tf) * tw)
            yc = h - (numerator / denominator)

            # Second moment of area
            self.Iy = (tw * yc**3 + b * (h - yc)**3 - (b - tw) * (h - yc - tf)**3) / 3
            self.Iz = ((h - tf) * tw**3 + b**3 * tf) / 12
            self.Ip = self.Iy + self.Iz

            # Radii of gyration
            self.ry = np.sqrt(self.Iy / self.A)
            self.rz = np.sqrt(self.Iz / self.A)
            self.rp = np.sqrt(self.ry**2 + self.rz**2)

            # Torsional constant
            self.It = (b * tf**3 + (h - tf/2) * tw**3) / 3

        except Exception as e:
            raise ValueError(f"Error attempting to calculate section properties: {str(e)}")

    def generate_polydata(self) -> pv.PolyData:
        # Define vertices of the section
        pontos = np.array([
            # Flanges
            [-self.b/2, self.h/2, 0],
            [ self.b/2, self.h/2, 0],
            [ self.b/2, self.h/2 - self.tf, 0],
            [-self.b/2, self.h/2 - self.tf, 0],

            # Web
            [-self.tw/2, -self.h/2, 0],
            [ self.tw/2, -self.h/2, 0],
            [ self.tw/2,  self.h/2 - self.tf, 0],
            [-self.tw/2,  self.h/2 - self.tf, 0],
        ])

        # Define contour
        contorno = pontos[[4, 5, 6, 2, 1, 0, 3, 7]]

        # Define faces based on the number of vertices
        num_pontos = len(contorno)
        faces = np.hstack([[num_pontos], np.arange(num_pontos)])

        # Return PolyData object
        return pv.PolyData(contorno, faces=faces).triangulate()  # type: ignore[return-value]

class Structure:
    """
    Main class representing the structural element_type.

    Manages nodes, members, finite elements, material properties,
    and boundary conditions. Acts as the central orchestrator for
    the analysis.

    Attributes:
        element_type (str): Type of analysis element_type ('beam' or 'truss').
        analysis_type (str): Linear, Nonlinear, Buckling.
        num_nodes (int): Total number of nodes in the generated mesh.
        dofs_per_node (int): Degrees of freedom per node (default: 6).
        nodes (Dict[int, Node]): Dictionary of final mesh nodes.
        original_members (Dict[int, dict]): Storage for original geometric members
                                            before discretization.
    """
    # Type hinting for class attributes
    analysis: str
    yaml_path: str
    num_nodes: int
    element_type: str
    is_buckling: bool
    num_elements: int
    subdivisions: int
    dofs_per_node: int
    condensation_data: list

    def __init__(
        self,
        metadata: Any,
        element_type: str,
        coordinates: npt.NDArray[np.float64],
        connectivity: npt.NDArray[np.integer],
        section_data: Dict[Union[int, slice, range], Dict[str, Any]],
        subdivisions: int,
        supports: List[Dict[str, Any]],
        releases: Dict[int, List[int]],
        nodal_loads: List[Dict[str, List[float]]],
        distributed_loads: List[Dict[str, List[float]]],
    ):
        """
        Initialize the structural model.

        Args:
            metadata (Any): Arbitrary metadata passed from the UI or input file.
            element_type (str): Element identifier (e.g., 'B33' and 'B32' for beams, 'T3D' and 'T2D' for trusses).
                                'B33' for Euler-Bernoulli beams (cubic formulation), 'B32' for Timoshenko beams (quadratic formulation).
                                'T3D' for 3D trusses.
            coordinates (np.ndarray): Array of nodal coord (N, 3).
            connectivity (np.ndarray): Array of member connectivity (M, 2).
            section_data (dict): Mapping of member indices to section properties.
            subdivisions (int): Number of finite elements per structural member.
            supports (dict): Mapping {node_id: [fixed_dofs]}.
            releases (dict): Mapping {member_id: [released_dofs]}.
            nodal_loads (dict): Mapping {node_id: [Fx, Fy, Fz, Mx, My, Mz]}.
            distributed_loads (dict): Mapping {member_id: [qx, qy, qz]}.
        """
        # Store initial data
        self.metadata = metadata
        self.element_type = element_type
        self.subdivisions = subdivisions
        self.coordenadas = np.array(coordinates, dtype=np.float64)

        # Check beam properties
        self.is_quadratic = (self.element_type == 'B32')
        self.dofs_per_node = 6
        self.nodes_per_element = 3 if self.is_quadratic else 2
        self.dofs_per_element = self.dofs_per_node * self.nodes_per_element

        # Create initial nodes
        self.original_nodes: Dict[int, Node] = {
            i: Node(node_id=i, coord=coord)
            for i, coord in enumerate(coordinates)
        }

        # Create new coords, if necessary
        if self.is_quadratic:
            # New coords for Timoshenko beams
            start_coords = coordinates[connectivity[:, 0]]
            end_coords = coordinates[connectivity[:, 1]]

            # Mid coords
            mid_coords = (start_coords + end_coords) / 2

            # Generate IDs for the new nodes
            num_originals_nodes = coordinates.shape[0]
            num_elements = connectivity.shape[0]
            mid_coords_ids = np.arange(num_originals_nodes, num_originals_nodes + num_elements)

            # Update coordinates and coonectivity
            coordinates = np.vstack((coordinates, mid_coords))
            connectivity = np.column_stack([
                connectivity[:, 0],
                mid_coords_ids,
                connectivity[:, 1]
            ])

        # Create current nodes
        self.nodes: Dict[int, Node] = {
            i: Node(node_id=i, coord=coord)
            for i, coord in enumerate(coordinates)
        }

        # Define original members (before generating mesh)
        self.original_members: Dict[int, Dict[str, Any]] = {}
        if self.is_quadratic:
            for i, (start_idx, mid_idx, end_idx) in enumerate(connectivity):
                self.original_members[i] = {
                    'nodes': (self.nodes[start_idx], self.nodes[mid_idx], self.nodes[end_idx]),
                    'distributed_load': None,    # To be filled by add_distributed_loads (if any)
                    'section': None,             # To be filled by assign_sections
                    'hinges': None               # To be filled by define_releases
            }
        else:
            for i, (start_idx, end_idx) in enumerate(connectivity):
                self.original_members[i] = {
                    'nodes': (self.nodes[start_idx], self.nodes[end_idx]),
                    'distributed_load': None,    # To be filled by add_distributed_loads (if any)
                    'section': None,             # To be filled by assign_sections
                    'hinges': None               # To be filled by define_releases
                }

        # Containers for final elements
        self.elements: Dict[int, Element] = {}
        self.num_elements: int = len(self.original_members)

        # Dictionaries to store the IDs to supports, loads and member_releases
        self.nodal_loads = {}
        self.member_releases = {}

        # Associar as propriedades aos membros originais
        self.assign_sections(section_data)
        self.define_supports(supports)
        self.define_releases(releases)

        # Gerar a malha final de elements
        self.generate_mesh(self.subdivisions)

        # Store condensation data
        self.condensation_data = []
        self.store_condensation_data(releases)

        # Adicionar as cargas nodais e distribuídas
        self.add_nodal_loads(nodal_loads)
        self.add_distributed_loads(distributed_loads, self.subdivisions)

    def add_node(self, node_id: int, coord: Sequence[float]) -> None:
        """
        Adds a single node to the structure.

        Args:
            node_id (int): Unique identifier for the node.
            coord (list/array): (x, y, z) coord.
        """
        self.nodes[node_id] = Node(node_id, coord)

    def assign_sections(self, section_map: Dict[Union[int, slice, range], Dict[str, Any]]) -> None:
        """
        Assigns cross-section properties to structural members.

        Optimized to avoid creating duplicate Section objects for identical definitions.

        Args:
            section_map (dict): Keys are member indices (int, slice, or range),
                                Values are property dictionaries.
        """
        # Initial data
        num_members = len(self.original_members)

        # Cache to store unique Section objects
        unique_sections_cache: Dict[Tuple, SectionProperties] = {}

        for indices_key, properties_dict in section_map.items():
            # Create a hashable key from the dictionary items to check uniqueness
            cache_key = tuple(sorted(properties_dict.items(), key=lambda item: str(item[0])))

            if cache_key not in unique_sections_cache:
                unique_sections_cache[cache_key] = self.create_section(properties_dict)

            # Retrieve cached instance
            section_instance = unique_sections_cache[cache_key]

            # Resolve indices (handle single int, slices, or lists)
            target_indices = []
            if isinstance(indices_key, int):
                target_indices = [indices_key]
            elif isinstance(indices_key, (slice, range, list, tuple)):
                if isinstance(indices_key, slice):
                    target_indices = list(range(*indices_key.indices(num_members)))
                else:
                    target_indices = list(indices_key)

            # Assign the section object to the target members
            for idx in target_indices:
                if idx in self.original_members:
                    self.original_members[idx]['section'] = section_instance

    @staticmethod
    def create_section(section_data: Dict[str, Any]) -> SectionProperties:
        """
        Method to instantiate the correct Section class.

        Args:
            section_data (dict): Must contain 'geometry' key (e.g., 'rectangular')
                                 and material properties ('E', 'nu').

        Returns:
            SectionProperties: Instantiated section object.

        Raises:
            ValueError: If geometry is missing or unknown.
        """
        # Create a copy of original dictionary
        data = section_data.copy()

        # Extract geometry type
        geometry = data.pop("geometry", '')
        if not geometry:
            raise ValueError("Section definition is missing the 'geometry' key.")

        # Extract constitutive properties
        elastic_modulus_val = data.pop("E", None)
        nu_val = data.pop("nu", 0.0)

        if elastic_modulus_val is None:
            raise ValueError(f"Material Young's Modulus (E) missing for section '{geometry}'.")

        # Create material
        material = Material(elastic_modulus=elastic_modulus_val, poisson_ratio=nu_val)

        # Map the geometry for the correct class
        geometry_class_map = {
            "rectangular": Rectangular,
            "rhs": RHS,
            "circular": Circular,
            "chs": CHS,
            "I": IProfile,
            "T": TProfile,
        }

        # Get the section class
        section_class = geometry_class_map.get(geometry)
        if section_class is None:
            valid_keys = list(geometry_class_map.keys())
            raise ValueError(f"Unknown section geometry '{geometry}'. Valid types: {valid_keys}")

        try:
            # Instantiate the section class
            return section_class(material=material, **data)
        except TypeError as e:
            raise ValueError(f"Failed to create '{geometry}' section. Check parameters. Details: {e}")

    def generate_mesh(self, mesh_param: int | float, decimals: int = 8):
        """
        Generates the finite element mesh by subdividing original structural members.

        If subdivisions > 1, splits each original member into multiple finite elements,
        creating intermediate nodes as necessary. Preserves connectivity and properties.

        Args:
            mesh_param (int | float): Number of finite elements per structural member or element length.
            decimals (int): Precision for node coordinate matching (merging).
        """
        # coord to index mapping
        coord_map = {tuple(np.round(n.coord, decimals)): i for i, n in self.nodes.items()}
        next_node_id = len(self.nodes)
        current_element_id = 0

        # Iterate over original members
        for member in self.original_members.values():
            # Get star and end nodes and hinges for the member
            start_node, end_node = member['nodes'][0], member['nodes'][-1]

            # Get the number of subdivisions based on the type of mesh parameter
            if isinstance(mesh_param, float):
                # Get the element length
                L = np.linalg.norm(end_node.coord - start_node.coord)

                # Get the number of subdivisions for this member
                num_subdivisions = int(np.ceil(L / mesh_param))
            elif isinstance(mesh_param, int):
                num_subdivisions = mesh_param

            # Get section and releases (hinges)
            section = member.get('section')
            member_releases = member.get('hinges', [[], []])
            start_hinge, end_hinge = member_releases

            # Generate intermediate points along the member
            num_points = (2 * num_subdivisions) + 1 if self.is_quadratic else (num_subdivisions + 1)
            segment_points = np.linspace(start_node.coord, end_node.coord, num=num_points)
            segment_node_indices = []

            # Create new intermediate nodes
            for point in segment_points:
                point_key = tuple(np.round(point, decimals))

                if point_key in coord_map:
                    # Reuse existing node
                    segment_node_indices.append(coord_map[point_key])
                else:
                    # Create new intermediate node
                    self.add_node(next_node_id, point)
                    coord_map[point_key] = next_node_id
                    segment_node_indices.append(next_node_id)
                    next_node_id += 1

            # Create finite elements connecting the nodes
            for k in range(num_subdivisions):
                # Determine connectivity based on element type
                if self.is_quadratic:
                    # Get first, middle and last nodes of the sub-element
                    n1_idx = 2 * k
                    n2_idx = 2 * k + 1
                    n3_idx = 2 * k + 2

                    n1 = segment_node_indices[n1_idx]
                    n2 = segment_node_indices[n2_idx]
                    n3 = segment_node_indices[n3_idx]

                    # Connectivity for quadratic elements
                    element_connectivity = (self.nodes[n1], self.nodes[n2], self.nodes[n3])

                else:
                    # Get first and last nodes of the sub-element
                    n1 = segment_node_indices[k]
                    n2 = segment_node_indices[k + 1]

                    # Connectivity for cubic elements
                    element_connectivity = (self.nodes[n1], self.nodes[n2])

                # Assign releases only to the physical ends of the member chain
                elem_start_hinge = start_hinge if k == 0 else []
                elem_end_hinge = end_hinge if k == (num_subdivisions - 1) else []

                # Combine hinges for this element
                current_elem_release = [elem_start_hinge, elem_end_hinge]

                # Add the new Element to the structure
                self.elements[current_element_id] = Element(
                    element_id=current_element_id,
                    conec=element_connectivity,
                    secao=section,
                    hinge=current_elem_release
                )
                current_element_id += 1

        # Update Structure state
        self.num_elements = len(self.elements)
        self.num_nodes = len(self.nodes)

    def store_condensation_data(self, releases_map: Dict[int, List[int]]) -> None:
        # Verify if releases_map is empty
        if not releases_map:
            return

        # Initial data
        dofs_per_element = self.dofs_per_element
        all_dofs = np.arange(dofs_per_element)

        # Initialize condensed elements list
        for elem_id, element in self.elements.items():
            # Start and end releases
            start_releases = element.hinges[0]
            end_releases = element.hinges[-1]

            # Skip if no hinges
            if not start_releases and not end_releases:
                continue

            # Calculate start degrees of freedom indices
            start_idx = start_releases

            # Calculate end degrees of freedom indices
            offset = (self.nodes_per_element - 1) * self.dofs_per_node
            end_idx = [dof + offset for dof in end_releases]

            # Store the condensed indices for the element
            condensed_indices = list(start_idx) + end_idx

            # Iterate over condened elements
            elim = np.array(condensed_indices, dtype=int)
            kept = np.setdiff1d(all_dofs, elim)

            # Store the condensed data
            self.condensation_data.append({
                'id': elem_id,
                'elim_indices': elim,
                'kept_indices': kept,
            })

    def define_releases(self, releases_map: Dict[int, List[int]]) -> None:
        """
        Assigns internal degree of freedom releases (hinges) to members.

        Args:
            releases_map (dict): Mapping {member_id: [[start_dofs], [end_dofs]]}.
                                 Example: {0: [[5], [5]]} releases Mz at both ends.
        """
        # Verify if releases_map is empty
        if not releases_map:
            return

        # Iterate over releases_map and add hinges accordingly
        for member_id, hinge in releases_map.items():
            self.original_members[member_id]['hinges'] = hinge

    def define_supports(self, supports_list: List[Dict[str, Any]]) -> None:
        """
        Applies boundary conditions (supports) to nodes.

        Args:
            supports_list (list): List of dicts, each containing:
                                    - 'node_id': int or list of ints
                                    - 'boundary_conditions': list of fixed DOFs (e.g. [0, 1, 2])
        """
        # Iterate over supports_list
        for support in supports_list:
            node_ids = support.get('node_id')
            constraints = support.get('boundary_conditions', [])

            # Normalize to list of IDs
            if isinstance(node_ids, int):
                node_ids_list = [node_ids]
            elif isinstance(node_ids, (list, tuple)):
                node_ids_list = node_ids
            else:
                continue

            # Validate DOF indices
            valid_constraints = [dof for dof in constraints if 0 <= dof <= self.dofs_per_node]

            # Apply to Node objects
            for no_idx in node_ids_list:
                self.nodes[no_idx].boundary_conditions = valid_constraints

    def add_nodal_loads(self, loads_data: List[Dict[str, List[float]]]) -> None:
        """
        Applies point loads to nodes.

        Args:
            loads_data (list): List of dicts containing:
                               - 'node_id': int or list of ints
                               - 'load': list of 6 values [Fx, Fy, Fz, Mx, My, Mz]
        """
        if not loads_data:
            return

        # Iterate over the load configuration
        for entry in loads_data:
            # Node IDs and loads
            node_id = entry.get('node_id', [])
            load = entry.get('load')

            # Skip if no values found
            if load is None:
                continue

            # Normalize list of node_ids
            node_list: List[int] = []

            if isinstance(node_id, int):
                node_list = [node_id]
            elif isinstance(node_id, list):
                node_list = [n for n in node_id if isinstance(n, int)]
            else:
                continue

            # Convert loads to numpy array
            load = np.array(load, dtype=float)

            # Apply load to specific nodes accordingly
            for node_id in node_list:
                if node_id in self.nodes:
                    self.nodal_loads[node_id] = np.array(load, dtype=float)

    def add_distributed_loads(self, loads_data: List[Dict[str, Any]], num_subdivisions: int):
        """
        Applies distributed loads to elements, handling mesh subdivision.

        Args:
            loads_data (list): List of dicts containing:
                                - 'element_id': int or list of member IDs.
                                - 'load': Tuple of (start_vector, end_vector).
                                          Each vector is [qx, qy, qz].
            num_subdivisions (int): The subdivision factor used in mesh generation.
        """
        if not loads_data:
            return

        # Iterate over the load configuration
        for entry in loads_data:
            # Element ID and loads
            elem_ids = entry.get('element_id', [])
            load = entry.get('load')

            if not load or len(load) != 2:
                continue

            # Extract start and end load vectors
            q_start = np.array(load[0], dtype=float)
            q_end = np.array(load[1], dtype=float)

            # Normalize element ids
            element_ids = [elem_ids] if isinstance(elem_ids, int) else list(elem_ids)

            for elem_id in element_ids:
                # Apply distributed load to the original member
                if elem_id in self.original_members:
                    self.original_members[elem_id]['distributed_load'] = (q_start, q_end)

                if num_subdivisions > 1 or self.is_quadratic:
                    # Determine number of points to interpolate
                    num_points = (2 * num_subdivisions + 1) if self.is_quadratic else (num_subdivisions + 1)

                    # Interpolation factors
                    xi = np.linspace(0, 1, num_points).reshape(-1, 1)

                    # Linear interpolation of loads
                    q_interp = (1 - xi) * q_start + xi * q_end

                    # Base element ID
                    base_id = elem_id * num_subdivisions

                    for i in range(num_subdivisions):
                        if self.is_quadratic:
                            # Get first, middle and last nodes of the sub-element
                            n1_idx = 2 * i
                            n2_idx = 2 * i + 1
                            n3_idx = 2 * i + 2

                            q_vals = [q_interp[n1_idx], q_interp[n2_idx], q_interp[n3_idx]]
                        else:
                            # Get first and last nodes of the sub-element
                            n1_idx = i
                            n2_idx = i + 1

                            q_vals = [q_interp[n1_idx], q_interp[n2_idx]]

                        # Global ID for the sub-element
                        id_sub = base_id + i

                        self.elements[id_sub].distributed_load += np.vstack(q_vals)
                else:
                    # No subdivisions
                    self.elements[elem_id].distributed_load += np.vstack([q_start, q_end])
