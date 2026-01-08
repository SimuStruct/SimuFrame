# Built-in libraries
from typing import Optional

# Third-party libraries
import numpy as np
import pyvista as pv
from tqdm import tqdm
import numpy.typing as npt
from joblib import Parallel, delayed

# Local libraries
from .support_visualization import plot_supports
from SimuFrame.core.model import Structure


def _create_double_arrow(radius=0.05, shaft_resolution=20):
    """
    Creates a custom double-arrow geometry to represent concentrated moments.
    Args:
        radius (float, optional): Radius of the shaft. Defaults to 0.05.
        shaft_resolution (int, optional): Resolution of the shaft. Defaults to 20.
    """
    # Create the shaft
    shaft = pv.Cylinder(
        radius=radius,
        height=0.7,
        center=(0.35, 0, 0),
        direction=(1, 0, 0),
        resolution=shaft_resolution
    )

    # Create the first tip (at the end)
    tip1 = pv.Cone(
        radius=2.5 * radius,
        height=0.3,
        center=(0.85, 0, 0),
        direction=(1, 0, 0),
        resolution=shaft_resolution
    )

    # Create the second tip (behind the first one)
    tip2 = pv.Cone(
        radius=2.5 * radius,
        height=0.3,
        center=(0.65, 0, 0),
        direction=(1, 0, 0),
        resolution=shaft_resolution
    )

    # Merge geometries
    arrow = shaft + tip1 + tip2

    return arrow


def plot_nodal_loads(plotter, structure, scale_factor=1.0):
    """
    Plots nodal loads (concentrated forces and moments) on the structure.
    Forces are rendered as single arrows, while moments are rendered as double arrows.

    Args:
        plotter (pyvista.QtInteractor): Plotter instance.
        structure (Structure): Instance of the Structure class.
        scale_factor (float): Global scale factor for the arrows.
    """
    # Containers for meshes
    force_data = {'points': [], 'dirs': []}
    moment_data = {'points': [], 'dirs': []}

    # Base vector for global axes
    base_vec = np.eye(3)

    # Containers for labels
    force_labels = {'pos': [], 'text': []}
    moment_labels = {'pos': [], 'text': []}

    # Visual offset for meshes and labels
    mesh_offset = 0.01 * scale_factor
    text_offset = 1.10 * scale_factor

    # Iterate through loaded nodes, {node_id: [Fx, Fy, Fz, Mx, My, Mz]}
    for node_id, loads in structure.nodal_loads.items():
        node = structure.nodes[node_id]

        for dof in range(6):
            value = loads[dof]

            # Skip negligible loads
            if np.abs(value) < 1e-9:
                continue

            # Get direction (x, y or z) and orientation (+ or -) of the load
            axis = dof % 3
            direction = base_vec[axis] * np.sign(value)

            # Offset positions
            coord = node.coord + mesh_offset * direction
            text_coord = node.coord + text_offset * direction

            # Separate forces from moments
            if dof < 3:
                force_data['points'].append(coord)
                force_data['dirs'].append(direction)

                force_labels['pos'].append(text_coord)
                force_labels['text'].append(f'{value:.2f} kN')
            else:
                moment_data['points'].append(node.coord)
                moment_data['dirs'].append(direction)

                moment_labels['pos'].append(text_coord)
                moment_labels['text'].append(f'{value:.2f} kNm')

        def _render_arrows(data, name, geom, color):
            """
            Internal function to add a glyph to the plotter.
            Args:
                data (dict): Dictionary containing points, directions, scalars and scales.
                name (str): Name of the glyph (forces or moments).
                geom (pv.PolyData): Geometry source for the glyphs (single or double arrows).
                color (str): Color name for the different arrows.
            """
            if not data['points']:
                return

            # Convert data to numpy arrays
            points = np.array(data['points'])
            vectors = np.array(data['dirs'])

            # Create PolyData from points
            polydata = pv.PolyData(points)
            polydata['vectors'] = vectors

            # Instance the arrow geometry at specified points
            glyphs = polydata.glyph(
                orient='vectors',
                scale=False,
                geom=geom,
                factor=scale_factor,
                tolerance=0.0
            )

            # Add glyphs to the plotter
            plotter.add_mesh(glyphs, color=color, name=name)

        def _render_labels(data, name, text_color):
            """
            Internal function to add labels to the plotter.

            Args:
                data (PolyData): PolyData object containing the labels.
                name (str): Name of the labels.
                text_color (str): Color name for the labels. Defaults to 1.0.
            """
            if not data['pos']:
                return

            # Add text labels
            plotter.add_point_labels(
                np.array(data['pos']),
                data['text'],
                name=name,
                font_size=16,
                text_color=text_color,
                shape_opacity=0.0,
                show_points=False,
                always_visible=True
            )

        # Render forces (crimson red, single arrow)
        _render_arrows(
            force_data,
            name='Forces (kN)',
            geom=pv.Arrow(tip_length=0.25, shaft_radius=0.05),
            color='crimson',
        )
        _render_labels(force_labels, name='Forces (kN)', text_color='crimson')

        # Plot moments (double arrow)
        _render_arrows(
            moment_data,
            name='Moments (kNm)',
            geom=_create_double_arrow(),
            color='royalblue',
        )
        _render_labels(moment_labels, name='Moments (kNm)', text_color='royalblue')


def plot_distributed_loads(plotter,
                           structure: Structure,
                           scale_factor=1.0,
                           num_arrows=5
                           ) -> None:
    """
    Plot distributed loads on the structure (uniform or trapezoidal).

    Args:
        plotter (pyvista.QtInteractor): Plotter instance.
        structure (Structure): Instance of the Structure class.
        scale_factor (float): Global visual scale for the arrows.
        num_arrows (int): Number of arrows to draw for each element.
    """
    # Containers for rendering
    force_data = {'points': [], 'dirs': []}
    force_labels = {'pos': [], 'text': []}

    # Visual configuration
    base_vec = np.eye(3)
    mesh_offset = 0.01 * scale_factor
    text_offset = 1.10 * scale_factor

    # Iterate through members
    for element in structure.original_members.values():
        load = element.get('distributed_load')

        # Skip if no load found
        if load is None:
            continue

        # Get nodes and coordinates
        n1, n2 = element['nodes'][0], element['nodes'][-1]
        pt1 = n1.coord
        pt2 = n2.coord

        # Calculate vector
        vector = pt2 - pt1

        # Iterate over axes (x, y, z)
        for axis in range(3):
            q_start = load[0][axis]
            q_end = load[-1][axis]

            # Skip if no load found
            if abs(q_start) < 1e-9 and abs(q_end) < 1e-9:
                continue

            # Create normalized segments
            segments = np.linspace(0, 1, num_arrows)

            # Interpolate load magnitudes: q(t) = q_start + (q_end - q_start) * t
            q_values = q_start + (q_end - q_start) * segments

            # Interpolate positions similarly to q_values
            points = pt1 + np.outer(segments, vector)

            # Calculate direction for each arrow
            directions = np.outer(np.sign(q_values), base_vec[axis])

            # Apply offset to prevent overlapping arrows
            points_offset = points + (directions * mesh_offset)

            # Store force data
            force_data['points'].append(points_offset)
            force_data['dirs'].append(directions)

            def _add_label_at(q_val, point_base, direction):
                """
                Internal function to add a label at a specific point.
                """
                if abs(q_val) < 1e-9:
                    return

                # Calculate text position
                text_coord = point_base + (direction * text_offset)

                force_labels['pos'].append(text_coord)
                force_labels['text'].append(f'{q_val:.2f} kN/m')

            # Label starting point
            _add_label_at(q_start, points_offset[0], directions[0])

            # Label ending point
            _add_label_at(q_end, points_offset[-1], directions[-1])

    if not force_data['points']:
        return

    # Flatten the lists of arrays
    all_points = np.vstack(force_data['points'])
    all_dirs = np.vstack(force_data['dirs'])

    # Create PolyData from points
    polydata = pv.PolyData(all_points)
    polydata['vectors'] = all_dirs

    # Create glyphs
    glyphs = polydata.glyph(
        orient='vectors',
        scale=False,
        geom=pv.Arrow(tip_length=0.25, shaft_radius=0.05),
        factor=scale_factor,
        tolerance=0.0
    )

    # Plot glyphs
    plotter.add_mesh(glyphs, name='Distributed Loads (kN/m)', color='mediumseagreen', show_scalar_bar=False)

    # Add labels
    if force_labels['pos']:
        plotter.add_point_labels(
            np.array(force_labels['pos']),
            force_labels['text'],
            name='Distributed Loads (kN/m)',
            font_size=16,
            text_color='mediumseagreen',
            shape_opacity=0.0,
            show_points=False,
            always_visible=True
        )

def plot_structure(plotter,
                   structure: Structure,
                   mesh: pv.MultiBlock,
                   opacity: float = 1.0,
                   plot_section: bool = True,
                   plot_loads: bool = True,
                   plot_nodes: bool = True,
                   name: Optional[str] = None,
                   arrow_scale_factor: float = 0.50,
                   node_scale_factor: float = 0.05,
                   support_scale_factor: float = 0.50,
                   min_font_size: int = 12,
                   max_font_size: int = 24):
    """
    Plots both undeformed and deformed structures.

    Args:
        plotter (QtInteractor): PyVista Plotter object.
        structure (Structure): Instance of the Structure class.
        mesh (dict): Dictionary containing structure meshes.
        opacity (float): Opacity of the cross-sections.
        plot_section (bool): Flag to plot cross-sections.
        plot_loads (bool): Flag to plot loads.
        plot_nodes (bool): Flag to plot nodes.
        name (str): Mesh name identifier.
        arrow_scale_factor (float): Scale multiplier for arrows.
        node_scale_factor (float): Scale multiplier for nodes.
        support_scale_factor (float): Scale multiplier for supports.
        min_font_size (int): Minimum font size for labels.
        max_font_size (int): Maximum font size for labels.
    """
    # Extract all coordinates (undeformed)
    coords = np.array([no.coord for no in structure.original_nodes.values()])

    # Calculate Bounding Box Diagonal
    if coords.size > 0:
        bbox_diag = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
    else:
        bbox_diag = 1.0

    # Avoid zero-size structures
    if bbox_diag < 1e-6:
        bbox_diag = 1.0

    # Calculate element lengths
    lengths = []
    for element in list(structure.original_members.values()):
        n1_coord = element['nodes'][0].coord
        n2_coord = element['nodes'][-1].coord
        lengths.append(np.linalg.norm(n1_coord - n2_coord))

    # Use median length
    median = np.percentile(lengths, 50) if lengths else 1.0

    # Define scale constraints based on bounding box
    base_value = 0.50 * median
    lower_limit = 0.025 * bbox_diag
    upper_limit = 0.25 * bbox_diag

    # Final computed scale
    scale = np.clip(base_value, lower_limit, upper_limit)

    # Derived scales
    arrow_scale = scale * arrow_scale_factor
    node_radius = scale * node_scale_factor
    support_scale = scale * support_scale_factor

    # Adicionar as malhas à estrutura
    # plotter.add_mesh(tubos.combine(), color='gray', opacity=transparencia, name=f'{name}_tubos')

    # Plot Cross-Sections
    if plot_section and 'section' in mesh:
        section_mesh = mesh['section']
        # Combine if it's a MultiBlock for better performance
        if isinstance(section_mesh, pv.MultiBlock):
            combined_mesh = section_mesh.combine()
        else:
            combined_mesh = section_mesh

        plotter.add_mesh(
            combined_mesh,
            color='lightblue',
            edge_color='gray',
            opacity=opacity,
            name=f'{name}_sections' if name else 'sections'
        )

    # Plot Nodes
    if plot_nodes:
        # Original nodes
        initial_coords = np.array([no.coord for no in structure.original_nodes.values()])

        # Create PolyData from coordinates
        pontos_nos = pv.PolyData(initial_coords)

        # Create sphere glyph
        sphere = pv.Sphere(radius=float(node_radius), theta_resolution=12, phi_resolution=12)

        # Apply glyph to points
        glyph = pontos_nos.glyph(scale=False, orient=False, geom=sphere)

        plotter.add_mesh(
            glyph,
            style='surface',
            color='yellow',
            name=f'{name}_glyphs' if name else 'glyphs'
        )

        # Add node labels
        plotter.add_point_labels(
            initial_coords,
            [str(i + 1) for i in structure.original_nodes.keys()],
            font_size=20,
            text_color='black',
            shape=None,
            always_visible=True,
            name=f'{name}_nodes' if name else 'node_labels'
        )

    # Plot Supports
    plot_supports(plotter, structure, support_scale)

    # Plot Loads
    if plot_loads:
        plot_nodal_loads(plotter, structure, scale_factor=float(arrow_scale))
        plot_distributed_loads(plotter, structure, scale_factor=float(arrow_scale))

    plotter.add_axes()


def add_scalars(grid: pv.MultiBlock,
                combined_grid: pv.UnstructuredGrid,
                start_values: npt.NDArray[np.float64] | None = None,
                end_values: npt.NDArray[np.float64] | None = None
                ) -> None:
    """
    Calculate and assign interpolated scalars to the grid.
    """
    if grid is None or grid.n_blocks == 0:
        return

    # Handle default values
    num_blocks = grid.n_blocks
    if start_values is None:
        start_values = np.zeros(num_blocks)
    if end_values is None:
        end_values = np.ones(num_blocks)

    # Extract cell counts
    n_cells_per_section = np.array([block.n_cells for block in grid], dtype=np.int32)

    # Generate ramps (check if all sections have the same number of cells)
    first_count = n_cells_per_section[0]
    is_uniform = np.all(n_cells_per_section == first_count)

    if is_uniform and first_count > 0:
        # Create one template ramp and repeat it
        template_ramp = np.linspace(0, 1, first_count)
        ramps = np.repeat(template_ramp, num_blocks)

        # Calculate repeated values
        repeated_starts = np.repeat(start_values, first_count)
        repeated_ends = np.repeat(end_values, first_count)

    else:
        # Variable subdivision, calculate ramps for each section
        ramps = np.concatenate([np.linspace(0, 1, n) for n in n_cells_per_section])

        # Standardize repeated values
        repeated_starts = np.repeat(start_values, n_cells_per_section)
        repeated_ends = np.repeat(end_values, n_cells_per_section)

    # Calculate scalars (linear interpolation)
    scalars = repeated_starts + (repeated_ends - repeated_starts) * ramps

    # Assign scalars to the grid
    combined_grid.cell_data['scalars'] = scalars

def map_unique_sections(structure: Structure) -> tuple[list[pv.PolyData], np.ndarray]:
    """
    Identifies unique sections across the structure to avoid redundant processing.

    Args:
        structure (Structure): Instance of the Structure class.

    Returns:
        tuple: A tuple containing:
            - secoes_unicas (list): Uma lista de objetos pv.PolyData.
            - secoes_indices (np.ndarray): Um array onde o índice 'i' armazena o
                                          índice da seção única para o elemento 'i'.
    """
    unique_sections = []
    section_indices = np.full(structure.num_elements, -1, dtype=int)

    # Dictionary to map a unique signature to an index
    signature_to_index = {}

    # Iterate over elements
    for elem_id, element in structure.elements.items():
        if elem_id >= len(section_indices):
            continue

        # Verify if the element has a section
        section = element.section
        if not section:
            continue

        # Create a signature based on properties
        signature = (type(section), tuple(section.__dict__.values()))

        if signature in signature_to_index:
            idx = signature_to_index[signature]
        else:
            # Generate the PolyData once per unique section
            polydata = section.generate_polydata()

            idx = len(unique_sections)
            signature_to_index[signature] = idx
            unique_sections.append(polydata)

        section_indices[elem_id] = idx

    return unique_sections, section_indices

def extrude_element_section(section,
                            start_coord,
                            extrude_vec,
                            rot_matrix
                            ) -> pv.PolyData | pv.MultiBlock | pv.DataObject | None:
    """
    Extrudes a section along an extrusion vector.
    """
    if section is None:
        return None

    # Rotate the section and translate to the start coordinate
    points_transformed = section.points @ rot_matrix.T + start_coord

    # Create new PolyData from the transformed points
    section_transformed = pv.PolyData(points_transformed, section.faces)

    # Extrude the section
    try:
        return section_transformed.extrude(extrude_vec, capping=True)
    except Exception:
        return None

def compute_rotation_matrices(
        coords: npt.NDArray[np.float64],
        ref_vectors: npt.NDArray[np.float64]):
    """
    Calculates all rotation matrices for each element in the mesh generation process.

    Args:
        structure (Structure): Instance of the Structure class.
        coords (np.ndarray): Nodal coordinates of the structure.
        ref_vectors (np.ndarray): Reference vectors for each element.

    Returns:
        np.ndarray: Array of rotation matrices for each element.
    """
    # Compute local x axis (x_)
    x_ = coords[:, -1] - coords[:, 0]
    norms = np.linalg.norm(x_, axis=1, keepdims=True)

    # Avoid division by zero and define unit vector
    norms[norms < 1e-10] = 1.0
    e1 = x_ / norms

    # Compute local y axis
    y_ = np.cross(ref_vectors, e1)
    y_norms = np.linalg.norm(y_, axis=1, keepdims=True)

    # Handles colinearity (singularity check)
    singular_indices = (y_norms < 1e-10)

    if np.any(singular_indices):
        # Change local y axis to either local x axis or local z axis
        fallback = np.zeros_like(e1[singular_indices])

        # Check if local x is close to global z
        is_vertical = np.abs(e1[singular_indices, 2]) > 0.9

        # Set fallback: [1., 0., 0.] for vertical, [0., 0., 1.] otherwise
        fallback[is_vertical] = np.array([1., 0., 0.])
        fallback[~is_vertical] = np.array([0., 0., 1.])

        # Recompute y for singular elements
        y_[singular_indices] = np.cross(fallback, e1[singular_indices])

    # Re-normalize e2 if necessary
    y_norms = np.linalg.norm(y_, axis=1, keepdims=True)
    e2 = y_ / y_norms

    # Compute local z axis (already normalized)
    e3 = np.cross(e1, e2)

    # Assemble rotation matrix
    rot_matrices = np.stack([e2, e3, e1], axis=-1)

    return rot_matrices

def process_element_geometry(args) -> tuple[pv.PolyData | pv.DataObject, pv.PolyData | pv.DataObject | None]:
    """
    Worker function for parallel processing.
    Generates the visualization geometry (tubes and extruded 3D sections).
    """
    # Get element data
    coords, base_section, rot_matrix = args
    num_points = coords.shape[0]

    # Generate the line connecting the nodes of the element
    if num_points == 3:
        # Create a spline with 3 points
        line = pv.Spline(coords, n_points=15)
    else:
        # Create a normal line with 2 points
        line = pv.Line(pointa=coords[0], pointb=coords[-1])

    # Create a tube from the line
    tube = line.tube(radius=0.01, n_sides=6)

    # Extrude the base section
    if base_section is None:
        return tube, None
    mesh = None

    if num_points == 2:
        # Extrude from initial coord to final coord
        L_vec = coords[-1] - coords[0]
        mesh = extrude_element_section(base_section, coords[0], L_vec, rot_matrix)
    else:
        # Segmented extrusion from spline
        blocks = pv.MultiBlock()

        # Iterate over the three points
        for i in range(num_points - 1):
            p_start = coords[i]
            p_end = coords[i + 1]
            p_vec = p_end - p_start

            # Extrude this segment
            seg_mesh = extrude_element_section(base_section, p_start, p_vec, rot_matrix)

            if seg_mesh:
                blocks.append(seg_mesh)

        # Merge the segments
        if len(blocks) > 0:
            mesh = blocks.combine()

    return tube, mesh

def generate_mesh(
        structure: Structure,
        sections: list,
        sections_idx: npt.NDArray[np.int64],
        coords: npt.NDArray[np.float64],
        ref_vector: dict[str, npt.NDArray[np.float64]],
        geometry_type: str = 'deformed',
        **kwargs
        ) -> dict[str, pv.MultiBlock | list[pv.MultiBlock]]:
    """
    Main function to generate the structural mesh (tubes and 3D solid sections).

    Args:
        estrutura (Structure): Instance of the Structure class.
        sections (list): List of unique sections.
        sections_idx (np.ndarray): Map of element index to section index.
        coords (np.ndarray): Array of element coordinates.
        ref_vector (dict): Reference vectors for orientation (deformed and undeformed).
        geometry_type (str): Structure geometry (deformed or undeformed).

    Returns:
        dict: {'tubes': MultiBlock, 'section': MultiBlock}            -
    """
    # Initial data
    num_elements = structure.num_elements if geometry_type == 'deformed' else len(structure.original_members)
    eigenvalues = kwargs.get('autovalores', np.array([]))
    num_modes = eigenvalues.shape[0] if eigenvalues.size > 0 else 0

    # Reference vector handling
    k_vec = ref_vector[geometry_type] if isinstance(ref_vector, dict) else ref_vector

    # Fix index alignment if mesh refinement occurred (undeformed mesh)
    if len(sections_idx) != num_elements:
        if len(sections_idx) > num_elements:
            ratio = len(sections_idx) / num_elements
            step = int(ratio) if ratio.is_integer() else 1
            sections_idx = sections_idx[::step]
        else:
            sections_idx = sections_idx[:num_elements]

    # Define jobs (parallel processing)
    jobs = []

    # Check for buckling analysis (multiple modes)
    if structure.is_buckling and geometry_type == 'deformed' and num_modes > 0:
        for mode in range(num_modes):
            rot_matrices = compute_rotation_matrices(coords[mode], k_vec)
            jobs.append({
                'coords': coords[mode],
                'rot_matrices': rot_matrices,
                'label': f"Buckling mode {mode + 1}"
            })
    else:
        # Standard static analysis
        rot_matrices = compute_rotation_matrices(coords, k_vec)
        jobs.append({
            'coords': coords,
            'rot_matrices': rot_matrices,
            'label': f"{geometry_type.capitalize()} mesh"
        })

    # Parallel processing
    final_tubes = []
    final_sections = []

    for job in jobs:
        # Prepare tasks for parallel processing
        tasks = []
        for i in range(num_elements):
            elem_section = sections[sections_idx[i]] if sections_idx[i] >= 0 else None
            tasks.append((job['coords'][i], elem_section, job['rot_matrices'][i]))

        # Start parallel processing
        results = Parallel(n_jobs=-1)(
            delayed(process_element_geometry)(task)
            for task in tqdm(tasks, desc=f"Processing {job['label']}")
        )

        # Unpack results
        tubes_job = []
        sections_job = []

        if results is not None:
            for tube_res, section_res in results:
                if tube_res:
                    tubes_job.append(tube_res)
                if section_res:
                    sections_job.append(section_res)

        final_tubes.append(tubes_job)
        final_sections.append(sections_job)

    # Format output
    def _to_multiblock(data_list):
        return pv.MultiBlock(data_list) if data_list else pv.MultiBlock()

    # If static analysis, return a single MultiBlock object
    if len(final_tubes) == 1:
        out_tubes = _to_multiblock(final_tubes[0])
        out_sections = _to_multiblock(final_sections[0])
    else:
        out_tubes = [_to_multiblock(tubes) for tubes in final_tubes]
        out_sections = [ _to_multiblock(sections) for sections in final_sections ]

    return {
        'tubes': out_tubes,
        'section': out_sections
    }
