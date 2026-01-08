# Third-party libraries
import numpy as np
import pyvista as pv


def plot_supports(plotter, structure, support_scale_factor):
    """
    Plots the structure supports with distinct visual symbols.

    Recognized types:
        - Fixed: Solid cube (red)
        - Pinned: Sphere (green)
        - Partial constraints: Cones and discs
    """
    # Collect meshes by type
    fixed_meshes = []
    pinned_meshes = []
    translation_meshes = []
    rotation_meshes = []

    # Standard support patterns
    SET_FIXED = {0, 1, 2, 3, 4, 5}
    SET_PINNED = {0, 1, 2}

    # Define element scaling
    s = support_scale_factor
    size_block = s * 0.5
    size_joint = s * 0.4
    size_arrow = s * 0.35
    offset_base = s * 0.35

    # Cartesian axes directions
    axes = {
        0: np.array([1, 0, 0]),  # X
        1: np.array([0, 1, 0]),  # Y
        2: np.array([0, 0, 1])   # Z
    }

    # Process each node in the structure
    for idx in range(structure.num_nodes):
        node = structure.nodes[idx]
        center = np.array(node.coord)
        bc_set = node.boundary_conditions

        # Special case: Fixed
        if bc_set == SET_FIXED:
            cube = pv.Cube(
                center=center,
                x_length=size_block,
                y_length=size_block,
                z_length=size_block
            )
            fixed_meshes.append(cube)
            continue

        # Special case: Pinned
        if bc_set == SET_PINNED:
            sphere = pv.Sphere(
                radius=size_joint * 0.7,
                center=center,
                theta_resolution=16,
                phi_resolution=16
            )
            pinned_meshes.append(sphere)
            continue

        # Translation constraints (DOFs 0, 1, 2)
        for dof in range(3):
            if dof not in bc_set:
                continue

            axis_vec = axes[dof]
            offset_pos = center - axis_vec * offset_base

            # Cone pointing towards the node
            cone = pv.Cone(
                center=offset_pos,
                direction=axis_vec,
                height=size_arrow,
                radius=size_arrow * 0.25,
                resolution=8
            )
            translation_meshes.append(cone)

        # Rotation constraints (DOFs 3, 4, 5)
        for dof in range(3, 6):
            if dof not in bc_set:
                continue

            axis_vec = axes[dof - 3]

            # Create thin disc perpendicular to axis
            # The disc represents the plane blocked by the constraint
            offset_pos = center + axis_vec * (offset_base * 0.8)

            disc = pv.Disc(
                center=offset_pos,
                inner=0,
                outer=size_arrow * 0.45,
                normal=axis_vec,
                r_res=1,
                c_res=16
            )
            rotation_meshes.append(disc)

    # Add meshes to the plotter by category
    if fixed_meshes:
        combined = pv.MultiBlock(fixed_meshes).combine()
        plotter.add_mesh(
            combined,
            color='#dc2626',
            opacity=0.9,
            label='Fixed',
            name='fixed_supports'
        )

    if pinned_meshes:
        combined = pv.MultiBlock(pinned_meshes).combine()
        plotter.add_mesh(
            combined,
            color='#16a34a',
            opacity=0.9,
            label='Pinned Constraints',
            name='pinned_supports'
        )

    if translation_meshes:
        combined = pv.MultiBlock(translation_meshes).combine()
        plotter.add_mesh(
            combined,
            color='#2563eb',
            opacity=0.85,
            label='Translation Constraints',
            name='translation_supports'
        )

    if rotation_meshes:
        combined = pv.MultiBlock(rotation_meshes).combine()
        plotter.add_mesh(
            combined,
            color='#ea580c',
            opacity=0.75,
            label='Rotation Constraints',
            name='rotation_supports'
        )


def _create_moment_arrow():
    """Create a double-headed arrow geometry for moment reactions."""
    shaft = pv.Arrow(tip_length=0.25, shaft_radius=0.02, tip_radius=0.05, scale=0.8)
    # Create a second tip to simulate the double arrow
    tip2 = pv.Cone(radius=0.04, height=0.25, center=(0.60, 0, 0), direction=(1, 0, 0))
    return shaft + tip2

def plot_reactions(plotter, support_data, components=None, scale_factor=1.00, offset_dist=1.25):
    """
    Plot support reactions (forces and moments).

    Args:
        plotter (QtInteractor): Plotter instance.
        support_data (dict): Dictionary co  ntaining the reactions and its coordinates.
        components (list): List of keys to plot e.g., ['rx', 'rmx'].
        scale_factor (float): Global scale factor for the arrows.
        offset_dist (float): Distance to offset the arrow tail from the node.
    """
    # Cleanup previous actors
    plotter.remove_actor('reaction_forces', render=False)
    plotter.remove_actor('reaction_moments', render=False)
    plotter.remove_actor('reaction_forces_labels', render=False)
    plotter.remove_actor('reaction_moments_labels', render=False)

    # if no components are specified, remove all and return
    if not components:
        plotter.render()
        return

    # Save current camera state
    camera_state = plotter.camera.copy()

    # Unpack data
    Ra = np.squeeze(support_data['R'])
    coords = support_data['coords']

    # Mapping for components
    dof_map = {
        'rx': 0, 'ry': 1, 'rz': 2,
        'rmx': 3, 'rmy': 4, 'rmz': 5
    }

    # Filter valid components based on user request
    requested_dofs = [dof_map[dof] for dof in components if dof in dof_map]
    if not requested_dofs:
        return

    def _render_reactions(dof_indices, name, color, geom, unit_label):
        """
        Internal function to add a glyph to the plotter.

        Args:
            dof_indices (list): List of indices corresponding to the requested DOFs.
            name (str): Name of the glyph (forces or moments).
            color (str): Color name for the different arrows.
            geom (PolyData): Geometry source for the glyphs (single or double arrows).
            unit_label (str): Unit label for the labels (e.g., 'kN' or 'kNm').
        """
        current_dofs = [dof for dof in requested_dofs if dof in dof_indices]

        if not current_dofs:
            return

        # Prepare containers
        points_list = []
        vecs_list = []
        labels_list = []
        pos_list = []

        # Unit vector
        base_vec = np.eye(3)
        tol = 1e-6

        # Iterate through requested DOFs
        for dof in current_dofs:
            axis = dof % 3
            values = Ra[:, dof]

            # Get the nodes where the reaction is non-zero
            active_mask = np.abs(values) > tol
            if not np.any(active_mask):
                continue

            # Extract valid nodes values
            active_coords = coords[active_mask]
            active_values = values[active_mask]

            # Direction calculation
            directions = np.outer(np.sign(active_values), base_vec[axis])

            # Calculate start points (tails) of each arrow
            if dof in [3, 4, 5]:
                tails = active_coords - 2 * offset_dist * directions
            else:
                tails = active_coords - offset_dist * directions

            # Store data for glyphs
            points_list.append(tails)
            vecs_list.append(directions)

            # Labels placed behind the tail
            labels_offset = tails - 0.1 * directions
            pos_list.append(labels_offset)

            # Formatted text
            labels_list.extend([f'{abs(val):.2f} {unit_label}' for val in active_values])

        if not points_list:
            return

        # Stack arrays
        points = np.vstack(points_list)
        vecs = np.vstack(vecs_list)
        pos = np.vstack(pos_list)

        # Create PolyData
        polydata = pv.PolyData(points)
        polydata['vectors'] = vecs

        # Create glyphs
        glyphs = polydata.glyph(
            orient='vectors',
            scale=False,
            factor=scale_factor,
            geom=geom,
            tolerance=0.0
        )

        # Add mesh
        plotter.add_mesh(glyphs, name=name, color=color, opacity=1.0)

        # Add labels
        plotter.add_point_labels(
            pos,
            labels_list,
            name=f'{name}_labels',
            font_size=16,
            text_color=color,
            shape_opacity=0.0,
            show_points=False,
            always_visible=True
        )

    # Plot forces (Indices 0, 1, 2)
    _render_reactions(
        dof_indices=[0, 1, 2],
        name='reaction_forces',
        color='crimson',
        geom=pv.Arrow(tip_length=0.25, shaft_radius=0.02, tip_radius=0.05, scale=0.8),
        unit_label='kN')

    # Plot moments (Indices 3, 4, 5)
    _render_reactions(
        dof_indices=[3, 4, 5],
        name='reaction_moments',
        color='royalblue',
        geom=_create_moment_arrow(),
        unit_label='kNm')

    # Restaurar o estado da câmera
    plotter.camera = camera_state
