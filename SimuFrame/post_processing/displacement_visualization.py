# Built-in libraries
from typing import Dict, List

# Third-party libraries
import numpy as np
import pyvista as pv
import numpy.typing as npt
from pyvistaqt import QtInteractor

# Local libraries
from .visualization import add_scalars
from SimuFrame.core.model import Structure

def plot_displacements(
    structure: Structure,
    deformed_mesh: Dict[str, pv.MultiBlock],
    deformed_coords: npt.NDArray[np.float64],
    global_deformation: npt.NDArray[np.float64],
    widget: QtInteractor,
    **kwargs
) -> None:
    """
    Plots structure displacements with colormap.

    Args:
        structure (Structure): Instance of the Structure class.
        deformed_mesh (dict): Dictionary with meshes {'tubes': ..., 'section': ...}.
        deformed_coords (np.ndarray): Coordinates of deformed points.
        global_deformation (np.ndarray): Array with displacements [elements, nodes, DOFs].
        widget (QtInteractor): PyVista plotter.
        **kwargs: buckling_mode, component, section_grid.
    """
    # Extract parameters
    buckling_mode: int = kwargs.get('buckling_mode', 0)
    component: str = kwargs.get('component', 'u')

    # Save camera state
    camera_state = widget.camera.copy()

    # Component map
    all_components = {
        'u': (slice(None), 'U (m)', 'Displacement Magnitude'),
        'x': (0, 'UX (m)', 'Displacement in X'),
        'y': (1, 'UY (m)', 'Displacement in Y'),
        'z': (2, 'UZ (m)', 'Displacement in Z'),
        'θx': (3, 'Rx (rad)', 'Rotation about X'),
        'θy': (4, 'Ry (rad)', 'Rotation about Y'),
        'θz': (5, 'Rz (rad)', 'Rotation about Z')
    }

    if component not in all_components:
        raise ValueError(f"Component '{component}' not supported. Use: {list(all_components.keys())}")

    axis_idx, title_label, _ = all_components[component]

    # Check if mesh recalculation is needed (only for buckling)
    needs_recalc = _needs_mesh_recalculation(widget, structure, buckling_mode)

    if needs_recalc:
        _plot_new_displacement_mesh(
            widget, structure, deformed_mesh, global_deformation,
            deformed_coords, component, axis_idx, title_label, buckling_mode
        )
    else:
        _update_existing_displacement_mesh(
            widget, global_deformation, deformed_mesh, component,
            axis_idx, title_label, buckling_mode, structure
        )

    #  maximum displacement marker
    _add_max_displacement_marker(
        widget, global_deformation, deformed_coords,
        component, axis_idx, buckling_mode, structure
    )

    # Restore camera
    widget.camera = camera_state


def _needs_mesh_recalculation(widget: QtInteractor, structure: Structure, mode: int) -> bool:
    """Checks if mesh recalculation is needed."""
    if not structure.is_buckling:
        return not _mesh_exists(widget)

    # For buckling, check mode change
    if not hasattr(widget, 'previous_state'):
        return True

    return widget.previous_state.get('mode') != mode


def _mesh_exists(widget: QtInteractor) -> bool:
    """Checks if deformed mesh already exists."""
    return ('deformed_mesh_tubes' in widget.actors and
            'deformed_mesh_sections' in widget.actors)


def _plot_new_displacement_mesh(
    widget: QtInteractor,
    structure: Structure,
    deformed_mesh: Dict[str, pv.MultiBlock],
    global_deformation: npt.NDArray[np.float64],
    deformed_coords: npt.NDArray[np.float64],
    component: str,
    axis_idx: int | slice,
    title_label: str,
    mode: int
) -> None:
    """Creates and plots new displacement mesh."""

    # Clear only result actors
    _clear_result_actors(widget)

    # Extract displacement values
    values: npt.NDArray[np.float64] = _extract_displacement_values(
        global_deformation, component, axis_idx, mode, structure
    )

    # Colormap limits
    initial_values: npt.NDArray[np.float64] = values[:, 0]
    final_values: npt.NDArray[np.float64] = values[:, -1]
    vmin, vmax = values.min(), values.max()

    # Extract meshes
    tubes = deformed_mesh['tubes'] if not structure.is_buckling else deformed_mesh['tubes'][mode]
    sections = deformed_mesh['section'] if not structure.is_buckling else deformed_mesh['section'][mode]

    # Combine meshes
    grid_tubes: pv.UnstructuredGrid = tubes.combine()
    grid_section: pv.UnstructuredGrid = sections.combine()

    # Add scalars
    add_scalars(sections, grid_section, initial_values, final_values)

    # Configure scalar bar
    scalar_bar_args = {
        'title': title_label,
        'title_font_size': 20,
        'label_font_size': 16,
        'n_labels': 10,
        'vertical': True,
        'fmt': '%.3e'
    }

    # Add to plotter
    widget.add_mesh(
        grid_tubes,
        color='sienna',
        opacity=1.0,
        name='deformed_mesh_tubes'
    )
    widget.add_mesh(
        grid_section,
        scalars='scalars',
        cmap="turbo",
        clim=(vmin, vmax),
        scalar_bar_args=scalar_bar_args,
        name='deformed_mesh_sections'
    )

    # Save state (for buckling)
    if structure.is_buckling:
        widget.previous_state = {'mode': mode}


def _update_existing_displacement_mesh(
    widget: QtInteractor,
    global_deformation: npt.NDArray[np.float64],
    deformed_mesh: Dict[str, pv.MultiBlock],
    component: str,
    axis_idx: int | slice,
    title_label: str,
    mode: int,
    structure: Structure
) -> None:
    """Updates existing mesh with new data."""
    # Ensure visibility
    widget.actors['deformed_mesh_tubes'].visibility = True
    widget.actors['deformed_mesh_sections'].visibility = True

    # Extract values
    values: npt.NDArray[np.float64] = _extract_displacement_values(
        global_deformation, component, axis_idx, mode, structure
    )

    # Colormap limits
    initial_values: npt.NDArray[np.float64] = values[:, 0]
    final_values: npt.NDArray[np.float64] = values[:, -1]
    vmin, vmax = values.min(), values.max()

    # Get actor and dataset
    section = deformed_mesh['section'] if not structure.is_buckling else deformed_mesh['section'][mode]
    section_actor = widget.actors['deformed_mesh_sections']
    section_grid = section_actor.mapper.dataset

    # Update scalars
    add_scalars(section, section_grid, initial_values, final_values)

    # Update mapper
    section_actor.mapper.scalar_range = (vmin, vmax)

    # Update scalar bar
    widget.update_scalar_bar_range([vmin, vmax])

    if widget.scalar_bars:
        widget.scalar_bar.SetTitle(title_label)
    else:
        widget.add_scalar_bar(
            title=title_label,
            title_font_size=20,
            label_font_size=16,
            n_labels=10,
            vertical=True,
            fmt='%.3e'
        )


def _extract_displacement_values(
    global_deformation: npt.NDArray[np.float64],
    component: str,
    axis_idx: int | slice,
    mode: int,
    structure: Structure
) -> npt.NDArray[np.float64]:
    """
    Extracts displacement values based on component.

    Returns:
        np.ndarray: Displacement values [elements, nodes]
    """
    if component == 'u':
        # Total displacement (norm of translations)
        if structure.is_buckling:
            trans_displ = global_deformation[mode, :, :, :3]
        else:
            trans_displ = global_deformation[:, :, :3]

        values = np.linalg.norm(trans_displ, axis=2)
    else:
        # Specific component
        if structure.is_buckling:
            values = global_deformation[mode, :, :, axis_idx]
        else:
            values = global_deformation[:, :, axis_idx]

    return values


def _add_max_displacement_marker(
    widget: QtInteractor,
    global_deformation: npt.NDArray[np.float64],
    deformed_coords: npt.NDArray[np.float64],
    component: str,
    axis_idx: int | slice,
    mode: int,
    structure: Structure
) -> None:
    """Adds visual marker at maximum displacement point."""
    # Remove old markers
    for actor_name in ['max_displacement_sphere', 'max_displacement_label']:
        try:
            widget.remove_actor(actor_name)
        except KeyError:
            pass

    # Extract values
    valores = _extract_displacement_values(
        global_deformation, component, axis_idx, mode, structure
    )

    # Find maximum (in absolute value)
    idx_max = np.unravel_index(np.abs(valores).argmax(), valores.shape)
    valor_max = valores[idx_max]

    # Get position
    if structure.is_buckling:
        pos_max = deformed_coords[mode][idx_max[0], idx_max[1]]
    else:
        pos_max = deformed_coords[idx_max[0], idx_max[1]]

    # Create sphere at maximum point
    esfera = pv.Sphere(radius=0.05, center=pos_max)
    widget.add_mesh(
        esfera,
        color='black',
        opacity=0.8,
        name='max_displacement_sphere'
    )

    # Add label
    widget.add_point_labels(
        pos_max,
        [f'Max: {valor_max:.3e}'],
        font_size=28,
        text_color='red',
        shape_color='white',
        shape_opacity=0.8,
        margin=5,
        always_visible=True,
        name='max_displacement_label'
    )


def _clear_result_actors(widget):
    """Removes old actors, except undeformed structure."""
    actors_to_keep: List[str] = [
        'undeformed_structure_sections', 'undeformed_structure_nodes',
        'fixed_supports', 'pinned_supports', 'translation_supports', 'rotation_supports'
    ]
    actors_to_remove = [
        name for name in widget.actors.keys()
        if name not in actors_to_keep
    ]

    for name in actors_to_remove:
        widget.remove_actor(name)
