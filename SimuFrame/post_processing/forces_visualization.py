# Built-in libraries
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Any

# Third-party libraries
import numpy as np
import pyvista as pv
import numpy.typing as npt
from pyvista.plotting import QtInteractor

# Local libraries
from .visualization import add_scalars
from SimuFrame.core.model import Structure


class VisualizationType(Enum):
    """Available visualization types."""
    DIAGRAM = 'Diagram'
    COLORMAP = 'Colormap'

class ForceType(Enum):
    """Force types and their configurations."""
    FX = ('Fx', 'b', 1, 2, 'Fx (kN)')
    FY = ('Fy', 'g', 1, 1, 'Fy (kN)')
    FZ = ('Fz', 'r', 1, 2, 'Fz (kN)')
    MX = ('Mx', 'peru', -1, 2, 'Mx (kN.m)')
    MY = ('My', 'purple', -1, 2, 'My (kN.m)')
    MZ = ('Mz', 'slateblue', 1, 1, 'Mz (kN.m)')

    def __init__(self, key, color, direction, axis, label):
        self.key = key
        self.color = color
        self.direction = direction
        self.axis = axis
        self.label = label

    @classmethod
    def from_key(cls, key: str):
        """Get ForceType from key."""
        for ft in cls:
            if ft.key == key:
                return ft
        raise ValueError(f"Unknown force type: {key}")


@dataclass
class ForceData:
    """Processed force data."""
    values: np.ndarray
    magnitude: float
    title: str
    force_type: ForceType


@dataclass
class PlotConfig:
    """Plotting configuration."""
    scale: float = 1.0
    force: str = 'Fx'
    visualization: str = 'Colormap'
    MT: float = 0.0
    section_grid: Optional[Dict[str, Any]] = None


class ForceDataExtractor:
    """Extracts and processes force data."""

    @staticmethod
    def extract(
        internal_forces,
        force_key: str
    ) -> ForceData:
        """
        Extracts force data from the structure.

        Args:
            internal_forces (dict): Dictionary containing internal forces.
            force_key (str): Key of the force type ('Fx', 'My', etc).

        Returns:
            ForceData with processed values.
        """
        force_type = ForceType.from_key(force_key)

        # Extract values
        values = internal_forces[force_key]

        # Calcular magnitude
        max_val = np.max(np.abs(values))
        magnitude = 1.0 / max_val if max_val > 1e-9 else 1.0

        return ForceData(
            values=values,
            magnitude=magnitude,
            title=force_type.label,
            force_type=force_type
        )

class DiagramPlotter:
    """Plots force diagrams."""

    def __init__(
        self,
        widget: QtInteractor,
        structure: Structure,
        coords: npt.NDArray[np.float64]
    ) -> None:
        # Store attributes
        self.widget = widget
        self.structure = structure
        self.coords = coords

    def plot(self, force_data: ForceData, config: PlotConfig):
        """Plots the force diagram."""
        self._hide_deformed_mesh()

        deformed_points = self._compute_deformed_points(force_data, config)
        diagram_mesh = self._create_diagram_mesh(deformed_points)

        # Create labels
        point_labels, text_labels = self._create_labels(force_data.values, deformed_points)

        self._add_to_plotter(
            diagram_mesh,
            point_labels,
            text_labels,
            force_data
        )

    def _hide_deformed_mesh(self):
        """Hides deformed mesh actors."""
        for actor_name in ['deformed_mesh_tubes', 'deformed_mesh_sections']:
                    if actor_name in self.widget.actors:
                        self.widget.actors[actor_name].visibility = False

        try:
            self.widget.scalar_bar.SetVisibility(False)
        except (AttributeError, StopIteration):
            pass

    def _compute_deformed_points(self, force_data: ForceData, config: PlotConfig) -> npt.NDArray[np.float64]:
        """Calculates points deformed by the force diagram."""
        # Get number of nodes from coords
        num_nodes = self.coords.shape[1]

        # Scale forces
        scaled_f = (force_data.magnitude * force_data.values * config.scale)

        # Create 3D force array for the correct axis
        f = np.zeros((self.structure.num_elements, num_nodes, 3))
        axis = force_data.force_type.axis
        direction = force_data.force_type.direction
        f[:, :, axis] = direction * scaled_f

        # Transform to global system
        f_global = np.einsum('eji,enj->eni', config.MT, f)

        return self.coords + config.scale * f_global

    def _create_diagram_mesh(self, force_vec: npt.NDArray[np.float64]) -> pv.PolyData:
        """Creates the polygonal mesh for the diagram."""
        num_elements, num_nodes, _ = self.coords.shape
        num_segments = num_nodes - 1

        # Combine start and end points
        combined_points = np.stack([self.coords, force_vec], axis=2)
        points = combined_points.reshape(-1, 3)


        # Generate indices
        node_indices = np.arange(num_segments)
        elem_indices = np.arange(num_elements)[:, np.newaxis]

        # Calculate start index of segment
        start_index = (elem_indices * (num_nodes * 2)) + (node_indices * 2)
        indices = start_index.flatten()

        # Define quad vertices
        p0 = indices        # Bottom left node
        p1 = indices + 1    # Top left node
        p2 = indices + 3    # Top right node
        p3 = indices + 2    # Bottom right node

        # Generate faces array (shape: (4, p0, p1, p2, p3, 4, p0...))
        total_quads = num_elements * num_segments
        fours = np.full(total_quads, 4)
        faces = np.column_stack((fours, p0, p1, p2, p3)).flatten()

        return pv.PolyData(points, faces)

    def _create_labels(self, values: npt.NDArray[np.float64], diagram_points: npt.NDArray[np.float64]) -> Tuple[list, list]:
        """Create labels for member ends."""
        lbl_points = []
        lbl_texts = []

        # Initial data
        num_elements = self.structure.num_elements
        num_initial_members = len(self.structure.original_members)
        stride = num_elements // num_initial_members

        # Indices for the initial and final members
        start_indices = np.arange(0, num_elements, stride)
        end_indices = start_indices + stride - 1

        # Extract values for the initial and final members
        start_vals = values[start_indices, 0]
        end_vals = values[end_indices, -1]

        # Extract position for the initial and final members
        start_points = diagram_points[start_indices, 0]
        end_points = diagram_points[end_indices, -1]

        # Process labels
        tol = 1e-9
        for vals, points in [(start_vals, start_points), (end_vals, end_points)]:
            mask = np.abs(vals) > tol
            for val, point in zip(vals[mask], points[mask]):
                lbl_points.append(point)
                lbl_texts.append(f"{val:.3f}")

        return lbl_points, lbl_texts

    def _add_to_plotter(
        self,
        diagram_mesh,
        label_points,
        label_texts,
        force_data: ForceData):
        """Adds mesh and labels to the plotter."""
        # Add diagram
        self.widget.add_mesh(
            diagram_mesh,
            color=force_data.force_type.color,
            opacity=0.7,
            name='force_diagram'
        )

        # Add title
        self.widget.add_text(
            force_data.title,
            position='upper_left',
            font_size=10,
            name='force_title'
        )

        # Add labels
        if label_points:
            self.widget.add_point_labels(
                label_points,
                label_texts,
                name='force_labels',
                font_size=16,
                text_color='black',
                shape=None,
                show_points=False,
                always_visible=True
            )


class ColormapPlotter:
    """Plots force colormaps."""

    def __init__(self, widget, deformed_mesh):
        self.widget = widget
        self.deformed_mesh = deformed_mesh

    def plot(self, force_data: ForceData, config: PlotConfig):
        """Plots the internal forces colormap."""
        if not self._mesh_exists():
            self._plot_new_mesh(force_data)
        else:
            self._update_existing_mesh(force_data)

        self._ensure_scalar_bar_visible()

    def _ensure_scalar_bar_visible(self):
        """Ensures the scalar bar is visible."""
        if hasattr(self.widget, 'scalar_bar') and self.widget.scalar_bar is not None:
            self.widget.scalar_bar.SetVisibility(True)

    def _mesh_exists(self) -> bool:
        """Checks if the mesh already exists in the scene."""
        return ('deformed_mesh_tubes' in self.widget.actors and
                'deformed_mesh_sections' in self.widget.actors)

    def _plot_new_mesh(self, force_data: ForceData):
        """Creates and plots a new mesh."""
        self._clear_old_actors()

        # Extract and combine meshes
        tubes_grid = self.deformed_mesh['tubes'].combine()
        section_grid = self.deformed_mesh['section'].combine()

        # Add scalars
        initial_values = force_data.values[:, 0]
        final_values = force_data.values[:, 1]
        add_scalars(self.deformed_mesh['section'], section_grid, initial_values, final_values)

        # Colormap limits
        vmin, vmax = force_data.values.min(), force_data.values.max()

        # Configure scalar bar
        scalar_bar_args = {
            'title': force_data.title,
            'title_font_size': 20,
            'label_font_size': 16,
            'n_labels': 10,
            'vertical': True,
            'fmt': '%.3e'
        }

        # Add to plotter
        self.widget.add_mesh(
            tubes_grid,
            color='sienna',
            opacity=1.0,
            name='deformed_mesh_tubes'
        )
        self.widget.add_mesh(
            section_grid,
            scalars='scalars',
            cmap="turbo",
            clim=(vmin, vmax),
            scalar_bar_args=scalar_bar_args,
            name='deformed_mesh_sections'
        )

    def _update_existing_mesh(self, force_data: ForceData):
        """Updates the existing mesh with new data."""
        # Ensure visibility
        for actor in ['deformed_mesh_tubes', 'deformed_mesh_sections']:
            if actor in self.widget.actors:
                self.widget.actors[actor].visibility = True

        # Get dataset and actors
        sections = self.deformed_mesh['section']
        section_actor = self.widget.actors['deformed_mesh_sections']
        section_grid = section_actor.mapper.dataset

        # Colormap limits
        initial_values = force_data.values[:, 0]
        final_values = force_data.values[:, 1]
        vmin, vmax = force_data.values.min(), force_data.values.max()

        # Update scalars
        add_scalars(sections, section_grid, initial_values, final_values)

        # Update mapper and scalar bar
        section_actor.mapper.scalar_range = (vmin, vmax)
        self.widget.update_scalar_bar_range([vmin, vmax])

        if self.widget.scalar_bars:
            self.widget.scalar_bar.SetTitle(force_data.title)
        else:
            self._add_scalar_bar(force_data.title)

    def _clear_old_actors(self):
        """Removes old actors except the undeformed structure."""
        actors_to_keep = [
            'undeformed_structure_sections', 'undeformed_structure_nodes',
            'fixed_supports', 'pinned_supports', 'translation_supports', 'rotation_supports'
        ]

        for name in list(self.widget.actors.keys()):
            if name not in actors_to_keep:
                self.widget.remove_actor(name)

    def _add_scalar_bar(self, title: str):
        """Adiciona barra de escalares."""
        self.widget.add_scalar_bar(
            title=title,
            title_font_size=20,
            label_font_size=16,
            n_labels=10,
            vertical=True,
            fmt='%.3e'
        )


class CameraManager:
    """Manages camera state."""

    def __init__(self, widget):
        self.widget = widget
        self.saved_state = None

    def save(self):
        """Saves current camera state."""
        self.saved_state = self.widget.camera.copy()

    def restore(self):
        """Restores saved camera state."""
        if self.saved_state is not None:
            self.widget.camera = self.saved_state

    def __enter__(self):
        """Context manager: saves on enter."""
        self.save()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: restores on exit."""
        self.restore()


def plot_internal_forces(
    structure: Structure,
    deformed_mesh: Dict[str, pv.MultiBlock],
    coords: npt.NDArray[np.float64],
    internal_forces: npt.NDArray[np.float64],
    widget: QtInteractor,
    **kwargs
) -> None:
    """
    Plots internal forces on the structure.

    Args:
        structure (Structure): Instance of the Structure class.
        deformed_mesh (dict): Dictionary containing deformed meshes (tubes and sections).
        coords (np.ndarray): Coordinates of the points.
        internal_forces (np.ndarray): Array of internal forces.
        widget (QtInteractor): PyVista interactor widget for plotting.
        **kwargs: Optional arguments (scale, force, visualization, MT, section_grid)
    """
    config = PlotConfig(
        scale=kwargs.get('scale', 1.0),
        force=kwargs.get('force', 'Fx'),
        visualization=kwargs.get('visualization', 'Colormap'),
        MT=kwargs.get('MT', []),
        section_grid=kwargs.get('section_grid')
    )

    with CameraManager(widget):
        force_data = ForceDataExtractor.extract(internal_forces, config.force)

        # Plot according to visualization type
        if config.visualization == VisualizationType.DIAGRAM.value:
            plotter = DiagramPlotter(widget, structure, coords)
            plotter.plot(force_data, config)

        elif config.visualization == VisualizationType.COLORMAP.value:
            plotter = ColormapPlotter(widget, deformed_mesh)
            plotter.plot(force_data, config)

        else:
            raise ValueError(f"Invalid visualization type: {config.visualization}")

        _cleanup_auxiliary_actors(widget)


def _cleanup_auxiliary_actors(widget):
    """Removes old auxiliary actors."""
    actors_to_remove = [
        'max_displacement_sphere',
        'max_displacement_label'
    ]

    for name in actors_to_remove:
        if name in widget.actors:
            widget.remove_actor(name)
