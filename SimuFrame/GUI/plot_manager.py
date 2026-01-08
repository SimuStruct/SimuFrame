# Built-in libraries
from typing import Dict, Optional, List
from abc import ABC, abstractmethod
from contextlib import contextmanager

# Third-party libraries
import pyvista as pv

# Local libraries
from SimuFrame.post_processing.visualization import plot_structure
from SimuFrame.post_processing.forces_visualization import plot_internal_forces
from SimuFrame.post_processing.displacement_visualization import plot_displacements
from SimuFrame.post_processing.support_visualization import plot_reactions


class PlotCommand(ABC):
    """Abstract command for plotting operations."""

    @abstractmethod
    def execute(self, plotter, state, **kwargs):
        """Execute the plotting command."""
        pass


class PlotDisplacementCommand(PlotCommand):
    """Command for plotting displacements."""

    def __init__(self, component: str):
        self.component = component

    def execute(self, plotter, state, **kwargs):
        """Plot displacement results."""
        if state.results.displacements is None:
            return

        viz_options = state.get_visualization_options()
        viz_options.update(kwargs)
        viz_options['component'] = self.component

        plot_displacements(
            state.structure,
            state.meshes.deformed,
            state.meshes.deformed_coords,
            state.results.displacements,
            plotter,
            **viz_options
        )


class PlotForceCommand(PlotCommand):
    """Command for plotting internal forces."""

    def __init__(self, force_type: str):
        self.force_type = force_type

    def execute(self, plotter, state, **kwargs):
        """Plot internal force results."""
        if state.results.internal_forces is None:
            return

        viz_options = state.get_visualization_options()
        viz_options.update(kwargs)
        viz_options['force'] = self.force_type

        plot_internal_forces(
            state.structure,
            state.meshes.deformed,
            state.meshes.coords,
            state.results.internal_forces,
            plotter,
            **viz_options
        )


class PlotReactionCommand(PlotCommand):
    """Command for plotting support reactions."""

    def execute(self, plotter, state, reactions_to_plot: Optional[List[str]] = None, **kwargs):
        """Plot support reaction results."""
        if state.results.support_reactions is None:
            return

        plot_reactions(
            plotter,
            state.results.support_reactions,
            components=reactions_to_plot or []
        )


class PlotManager:
    """Central manager for all plotting operations."""

    def __init__(self, plotter, state_manager):
        self.plotter = plotter
        self.state = state_manager
        self.commands = self._create_commands()
        self.dark_mode = False

    @contextmanager
    def render_lock(self):
        """Lock VTK rendering during batch operations to improve performance."""
        render_window = self.plotter.ren_win
        render_window.SetSwapBuffers(0)

        try:
            yield
        finally:
            render_window.SetSwapBuffers(1)
            self.plotter.render()

    @staticmethod
    def _create_commands() -> Dict[str, PlotCommand]:
        """Create dictionary of available plotting commands."""
        return {
            # Global displacements
            'u': PlotDisplacementCommand('u'),
            'ux': PlotDisplacementCommand('x'),
            'uy': PlotDisplacementCommand('y'),
            'uz': PlotDisplacementCommand('z'),
            'θx': PlotDisplacementCommand('θx'),
            'θy': PlotDisplacementCommand('θy'),
            'θz': PlotDisplacementCommand('θz'),

            # Internal forces
            'fx': PlotForceCommand('Fx'),
            'fy': PlotForceCommand('Fy'),
            'fz': PlotForceCommand('Fz'),
            'mx': PlotForceCommand('Mx'),
            'my': PlotForceCommand('My'),
            'mz': PlotForceCommand('Mz'),

            # Support reactions
            'support_reactions': PlotReactionCommand()
        }

    def clear_result_actors(self):
        """Remove only result-related actors from the scene."""
        actors_to_remove = [
            # Force diagrams and labels
            'force_diagram',
            'force_labels',
            'force_legends',

            # Reactions
            'reaction_forces',
            'reaction_forces_labels',
            'reaction_moments',
            'reaction_moments_labels',

            # Displacement markers
            'max_displacement_sphere',
            'max_displacement_label'
        ]

        for name in actors_to_remove:
            try:
                self.plotter.remove_actor(name, render=False)
            except KeyError:
                pass

    def plot_result(self, result_key: str, **options):
        """Plot a specific result using the command pattern."""
        if result_key not in self.commands:
            print(f"[PlotManager] Key '{result_key}' not found.")
            return

        command = self.commands[result_key]

        with self.render_lock():
            self.clear_result_actors()
            command.execute(self.plotter, self.state, **options)

    def plot_base_structure(self):
        """Plot only the base undeformed structure."""
        if not self.state.has_structure():
            return

        self.plotter.clear_actors()
        self.setup_initial_view()

        if self.state.meshes.undeformed:
            plot_structure(
                self.plotter,
                self.state.structure,
                self.state.meshes.undeformed,
                opacity=0.2,
                plot_section=self.state.viz_options.plot_section,
                plot_loads=False,
                plot_nodes=False,
                name='undeformed_structure'
            )

    def update_section_visibility(self, visible: bool):
        """Update visibility of cross-sections on the undeformed structure."""
        try:
            actor = self.plotter.actors.get('undeformed_structure_sections')
            if actor:
                actor.visibility = visible
                self.plotter.render()
        except (KeyError, AttributeError):
            pass

    def toggle_background(self):
        """Toggle between light and dark background."""
        if self.dark_mode:
            self.plotter.set_background('darkgray', top='white')
            self.dark_mode = False
        else:
            self.plotter.set_background('midnightblue', top='black')
            self.dark_mode = True

    def take_screenshot(self, filepath: str):
        """Save a screenshot of the current view."""
        self.plotter.screenshot(filepath)

    def setup_initial_view(self):
        """Configure the initial view with axes and orientation widget."""
        self.plotter.add_axes()
        self.plotter.add_orientation_widget(pv.AxesActor())

    def reset_camera(self):
        """Reset camera to default view."""
        self.plotter.reset_camera()
