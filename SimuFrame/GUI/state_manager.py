# Third-party libraries
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable

# Local libraries
from SimuFrame.core.model import Structure

@dataclass
class MeshData:
    """Stores mesh data for structure visualization."""
    undeformed: Optional[Any] = None
    deformed: Optional[Any] = None
    sections: Optional[Any] = None
    sections_idx: Optional[Any] = None
    coords: Optional[np.ndarray] = None
    deformed_coords: Optional[np.ndarray] = None


@dataclass
class ResultsData:
    """Stores analysis results data."""
    internal_forces: Optional[Dict] = None
    displacements: Optional[Dict] = None
    support_reactions: Optional[Dict] = None
    eigenvalues: Optional[np.ndarray] = None
    convergence_data: Optional[Dict] = None
    history: Optional[list] = None
    MT: Optional[np.ndarray] = None


@dataclass
class AnalysisConfig:
    """Configuration settings for structural analysis."""
    analysis: str = 'Linear'
    nonlinear_method: str = 'Newton-Raphson'
    initial_steps: int = 10
    max_iterations: int = 50
    max_load_factor: float = 1.0
    arc_type: str = 'Cylindrical'
    psi: float = 1.0
    subdivisions: List[int] = field(default_factory=lambda: [1])
    num_modes: int = 0
    buckling_mode: int = 0


@dataclass
class VisualizationOptions:
    """Options for result visualization."""
    scale: float = 1.0
    plot_section: bool = True
    viz_style: str = 'Colormap'
    dark_mode: bool = False


class StateManager:
    """Central state manager for application data and configuration."""
    def __init__(self):
        self.structure: Structure
        self.meshes = MeshData()
        self.results = ResultsData()
        self.config = AnalysisConfig()
        self.viz_options = VisualizationOptions()
        self._observers: List[Callable] = []

    def reset_results(self):
        """Reset only results data while keeping the structure."""
        cached_structure = self.structure
        self.meshes = MeshData()
        self.results = ResultsData()
        self.config = AnalysisConfig()
        self.structure = cached_structure
        self._notify_observers('results_reset')

    def reset_all(self):
        """Reset entire application state."""
        self.__init__()
        self._notify_observers('state_reset')

    def set_structure(self, structure):
        """Set the loaded structure."""
        self.structure = structure
        self._notify_observers('structure_loaded')

    def set_analysis_results(self, **kwargs):
        """Update analysis results from solver output."""
        # Update structure if provided
        if 'structure' in kwargs:
            self.structure = kwargs['structure']

        # Update meshes
        self.meshes.undeformed = kwargs.get('undeformed_mesh')
        self.meshes.deformed = kwargs.get('deformed_mesh')
        self.meshes.coords = kwargs.get('coords')
        self.meshes.deformed_coords = kwargs.get('deformed_coords')

        # Update results
        self.results.internal_forces = kwargs.get('internal_forces')
        self.results.displacements = kwargs.get('displacements')
        self.results.support_reactions = kwargs.get('support_reactions')
        self.results.eigenvalues = kwargs.get('eigenvalues')
        self.results.convergence_data = kwargs.get('convergence_data')
        self.results.history = kwargs.get('history')
        self.results.MT = kwargs.get('MT')

        # Update configuration
        if self.structure is not None:
            self.config.num_modes = (
                self.results.eigenvalues.size
                if self.results.eigenvalues is not None else 0
            )

            analise_map = {
                'linear': 'Linear',
                'nonlinear': 'Nonlinear',
                'buckling': 'Buckling'
            }
            self.config.analysis = analise_map.get(self.config.analysis, 'Linear')

        self._notify_observers('results_updated')

    def has_structure(self) -> bool:
        """Check if a structure is loaded."""
        return self.structure is not None

    def has_results(self) -> bool:
        """Check if analysis results are available."""
        return (
            self.results.displacements is not None or
            self.results.internal_forces is not None
        )

    def is_buckling_analysis(self) -> bool:
        """Check if current analysis is buckling analysis."""
        return self.config.analysis == 'Buckling'

    def add_observer(self, callback: Callable):
        """Add an observer for state changes."""
        if callback not in self._observers:
            self._observers.append(callback)

    def remove_observer(self, callback: Callable):
        """Remove an observer."""
        if callback in self._observers:
            self._observers.remove(callback)

    def _notify_observers(self, event_type: str):
        """Notify all observers about state changes."""
        for observer in self._observers:
            try:
                observer(event_type, self)
            except Exception as e:
                print(f"Error notifying observer: {e}")

    def get_current_buckling_mode(self) -> int:
        """Get the current buckling mode index."""
        return self.config.buckling_mode

    def set_buckling_mode(self, mode: int):
        """Set the current buckling mode index."""
        if 0 <= mode < self.config.num_modes:
            self.config.buckling_mode = mode
            self._notify_observers('buckling_mode_changed')

    def get_visualization_options(self) -> Dict[str, Any]:
        """Get visualization options as a dictionary."""
        return {
            'scale': self.viz_options.scale,
            'plot_section': self.viz_options.plot_section,
            'sections': self.meshes.sections,
            'buckling_mode': self.config.buckling_mode,
            'MT': self.results.MT,
        }
