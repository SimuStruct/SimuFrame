# Built-in libraries
import traceback

# Third-party libraries
import numpy as np
from ruamel.yaml import YAML
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox, QFileDialog

# Local libraries
from SimuFrame.core.model import Structure
from SimuFrame.core.solver import solve_structure
from SimuFrame.core.assembly import (
    extract_external_forces, compute_equivalent_nodal_forces, assemble_global_force_vector, assemble_internal_forces,
    degrees_of_freedom, extract_support_reactions, assemble_elastic_stiffness_matrix
)
from SimuFrame.post_processing.visualization import generate_mesh
from SimuFrame.utils.helpers import (
    extract_element_data, orientation_vector, transformation_matrix,
    get_deformed_coords, get_global_displacements
)


class SimuController(QObject):
    """
    Controller of the GUI.
    Responsible for communication between the UI and the core functions in SimuFrame/core.
    """
    # Signals for communication with the UI
    status_message = Signal(str)
    analysis_started = Signal()
    analysis_completed = Signal()
    analysis_failed = Signal(str)
    structure_loaded = Signal()

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.state = main_window.state

    @staticmethod
    def _get_yaml_handler():
        """Returns a YAML handler configured to preserve comments and indentation."""
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)
        return yaml

    def open_structure_creator(self):
        """Open the Structure Creator window."""
        # TODO: Implement the open_structure_creator method
        pass

    def load_project_from_yaml(self):
        """Load a project from a YAML file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select YAML File",
            "SimuFrame/examples",
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )

        if not file_path:
            return

        try:
            self.status_message.emit(f"Loading {file_path}...")

            # Load YAML file
            yaml_data = self._load_yaml_file(file_path)

            # Create structure object
            structure = self._create_structure_from_yaml(yaml_data)

            if structure:
                # Store in state
                structure.yaml_path = file_path
                self.state.set_structure(structure)

                # Emit signal
                self.structure_loaded.emit()
                self.status_message.emit("File loaded successfully.")

        except Exception as e:
            error_msg = f"Failed to open file:\n\n{str(e)}"
            self._show_error("Failed to Load File", error_msg)
            self.status_message.emit("Failed to load file.")

    def _load_yaml_file(self, filepath: str) -> dict:
        """Load YAML file."""
        yaml = self._get_yaml_handler()
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return yaml.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Error decoding YAML: {e}")

    def save_yaml_file(self):
        """Save the current YAML data to the original file."""
        if not self.state.has_structure or not self.state.structure.metadata:
            self._show_error("Error", "No project loaded.")
            return False

        # Get the YAML handler
        yaml = self._get_yaml_handler()

        try:
            # Save YAML data to the original file
            with open(self.state.structure.yaml_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.state.structure.metadata, file)

            self.status_message.emit("YAML saved successfully.")
            return True

        except Exception as e:
            self._show_error("Error", f"Failed to save YAML: {e}")
            return False

    def _create_structure_from_yaml(self, yaml_data: dict) -> Structure | None:
        """Create a structure from YAML data."""
        try:
            # Extract analysis data
            analysis = yaml_data.get('analysis', {})
            analysis_type = analysis.get('analysis_type', 'linear')
            structural_model = analysis.get('element_type', 'beam')
            subdivisions = analysis.get('mesh_parameter', 1)

            # Extract structural data
            materials = yaml_data.get('materials', {})
            sections = yaml_data.get('sections', {})
            nodes = yaml_data.get('nodes', [])
            elements = yaml_data.get('elements', [])
            supports = yaml_data.get('supports', {})
            nodal_loads = yaml_data.get('nodal_loads', [])
            distributed_loads = yaml_data.get('distributed_loads', [])

            # Initialize arrays
            connec = np.zeros((len(elements), 2), dtype=int)
            coords = np.array(nodes, dtype=float)
            releases = {}
            section_properties = {}

            # Process materials
            loaded_materials = {}
            for mat_id, material in materials.items():
                loaded_materials[mat_id] = {
                    "E": float(material.get("E", 1.0)),
                    "nu": float(material.get("nu", 0.0))
                }

            # Process elements
            for elem_id, element in enumerate(elements):
                # Define element connectivity and member_releases
                connec[elem_id] = element.get('connec')
                releases[elem_id] = element.get('hinges', [[], []])

                # Check if the section exists in the database
                section_id = element.get('section_id')
                if not section_id or section_id not in sections:
                    raise ValueError(f"Section '{section_id}' for element {elem_id} not found in sections.")

                # Get element section properties
                elem_section = sections[section_id]

                # Check if the material exists in the database
                material_id = elem_section.get('material_id')
                if not material_id or material_id not in loaded_materials:
                    raise ValueError(f"Material '{material_id}' for section '{section_id}' not found in materials.")

                # Get element material properties
                elem_material = loaded_materials[material_id]

                # Get the dictionary of geometric parameters directly
                geometry_params = elem_section.get('geometry_params', {})
                if not geometry_params:
                    print(f"Warning: No 'geometry_params' found for section '{section_id}'.")

                # Define the section_data properties
                section_properties[elem_id] = {
                    "geometry": elem_section.get('geometry'),
                    **elem_material,    # Unpack material properties (E, nu, G...)
                    **geometry_params   # Unpack geometric properties (width, height, thickness...)
                }

            # Create Structure object
            structure = Structure(
                metadata=yaml_data,
                element_type=structural_model,
                subdivisions=subdivisions,
                coordinates=coords,
                connectivity=connec,
                section_data=section_properties,
                supports=supports,
                releases=releases,
                nodal_loads=nodal_loads,
                distributed_loads=distributed_loads
            )

            # Update the config state
            self.state.config.analysis = analysis_type
            self.state.config.subdivisions = subdivisions

            return structure

        except Exception as e:
            error_msg = f"Unable to load structure:\n\n{str(e)}"
            self._show_error("Error decoding YAML", error_msg)
            return None

    def update_analysis_parameters(self, parameters: dict):
        """Update the analysis parameters, if the user requests."""
        if not self.state.has_structure():
            self._show_error("Error", "No structure has been loaded.")
            return False

        # Extract new parameters
        try:
            analysis_type = parameters.get('analysis_type', 'Linear').lower()

            # Define mapping between parameter keys and YAML paths
            param_mapping = {
                'mesh_parameter': ('analysis', 'mesh_parameter'),
                'analysis_type': ('analysis', 'analysis_type'),
                'initial_steps': ('analysis', 'config', 'initial_steps'),
                'max_iterations': ('analysis', 'config', 'max_iterations'),
                'method': ('analysis', 'config', 'method'),
                'max_load_factor': ('analysis', 'config', 'max_load_factor'),
                'arc_type': ('analysis', 'config', 'arc_type'),
                'psi': ('analysis', 'config', 'psi'),
                'buckling_modes': ('analysis', 'config', 'buckling_modes')
            }

            # Update YAML metadata based on parameters
            for param_key, yaml_path in param_mapping.items():
                if param_key in parameters:
                    value = parameters[param_key]
                    if param_key == 'analysis_type':
                        value = value.lower()
                    elif param_key == 'mesh_parameter':
                        value = float(value)

                    self._set_nested_value(self.state.structure.metadata, yaml_path, value)

            # Map to display type
            ui_analysis_type = {
                'linear': 'Linear',
                'nonlinear': 'Nonlinear',
                'buckling': 'Buckling'
            }

            # Update visual configuration of the interface
            self.state.config.analysis = ui_analysis_type.get(analysis_type, 'Linear')
            self.state.config.subdivisions = parameters.get('mesh_parameter', 1)

            # Update the analysis parameters if 'nonlinear' or 'buckling' is selected
            if analysis_type == 'nonlinear':
                self.state.config.nonlinear_method = parameters.get('method', 'Newton-Raphson')
                self.state.config.initial_steps = parameters.get('initial_steps', 10)
                self.state.config.max_iterations = parameters.get('max_iterations', 100)

                if self.state.config.nonlinear_method == 'Arc-Length':
                    self.state.config.max_load_factor = parameters.get('max_load_factor', 1)
                    self.state.config.arc_type = parameters.get('arc_type', 'Spherical')
                    self.state.config.psi = parameters.get('psi', 1.0)

            elif analysis_type == 'buckling':
                self.state.config.buckling_modes = parameters.get('buckling_modes', 5)

            # Recreate the structure to apply the changes
            structure = self._create_structure_from_yaml(self.state.structure.metadata)

            if structure:
                # Store the updated structure
                structure.yaml_path = self.state.structure.yaml_path
                self.state.set_structure(structure)

                # Emit signals
                self.structure_loaded.emit()
                self.status_message.emit("Structure updated successfully.")

            return True

        except Exception as e:
            self._show_error("Error", f"Failed to update structure parameters: {str(e)}")
            return False

    def _set_nested_value(self, dictionary, path, value):
        """Set a value in a nested dictionary using a tuple path."""
        for key in path[:-1]:
            if key not in dictionary:
                dictionary[key] = {}
            dictionary = dictionary[key]
        dictionary[path[-1]] = value

    def handle_analysis_request(self):
        """Handle analysis request."""
        if not self.state.has_structure():
            self._show_error("Error", "No structure loaded.")
            return

        self.analysis_started.emit()
        self.status_message.emit("Running analysis...")

        try:
            # Run analysis
            results = self._run_analysis()

            if results:
                # Store results
                self.state.set_analysis_results(**results)

                # Emit signals
                self.analysis_completed.emit()
                self.status_message.emit("Analysis completed.")

        except Exception as e:
            error_details = traceback.format_exc()
            error_msg = (f"An error occurred during analysis.\n\n"
                        f"Technical Details:\n"
                        f"{type(e).__name__}: {e}\n\n"
                        f"{error_details}")

            self._show_error("Analysis Error", error_msg)
            self.analysis_failed.emit(str(e))
            self.status_message.emit("Analysis failed.")

    def _run_analysis(self) -> dict:
        """Run the analysis."""
        config = self.state.config
        structure = self.state.structure
        # structure.analysis = self.state.config.analysis
        structure.is_buckling = (config.analysis == 'buckling')

        # Get external loads
        self.status_message.emit("Processing external loads...")
        P, q = extract_external_forces(structure)

        # Get geometric and material properties
        self.status_message.emit("Processing input data...")
        coords, initial_coords, conec, properties = extract_element_data(structure)
        structure.coordinates = coords
        structure.connectivity = conec
        ref_vector = orientation_vector(structure, coords, initial_coords)

        # Generate undeformed mesh
        self.status_message.emit("Generating undeformed mesh...")
        sections = self.state.meshes.sections
        sections_indices = self.state.meshes.sections_idx
        undeformed_mesh = self.state.meshes.undeformed

        # Defining degrees of freedom and transformation matrices
        self.status_message.emit("Defining degrees of freedom...")
        free_dofs, el_dofs, num_dofs = degrees_of_freedom(structure, conec)
        T, MT = transformation_matrix(structure, coords)

        # Assemble stiffness matrix
        self.status_message.emit("Assembling stiffness matrix...")
        fq = compute_equivalent_nodal_forces(structure, properties, q, T)
        Ke, ke, fq = assemble_elastic_stiffness_matrix(structure, properties, num_dofs, el_dofs, free_dofs, fq, T)

        # Solve the system
        self.status_message.emit("Solving system...")
        Fe = assemble_global_force_vector(P, fq, el_dofs, num_dofs)
        displacements, forces, history, convergence_data = solve_structure(
            structure, config, properties, num_dofs, free_dofs, el_dofs, T, Ke, ke, Fe, fq
        )

        # Post-processing
        self.status_message.emit("Post-processing results...")
        eigvals = forces.get('autovalores', np.array([]))
        internal_forces = assemble_internal_forces(structure, forces)
        support_reactions = extract_support_reactions(structure, forces, num_dofs, free_dofs)
        deformed_coords = get_deformed_coords(structure, coords, displacements, el_dofs)
        global_deformation = get_global_displacements(structure, displacements['d'], el_dofs)

        # Generate deformed mesh
        self.status_message.emit("Generating deformed mesh...")
        malha_deformada = generate_mesh(
            structure, sections, sections_indices,
            deformed_coords, ref_vector,
            geometry_type='deformed', **{'autovalores': eigvals}
        )

        # Return results
        return {
            'structure': structure,
            'undeformed_mesh': undeformed_mesh,
            'deformed_mesh': malha_deformada,
            'convergence_data': convergence_data,
            'history': history,
            'internal_forces': internal_forces,
            'displacements': global_deformation,
            'support_reactions': support_reactions,
            'eigenvalues': eigvals,
            'coords': coords,
            'deformed_coords': deformed_coords,
            'MT': MT
        }

    def _show_error(self, title: str, message: str):
        """Show error message."""
        msg_box = QMessageBox(self.main_window)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #1e293b;
                border: 1px solid #4a5568;
                border-radius: 12px;
            }
            QMessageBox QLabel {
                color: #f8fafc;
                font-size: 14px;
            }
            QMessageBox QPushButton {
                background-color: #3b82f6;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 8px;
                padding: 10px 25px;
                min-width: 80px;
            }
            QMessageBox QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        msg_box.exec()
