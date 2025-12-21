# Built-in libraries
import traceback

# Third-party libraries
import numpy as np
from ruamel.yaml import YAML
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox, QFileDialog

# Internal libraries
from SimuFrame.core.model import Structure
from SimuFrame.core.assembly import (
    extract_external_forces, compute_equivalent_nodal_forces, assemble_global_force_vector, assemble_internal_forces,
    degrees_of_freedom, extract_support_reactions, assemble_elastic_stiffness_matrix
)
from SimuFrame.core.solver import calcular_estrutura
from SimuFrame.post_processing.visualization import generate_mesh, criar_secoes_base, plot_structure
from SimuFrame.utils.helpers import (
    extract_element_data, orientation_vector, transformation_matrix,
    get_deformed_coords, deslocamentos_globais
)


class SimuController(QObject):
    """
    Controlador da interface gráfica da SimuFrame.
    Responsável pela comunicação entre a UI e o core.
    """
    # Sinais para comunicação com a UI
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
        """Retorna um handler YAML configurado para preservar comentários e indentação."""
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)
        return yaml

    def open_structure_creator(self):
        """Abre a janela de criação de estruturas."""
        # Importação local para evitar ciclo de imports
        from SimuFrame.GUI.structure_creator import StructureCreatorDialog

        dialog = StructureCreatorDialog(self)
        dialog.show()

    def update_structure_preview(self, project_data: dict, plotter):
        """
        Gera uma estrutura temporária a partir do dicionário de dados
        e plota no visualizador fornecido (preview).
        """
        try:
            # 1. Criar estrutura temporária (sem afetar o self.state)
            temp_structure = self._create_structure_from_yaml(project_data)

            if not temp_structure:
                plotter.clear_actors()
                return

            # Hack para visualização: definir original_nodes se não existir
            if not hasattr(temp_structure, 'original_nodes'):
                temp_structure.original_nodes = temp_structure.nodes

            # 2. Gerar malha de visualização
            secoes_unicas, secoes_indices = criar_secoes_base(temp_structure)

            coords, initial_coords, _, _ = extract_element_data(temp_structure)

            # Se não houver elements válidos ainda, aborta plotagem complexa
            if len(initial_coords) == 0:
                plotter.clear_actors()
                return

            ref_vector = orientation_vector(temp_structure, coords, initial_coords)

            malha = generate_mesh(
                temp_structure, secoes_unicas, secoes_indices,
                initial_coords, ref_vector, geometry='undeformed'
            )

            # 3. Plotar
            plotter.clear_actors()
            plotter.add_axes()

            if malha and malha['section']:
                plot_structure(
                    plotter,
                    temp_structure,
                    malha,
                    transparencia=1.0,
                    plotar_secao=True,
                    plotar_cargas=False,  # Pode ser ativado se desejar visualizar cargas no preview
                    plotar_nos=True
                )
                plotter.reset_camera()

        except Exception as e:
            # Erros de plotagem no preview não devem crashar a app, apenas logar
            print(f"Erro no preview da estrutura: {e}")

    def save_new_project(self, project_data: dict):
        """Salva um novo projeto criado via diálogo."""
        file_path, _ = QFileDialog.getSaveFileName(
            self.main_window, "Salvar Novo Projeto", "SimuFrame/elements", "YAML Files (*.yaml)"
        )

        if file_path:
            try:
                yaml = self._get_yaml_handler()
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(project_data, f)

                QMessageBox.information(self.main_window, "Sucesso", "Projeto salvo com sucesso!")

                # Opcional: Carregar automaticamente o projeto salvo
                # self.load_project_from_path(file_path)

            except Exception as e:
                self._show_error("Erro ao Salvar", f"Não foi possível salvar o arquivo:\n{e}")

    def load_project_from_yaml(self):
        """Abre diálogo e carrega projeto YAML."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Selecionar arquivo YAML",
            "SimuFrame/elements",
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )

        if not file_path:
            return

        try:
            self.status_message.emit(f"Carregando {file_path}...")

            # Carregar YAML
            yaml_data = self._load_yaml_file(file_path)

            # Criar estrutura
            structure = self._create_structure_from_yaml(yaml_data)

            if structure:
                # Armazenar no estado
                structure.yaml_path = file_path
                self.state.set_structure(structure)

                # Emitir sinal
                self.structure_loaded.emit()
                self.status_message.emit("Arquivo carregado com sucesso.")

        except Exception as e:
            error_msg = f"Não foi possível abrir o arquivo:\n\n{str(e)}"
            self._show_error("Falha ao Carregar Arquivo", error_msg)
            self.status_message.emit("Falha ao carregar arquivo.")

    def _load_yaml_file(self, filepath: str) -> dict:
        """Carrega arquivo YAML."""
        yaml = self._get_yaml_handler()
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return yaml.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        except Exception as e:
            raise ValueError(f"Erro ao decodificar o YAML: {e}")

    def save_yaml_file(self):
        """Salva o estado atual do yaml_data no arquivo original."""
        if not self.state.has_structure or not self.state.estrutura.metadata:
            self._show_error("Erro", "Nenhum projeto carregado.")
            return False

        # Obter o handler do yaml
        yaml = self._get_yaml_handler()

        try:
            # Salvar yaml_data no arquivo original
            with open(self.state.estrutura.yaml_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.state.estrutura.metadata, file)

            self.status_message.emit("Arquivo YAML salvo com sucesso.")
            return True

        except Exception as e:
            self._show_error("Erro", f"Falha ao salvar arquivo: {e}")
            return False

    def _create_structure_from_yaml(self, yaml_data: dict) -> Structure | None:
        """Cria e retorna um objeto Structure a partir dos dados do YAML."""
        try:
            # Extrair dados de análise
            analysis = yaml_data.get('analysis', {})
            analysis_type = analysis.get('analysis_type', 'linear')
            structural_model = analysis.get('element_type', 'beam')
            subdivisions = analysis.get('number_of_subdivisions', 1)

            # Extrair dados estruturais
            materials = yaml_data.get('materials', {})
            sections = yaml_data.get('sections', {})
            nodes = yaml_data.get('nodes', [])
            elements = yaml_data.get('elements', [])
            supports = yaml_data.get('supports', {})
            nodal_loads = yaml_data.get('nodal_loads', [])
            distributed_loads = yaml_data.get('distributed_loads', [])

            # Processar elements
            connec = np.zeros((len(elements), 2), dtype=int)
            coords = np.array(nodes, dtype=float)
            releases = {}
            section_properties = {}

            # Pré-carregar materiais
            loaded_materials = {}
            for mat_id, material in materials.items():
                loaded_materials[mat_id] = {
                    "E": float(material.get("E", 1.0)),
                    "nu": float(material.get("nu", 0.0))
                }

            for elem_id, element in enumerate(elements):
                # Define element connectivity and member_releases
                connec[elem_id] = element.get('connec')
                elem_release = [[], []] if structural_model == 'beam' else [[3, 4, 5], [3, 4, 5]]
                releases[elem_id] = element.get('hinges', elem_release)

                # Check if the section exists in the database
                section_id = element.get('section_id')
                if not section_id or section_id not in sections:
                    raise ValueError(f"Section '{section_id}' for element {elem_id} not found in section_data.")

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
            self.state.config.tipo = analysis_type
            self.state.config.subdivisions = subdivisions

            return structure

        except Exception as e:
            error_msg = f"Não foi possível criar a estrutura:\n\n{str(e)}"
            self._show_error("Erro ao Processar YAML", error_msg)
            return None

    def update_analysis_parameters(self, parameters: dict):
        """Atualiza os parâmetros de análise, caso o usuário solicite."""
        if not self.state.has_structure():
            self._show_error("Erro", "Nenhuma estrutura foi carregada.")
            return False

        try:
            # Extrair novos parâmetros
            analysis_type = parameters.get('analysis_type', 'linear')
            subdivisions = parameters.get('number_of_subdivisions', 1)

            # Normalizar nome da análise
            analysis_map = {
                'linear': 'linear',
                'nao-linear': 'nonlinear', 'não-linear': 'nonlinear', 'não linear': 'nonlinear',
                'nao linear': 'nonlinear', 'naolinear': 'nonlinear', 'nonlinear': 'nonlinear',
                'flambagem': 'buckling', 'buckling': 'buckling', 'instabilidade': 'buckling'
            }

            # Converter para minúsculas e remover espaços extras
            analysis_type_clean = analysis_type.lower().strip()
            analysis_normalized = analysis_map.get(analysis_type_clean, 'linear')

            # Mapear para exibição na interface
            ui_analysis_type = {
                'linear': 'Linear', 'nonlinear': 'Não Linear', 'buckling': 'Flambagem'
            }
            ui_display_type = ui_analysis_type.get(analysis_normalized, 'Linear')

            # Atualizar a instância da classe Structure
            self.state.estrutura.metadata['analysis']['analysis_type'] = analysis_normalized
            self.state.estrutura.metadata['analysis']['number_of_subdivisions'] = int(subdivisions)

            # Atualizar configurações visuais da interface
            self.state.config.tipo = ui_display_type
            self.state.config.subdivisions = int(subdivisions)

            # Recriar a estrutura para aplicar as alterações
            structure = self._create_structure_from_yaml(self.state.estrutura.metadata)

            if structure:
                # Armazenar no estado
                structure.yaml_path = self.state.estrutura.yaml_path
                self.state.set_structure(structure)

                # Emitir sinal
                self.structure_loaded.emit()
                self.status_message.emit("Estrutura atualizada com sucesso.")

            return True

        except Exception as e:
            self._show_error("Erro", f"Erro ao atualizar parâmetros de análise: {str(e)}")
            return False

    def handle_analysis_request(self):
        """Executa a análise estrutural."""
        if not self.state.has_structure():
            self._show_error("Erro", "Nenhuma estrutura foi carregada.")
            return
        
        self.analysis_started.emit()
        self.status_message.emit("Iniciando análise...")

        try:
            # Executar análise
            results = self._run_analysis()
            
            if results:
                # Armazenar resultados no estado
                self.state.set_analysis_results(**results)
                
                # Notificar conclusão
                self.analysis_completed.emit()
                self.status_message.emit("Análise concluída.")
            
        except Exception as e:
            error_details = traceback.format_exc()
            error_msg = (f"Ocorreu um erro inesperado.\n\n"
                        f"Detalhes Técnicos:\n"
                        f"{type(e).__name__}: {e}\n\n"
                        f"{error_details}")
            
            self._show_error("Erro Durante a Análise", error_msg)
            self.analysis_failed.emit(str(e))
            self.status_message.emit("Falha na análise")
    
    def _run_analysis(self) -> dict:
        """Run the analysis."""
        structure = self.state.estrutura
        structure.analysis = self.state.config.tipo
        structure.is_buckling = (structure.analysis == 'buckling')

        # Get external loads
        self.status_message.emit("Processing external loads...")
        P, q = extract_external_forces(structure)

        # Get geometric and material properties
        self.status_message.emit("Processing input data...")
        coords, initial_coords, conec, propriedades = extract_element_data(structure)
        structure.coordinates = coords
        structure.connectivity = conec
        ref_vector = orientation_vector(structure, coords, initial_coords)

        # Generate undeformed mesh
        self.status_message.emit("Generating undeformed mesh...")
        secoes = self.state.malhas.secoes
        secoes_indices = self.state.malhas.secoes_indices
        malha_indeformada = self.state.malhas.indeformada

        # Defining degrees of freedom and transformation matrix
        self.status_message.emit("Defining degrees of freedom...")
        GLL, GLe, numDOF = degrees_of_freedom(structure, conec)
        T, MT = transformation_matrix(structure, coords)

        # Assemble stiffness matrix
        self.status_message.emit("Assembling stiffness matrix...")
        fq = compute_equivalent_nodal_forces(structure, propriedades, q, T)
        Ke, ke, fq = assemble_elastic_stiffness_matrix(structure, propriedades, numDOF, GLe, GLL, fq, T)

        # Solve the system
        self.status_message.emit("Solving system...")
        Fe = assemble_global_force_vector(P, fq, GLe, numDOF)
        deslocamentos, esforcos, f_vs_d, convergence_data = calcular_estrutura(
            structure, propriedades, coords, numDOF, GLL, GLe, T, Ke, ke, Fe, fq
        )

        # Post-processing
        self.status_message.emit("Post-processing results...")
        eigvals = esforcos.get('autovalores', np.array([]))
        internal_forces = assemble_internal_forces(structure, esforcos)
        support_reactions = extract_support_reactions(structure, esforcos, numDOF, GLL)
        deformed_coords = get_deformed_coords(structure, coords, deslocamentos, GLe)
        global_deformation = deslocamentos_globais(structure, deslocamentos['d'], GLe)

        # Generate deformed mesh
        self.status_message.emit("Generating deformed mesh...")
        malha_deformada = generate_mesh(
            structure, secoes, secoes_indices,
            deformed_coords, ref_vector, 
            geometry='deformed', **{'autovalores': eigvals}
        )

        # Return results
        return {
            'estrutura': structure,
            'malha_indeformada': malha_indeformada,
            'malha_deformada': malha_deformada,
            'convergence_data': convergence_data,
            'f_vs_d': f_vs_d,
            'esforcos_int': internal_forces,
            'deslocamentos': global_deformation,
            'reacoes_nos_apoios': support_reactions,
            'autovalores': eigvals,
            'coords': coords,
            'coords_deformadas': deformed_coords,
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