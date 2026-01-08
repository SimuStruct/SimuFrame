# Third-party libraries
import math
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QTreeWidgetItem, QTableWidgetItem

# Local libraries
from .widgets import create_results_tree
from SimuFrame.utils.helpers import extract_element_data, orientation_vector
from SimuFrame.post_processing.visualization import map_unique_sections, generate_mesh


class DataTreePopulator:
    """Populates the structure's data tree."""

    @staticmethod
    def populate(tree_widget, structure):
        """Populates the tree with structure data."""
        tree_widget.clear()
        yaml_data = structure.metadata

        # Model root item
        model_item = DataTreePopulator._add_parent(
            tree_widget,
            f"Model ({yaml_data.get('info', {}).get('project_name', 'N/A')})"
        )

        # Materials
        mats_item = DataTreePopulator._add_parent(
            model_item,
            f"Materials ({len(yaml_data.get('materials', {}))})"
        )
        for mat_id in yaml_data.get('materials', {}).keys():
            DataTreePopulator._add_child(mats_item, mat_id)

        # Sections
        secs_item = DataTreePopulator._add_parent(
            model_item,
            f"Sections ({len(yaml_data.get('section_data', {}))})"
        )
        for sec_id in yaml_data.get('section_data', {}).keys():
            DataTreePopulator._add_child(secs_item, sec_id)

        # Loads
        loads_item = DataTreePopulator._add_parent(model_item, "Loads")
        DataTreePopulator._add_child(
            loads_item,
            f"Nodal Loads ({len(yaml_data.get('nodal_loads') or [])})"
        )
        DataTreePopulator._add_child(
            loads_item,
            f"Distributed Loads ({len(yaml_data.get('distributed_loads', []))})"
        )

        tree_widget.expandAll()

    @staticmethod
    def _add_parent(parent_widget, name):
        """Adds a parent item to the tree."""
        item = QTreeWidgetItem(parent_widget, [name])
        item.setExpanded(True)
        flags = item.flags()
        flags &= ~Qt.ItemFlag.ItemIsUserCheckable
        flags |= Qt.ItemFlag.ItemIsSelectable
        item.setFlags(flags)
        font = QFont()
        font.setBold(True)
        item.setFont(0, font)
        return item

    @staticmethod
    def _add_child(parent_item, name, data_key=None):
        """Adds a child item to the tree."""
        item = QTreeWidgetItem(parent_item, [name])
        if data_key:
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Unchecked)
            item.setData(0, Qt.ItemDataRole.UserRole, data_key)
        else:
            flags = item.flags()
            flags &= ~Qt.ItemFlag.ItemIsUserCheckable
            item.setFlags(flags)
        return item


class TablesPopulator:
    """Populates data tables."""

    @staticmethod
    def populate_nodes_table(table, structure):
        """Populates the nodes table."""
        yaml_data = structure.metadata
        nodes = yaml_data.get('nodes', [])

        # Mapping degrees of freedom to corresponding boundary conditions
        dof_map = {
            0: 'UX', 1: 'UY', 2: 'UZ',
            3: 'RX', 4: 'RY', 5: 'RZ'
        }

        table.setRowCount(0)
        for node_id, coords in enumerate(nodes):
            row = table.rowCount()
            table.insertRow(row)

            # Boundary condition indices
            bc_idx = structure.nodes[node_id].boundary_conditions

            # Boundary conditions labels
            labels = [dof_map[i] for i in sorted(bc_idx) if i in dof_map]

            # Add boundary conditions to node restrictions
            if len(labels) == 6:
                restrictions = 'Fixed'
            elif len(labels) == 3 and labels == ['UX', 'UY', 'UZ']:
                restrictions = 'Pinned'
            else:
                restrictions = ", ".join(labels)

            table.setItem(row, 0, QTableWidgetItem(str(node_id + 1)))
            table.setItem(row, 1, QTableWidgetItem(f"{coords[0]:.3f}"))
            table.setItem(row, 2, QTableWidgetItem(f"{coords[1]:.3f}"))
            table.setItem(row, 3, QTableWidgetItem(f"{coords[2]:.3f}"))
            table.setItem(row, 4, QTableWidgetItem(restrictions))

    @staticmethod
    def populate_elements_table(table, yaml_data):
        """Populates the elements table."""
        elements = yaml_data.get('elements', [])
        sections = yaml_data.get('section_data', {})
        nodes = yaml_data.get('nodes', [])

        table.setRowCount(0)
        for elem_id, element in enumerate(elements):
            row = table.rowCount()
            table.insertRow(row)

            connec = element.get('connec', [])
            section_id = element.get('section_id', 'N/A')
            section = sections.get(section_id, {})
            material_id = section.get('material_id', 'N/A')

            # Calculate length
            length = 0.0
            if len(connec) >= 2 and connec[0] < len(nodes) and connec[1] < len(nodes):
                node_i, node_j = nodes[connec[0]], nodes[connec[1]]
                length = math.sqrt(sum((ni - nj) ** 2 for ni, nj in zip(node_i, node_j)))

            table.setItem(row, 0, QTableWidgetItem(str(elem_id)))
            table.setItem(row, 1, QTableWidgetItem(str(connec[0]) if connec else '-'))
            table.setItem(row, 2, QTableWidgetItem(str(connec[1]) if len(connec) > 1 else '-'))
            table.setItem(row, 3, QTableWidgetItem(section_id))
            table.setItem(row, 4, QTableWidgetItem(material_id))
            table.setItem(row, 5, QTableWidgetItem(f"{length:.3f}"))

    @staticmethod
    def populate_sections_tree(tree, structure):
        """Populates the sections tree."""
        tree.clear()

        if not structure or not hasattr(structure, 'elements'):
            return

        # Collect unique sections
        unique_sections = {}
        for element in structure.elements.values():
            sec_obj = element.section
            if sec_obj and id(sec_obj) not in unique_sections:
                sec_name = getattr(sec_obj, 'name', 'N/A')
                unique_sections[id(sec_obj)] = (sec_name, sec_obj)

        # Add each section to the tree
        for sec_name, section in unique_sections.values():
            item = QTreeWidgetItem(tree, [sec_name])
            font = QFont()
            font.setBold(True)
            item.setFont(0, font)
            item.setExpanded(True)

            # Add properties
            TablesPopulator._add_section_properties(item, section)

    @staticmethod
    def _add_section_properties(parent_item, section):
        """Adds section properties."""
        properties = [
            ('h', 'Depth', 'h', 1e2, 'cm'),
            ('b', 'Width', 'b', 1e2, 'cm'),
            ('r', 'Radius', 'r', 1e2, 'cm'),
            ('ro', 'Radius', 'r', 1e2, 'cm'),
            ('t', 'Section thickness', 't', 1e2, 'cm'),
            ('tf', 'Flange thickness', 'tf', 1e2, 'cm'),
            ('tw', 'Web thickness', 'tw', 1e2, 'cm'),
            ('A', 'Sectional area', 'A', 1e4, 'cm²'),
            ('Iy', 'Area moment of inertia about y-axis', 'Iy', 1e8, 'cm⁴'),
            ('Iz', 'Area moment of inertia about z-axis', 'Iz', 1e8, 'cm⁴'),
            ('It', 'Torsional constant (t)', 'It', 1e8, 'cm⁴')
        ]

        for attr, desc, sym, scale, unit in properties:
            if hasattr(section, attr):
                value = getattr(section, attr)
                item = QTreeWidgetItem(parent_item, [f"  {desc}", sym, f"{scale * value:.2f}", unit])
                item.setTextAlignment(2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)


class ResultsTreePopulator:
    """Populates the results tree."""

    @staticmethod
    def populate(tree_widget, main_window, is_buckling):
        """Recreates the results tree."""
        tree_widget.clear()

        # Create new tree using helper function
        new_tree = create_results_tree(main_window, is_buckling)

        # Copy items to existing tree
        iterator = new_tree.invisibleRootItem()
        for i in range(iterator.childCount()):
            item = iterator.child(i).clone()
            tree_widget.addTopLevelItem(item)

        return tree_widget


class MeshLoader:
    """Loads and creates structure meshes."""

    @staticmethod
    def load_undeformed_mesh(structure, config):
        """Loads the undeformed mesh."""
        structure.is_buckling = (config.analysis == 'buckling')

        try:
            coords, initial_coords, *_ = extract_element_data(structure)
            ref_vector = orientation_vector(structure, coords, initial_coords)
            secoes, secoes_indices = map_unique_sections(structure)

            malha_indeformada = generate_mesh(
                structure, secoes, secoes_indices,
                initial_coords, ref_vector,
                geometry_type='undeformed'
            )

            return malha_indeformada, secoes, secoes_indices
        except Exception as e:
            print(f"Error loading undeformed mesh: {e}")
            return None, None, None
