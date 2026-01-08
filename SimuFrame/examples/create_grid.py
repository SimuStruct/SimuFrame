import numpy as np
from dataclasses import dataclass
from typing import Tuple

class GrillageMeshGenerator:
    def __init__(self, length_x: float, width_y: float, pressure: float,
                 div_x: int, div_y: int):
        """
        Args:
            length_x: Total size in X direction.
            width_y: Total size in Y direction.
            pressure: Surface pressure (Pa or N/m^2).
            div_x: Number of subdivisions along X (panels).
            div_y: Number of subdivisions along Y (panels).
        """
        self.Lx = length_x
        self.Ly = width_y
        self.P = pressure
        self.nx = div_x
        self.ny = div_y

    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates the mesh and loads.

        Returns:
            nodes: np.ndarray shape (num_nodes, 3) -> [x, y, z]
            connectivity: np.ndarray shape (num_elems, 2) -> [start_node, end_node]
            loads: np.ndarray shape (num_elems,) -> [q] (distributed load N/m)
        """

        # Grid steps
        dx = self.Lx / self.nx
        dy = self.Ly / self.ny

        # Number of nodes in each direction
        num_nodes_x = self.nx + 1
        num_nodes_y = self.ny + 1

        # --- 1. Generate Nodes ---
        # Total nodes = (nx+1) * (ny+1)
        # We assume Z = 0.0 for a flat grillage

        nodes_list = []

        # Loop order: Y then X (Row by Row)
        # Node ID = j * num_nodes_x + i
        for j in range(num_nodes_y):
            for i in range(num_nodes_x):
                x = i * dx
                y = j * dy
                z = 0.0
                nodes_list.append([x, y, z])

        nodes = np.array(nodes_list, dtype=np.float64)

        # --- 2. Generate Elements (Connectivity) & Loads ---
        elements_list = []
        loads_list = []

        # Helper function to get Node ID from (i, j) grid indices
        def get_node_id(i, j):
            return j * num_nodes_x + i

        # A) Horizontal Elements (Along X)
        # We iterate through each "line" of Y (j) and create segments along X (i)
        for j in range(num_nodes_y):

            # Calculate Tributary Width for this horizontal line
            if j == 0 or j == self.ny:
                trib_width = dy / 2.0  # Edge
            else:
                trib_width = dy        # Internal

            q_horizontal = self.P * trib_width

            for i in range(self.nx):
                n1 = get_node_id(i, j)
                n2 = get_node_id(i + 1, j)

                elements_list.append([n1, n2])
                loads_list.append(q_horizontal)

        # B) Vertical Elements (Along Y)
        # We iterate through each "line" of X (i) and create segments along Y (j)
        for i in range(num_nodes_x):

            # Calculate Tributary Width for this vertical line
            if i == 0 or i == self.nx:
                trib_width = dx / 2.0  # Edge
            else:
                trib_width = dx        # Internal

            q_vertical = self.P * trib_width

            for j in range(self.ny):
                n1 = get_node_id(i, j)
                n2 = get_node_id(i, j + 1)

                elements_list.append([n1, n2])
                loads_list.append(q_vertical)

        connectivity = np.array(elements_list, dtype=np.int64)
        loads = np.array(loads_list, dtype=np.float64)

        return nodes, connectivity, loads

# --- Example Usage ---
if __name__ == "__main__":
    # Parameters
    L = 6.0
    W = 6.0
    Pressure = 3.75 # kPa
    div_x = 6  # panels in X
    div_y = 6  # panels in Y

    # Generate
    generator = GrillageMeshGenerator(L, W, Pressure, div_x, div_y)
    nodes, conn, loads = generator.generate()

    print(f"Total Nodes: {len(nodes)}")
    print(f"Total Elements: {len(conn)}")

    print(nodes)
    xi_indices = np.where(nodes[:, 0] == 0)
    yi_indices = np.where(nodes[:, 1] == 0)
    xf_indices = np.where(nodes[:, 0] == L)
    yf_indices = np.where(nodes[:, 1] == W)

    print(xi_indices)
    print(yi_indices)
    print(xf_indices)
    print(yf_indices)

    print(conn)
    print(loads)

    load_50 = np.where(loads == Pressure / 2)
    load_100 = np.where(loads == Pressure)

    print(load_50)
    print(load_100)
