from matplotlib.typing import ColorType
import numpy as np
from typing import Tuple
import numpy.typing as npt


def create_dome(
    base_radius: float,
    height: float,
    num_meridians: int = 16,
    num_parallels: int = 8,
    include_apex: bool = True
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], dict]:
    """
    Create a geodesic dome structure with arbitrary dimensions.

    Parameters
    ----------
    base_radius : float
        Radius of the dome base (m).
    height : float
        Height of the dome apex from the base (m).
    num_meridians : int, optional
        Number of vertical divisions (longitude lines). Default: 16.
    num_parallels : int, optional
        Number of horizontal divisions (latitude lines). Default: 8.
    include_apex : bool, optional
        Whether to include a single apex node at the top. Default: True.

    Returns
    -------
    coordinates : ndarray
        Node coordinates array (n_nodes x 3) with columns [x, y, z].
    connectivity : ndarray
        Element connectivity array (n_elements x 2) with columns [node_i, node_j].
    metadata : dict
        Dictionary containing dome information:
        - 'num_nodes': Total number of nodes
        - 'num_elements': Total number of elements
        - 'base_nodes': Indices of nodes at the base
        - 'apex_node': Index of apex node (if included)
        - 'sphere_radius': Radius of the sphere that dome is part of

    Notes
    -----
    The dome is created as a section of a sphere. The sphere radius is calculated
    based on the base radius and height using the spherical cap formula:

    R = (base_radius² + height²) / (2 * height)
    """
    # Calculate sphere radius from base radius and height
    # Using spherical cap formula: R = (r² + h²) / (2h)
    sphere_radius = (base_radius**2 + height**2) / (2 * height)

    # Calculate angular parameters
    # Angle from sphere center to base edge
    theta_max = np.arcsin(base_radius / sphere_radius)

    # Vertical position of sphere center relative to base
    center_z = sphere_radius - height

    # Generate nodes
    nodes = []
    node_id = 0
    node_map = {}  # Maps (parallel, meridian) to node_id

    # Generate parallel circles (horizontal layers)
    for i in range(num_parallels + 1):
        # Angle from vertical axis (0 at top, theta_max at base)
        theta = (i / num_parallels) * theta_max

        # Radius of this parallel circle
        r = sphere_radius * np.sin(theta)

        # Height of this parallel
        z = sphere_radius * np.cos(theta) - center_z

        # Special case: apex node
        if i == 0 and include_apex:
            nodes.append([0.0, 0.0, height])
            node_map[(i, 0)] = node_id
            node_id += 1
            continue

        # Generate nodes around this parallel
        for j in range(num_meridians):
            # Angle around vertical axis
            phi = (j / num_meridians) * 2 * np.pi

            # Cartesian coordinates
            x = r * np.cos(phi)
            y = r * np.sin(phi)

            nodes.append([x, y, z])
            node_map[(i, j)] = node_id
            node_id += 1

    coordinates = np.array(nodes)

    # Generate element connectivity
    elements = []

    # Connect apex to first parallel (if apex exists)
    if include_apex:
        apex_id = node_map[(0, 0)]
        for j in range(num_meridians):
            j_next = (j + 1) % num_meridians
            node_j = node_map[(1, j)]
            node_j_next = node_map[(1, j_next)]

            # Meridional elements from apex
            elements.append([apex_id, node_j])

            # Circumferential elements on first parallel
            elements.append([node_j, node_j_next])

    # Connect remaining parallels
    start_parallel = 1 if include_apex else 0

    for i in range(start_parallel, num_parallels):
        for j in range(num_meridians):
            j_next = (j + 1) % num_meridians

            # Current and next parallel
            node_curr = node_map[(i, j)]
            node_next = node_map[(i + 1, j)]
            node_curr_next = node_map[(i, j_next)]
            node_next_next = node_map[(i + 1, j_next)]

            # Meridional element (vertical)
            elements.append([node_curr, node_next])

            # Circumferential element (horizontal)
            if i < num_parallels:  # Not needed on last parallel (already done above)
                elements.append([node_next, node_next_next])

            # Diagonal element for triangulation (optional for stiffness)
            # elements.append([node_curr, node_next_next])

    connectivity = np.array(elements)

    # Identify base nodes (last parallel)
    base_nodes = [node_map[(num_parallels, j)] for j in range(num_meridians)]

    # Create metadata
    metadata = {
        'num_nodes': len(coordinates),
        'num_elements': len(connectivity),
        'base_nodes': base_nodes,
        'apex_node': node_map[(0, 0)] if include_apex else None,
        'sphere_radius': sphere_radius,
        'base_radius': base_radius,
        'height': height,
        'num_meridians': num_meridians,
        'num_parallels': num_parallels,
    }

    return coordinates, connectivity, metadata


def visualize_dome(
    coordinates: npt.NDArray[np.float64],
    connectivity: npt.NDArray[np.int_],
    show_node_labels: bool = False
):
    """
    Visualize the dome structure using matplotlib.

    Parameters
    ----------
    coordinates : ndarray
        Node coordinates from create_dome().
    connectivity : ndarray
        Element connectivity from create_dome().
    show_node_labels : bool, optional
        Whether to show node numbers. Default: False.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is required for visualization.")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot elements
    for elem in connectivity:
        node_i, node_j = elem
        xs = [coordinates[node_i, 0], coordinates[node_j, 0]]
        ys = [coordinates[node_i, 1], coordinates[node_j, 1]]
        zs = [coordinates[node_i, 2], coordinates[node_j, 2]]
        ax.plot(xs, ys, zs, 'b-', linewidth=1.5, alpha=0.7)

    # Plot nodes
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        c='red',
        s=50,
        alpha=0.8
    )

    # Optional: show node labels
    if show_node_labels:
        for i, coord in enumerate(coordinates):
            ax.text(coord[0], coord[1], coord[2], f'{i}', fontsize=8)

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Dome Structure')

    # Equal aspect ratio
    max_range = np.array([
        coordinates[:, 0].max() - coordinates[:, 0].min(),
        coordinates[:, 1].max() - coordinates[:, 1].min(),
        coordinates[:, 2].max() - coordinates[:, 2].min()
    ]).max() / 2.0

    mid_x = (coordinates[:, 0].max() + coordinates[:, 0].min()) * 0.5
    mid_y = (coordinates[:, 1].max() + coordinates[:, 1].min()) * 0.5
    mid_z = (coordinates[:, 2].max() + coordinates[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create dome: 10m base radius, 5m height
    coords, connect, info = create_dome(
        base_radius=10.0,
        height=5.0,
        num_meridians=12,
        num_parallels=6,
        include_apex=True
    )

    print("Dome Information:")
    print(f"  Nodes: {info['num_nodes']}")
    print(f"  Elements: {info['num_elements']}")
    print(f"  Sphere radius: {info['sphere_radius']:.3f} m")
    print(f"  Base nodes: {info['base_nodes']}")
    print(f"  Apex node: {info['apex_node']}")

    # print("\nFirst 5 node coordinates:")
    # print(coords[:5])

    # print("\nFirst 5 element connectivities:")
    # print(connect[:5])
    print(coords)
    print(connect)

    # Visualize
    visualize_dome(coords, connect, show_node_labels=False)
