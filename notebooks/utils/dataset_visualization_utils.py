import json
import numpy as np
import copy
import os
import random
import glob

from shapely.geometry import Point, Polygon

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import matplotlib.pyplot as plt
from PIL import Image


def create_adj_mat(points: list[list[float]]):
    # Extract unique points
    unique_points = list({tuple(point) for sublist in points for point in sublist})
    unique_points.sort()  # Optional: Sort for consistent ordering

    # Create a dictionary to map points to indices
    point_index = {point: index for index, point in enumerate(unique_points)}

    # Step 2: Create adjacency matrix
    num_points = len(unique_points)
    adj_matrix = np.zeros((num_points, num_points))

    # Fill the adjacency matrix
    for sublist in points:
        for i in range(len(sublist) - 1):
            point1 = tuple(sublist[i])
            point2 = tuple(sublist[i + 1])
            idx1 = point_index[point1]
            idx2 = point_index[point2]
            adj_matrix[idx1, idx2] = 1
            adj_matrix[idx2, idx1] = 1  # Assuming undirected graph
    points = np.array(unique_points)

    return points, adj_matrix


def _get_footprint_polygon(directory_path, sample_file_path):
    footprint_dir = directory_path.replace("final_building_outdir", "deeplayout_footprint_outdir")

    # Substring you're looking for in the filename
    substring_all = f"{sample_file_path.split('/')[-2]}_0_ALL_original.json"
    substring_bottom = f"{sample_file_path.split('/')[-2]}_0_BOTTOM_original.json"

    # Use glob to find all files and filter for the substring
    matches_all = [f for f in glob.glob(os.path.join(footprint_dir, "*")) if substring_all in os.path.basename(f)]
    matches_bottom = [f for f in glob.glob(os.path.join(footprint_dir, "*")) if substring_bottom in os.path.basename(f)]

    if len(matches_all) > len(matches_bottom):
        # Open and load the JSON file
        with open(matches_all[0], 'r') as f:
            data = json.load(f)
        return Polygon(data['geometry'])

    else:
        with open(matches_bottom[0], 'r') as f:
            data = json.load(f)
        return Polygon(data['geometry'])


def plot_3d_graph(points_list, adj_matrix_list, title=None, save_plot=False, camera_eye_dict=None, color_scheme=None):
    """
    Plot 3D graph visualization with optional color scheme for the first graph.

    Parameters:
    -----------
    points_list : list
        List of point arrays for each graph to plot
    adj_matrix_list : list
        List of adjacency matrices for each graph
    title : str, optional
        Title for the plot
    save_plot : bool, optional
        Whether to save the plot
    camera_eye_dict : dict, optional
        Camera positioning parameters
    color_scheme : str, optional
        Name of the colorscale to use for the first graph (e.g., 'viridis')
        If None, the first graph will be colored orange

    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Default colors for graphs (if not using color scheme)
    default_colors = ["orange", "blue", "red"]

    # Create a figure
    fig = go.Figure()

    for idx in range(len(points_list)):
        # Ensure that points_list[idx] is a NumPy array
        points_array = np.array(points_list[idx])

        # Ensure that adj_matrix_list[idx] is a NumPy array
        adj_matrix_array = np.array(adj_matrix_list[idx])

        # Extract coordinates
        x = points_array[:, 0]
        y = points_array[:, 1]
        z = points_array[:, 2]

        # Create traces for edges
        edge_traces = []
        for i in range(adj_matrix_array.shape[0]):
            for j in range(i + 1, adj_matrix_array.shape[1]):
                if adj_matrix_array[i, j] > 0:
                    edge_trace = go.Scatter3d(
                        x=[x[i], x[j]],
                        y=[y[i], y[j]],
                        z=[z[i], z[j]],
                        mode='lines',
                        line=dict(width=3.5, color="grey"),
                        showlegend=False
                    )
                    edge_traces.append(edge_trace)

        # Apply color scheme only to the first graph
        if idx == 0 and color_scheme is not None:
            # Use the specified color scheme for the first graph
            import plotly.express as px

            # Get the colorscale
            if color_scheme == 'viridis':
                colorscale = px.colors.sequential.Viridis
            elif color_scheme == 'plasma':
                colorscale = px.colors.sequential.Plasma
            elif color_scheme == 'inferno':
                colorscale = px.colors.sequential.Inferno
            elif color_scheme == 'magma':
                colorscale = px.colors.sequential.Magma
            elif color_scheme == 'cividis':
                colorscale = px.colors.sequential.Cividis
            elif hasattr(px.colors.sequential, color_scheme):
                colorscale = getattr(px.colors.sequential, color_scheme)
            else:
                # Default to viridis if the specified scheme is not found
                colorscale = px.colors.sequential.Viridis

            # Create normalized values for coloring the points (0 to 1)
            # Using the z-coordinate for coloring
            z_min, z_max = min(z), max(z)
            if z_max > z_min:
                norm_values = [(val - z_min) / (z_max - z_min) for val in z]
            else:
                norm_values = [0.5 for _ in z]  # Use constant if all values are the same

            # Map normalized values to colors in the colorscale
            colors_list = []
            for norm_val in norm_values:
                index = min(int(norm_val * (len(colorscale) - 1)), len(colorscale) - 1)
                colors_list.append(colorscale[index])

            # Create trace for nodes with the color scheme
            node_trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=3.5,
                    color=z,  # Use z values for color mapping
                    colorscale=color_scheme,
                    opacity=1
                ),
                text=[f'Point {i}' for i in range(len(points_array))],
                name=f'Graph {idx + 1}'
            )
        else:
            # Use default colors for other graphs
            node_trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(size=3.5, color=default_colors[idx % len(default_colors)], opacity=1),
                text=[f'Point {i}' for i in range(len(points_array))],
                name=f'Graph {idx + 1}'
            )

        # Add traces to figure
        fig.add_traces(edge_traces)
        fig.add_trace(node_trace)

    if title is None:
        title = '3D Graph Visualization'

    # Update layout to hide grid, zero lines, and axes
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title='',  # Set title to empty string
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                showbackground=False,
            ),
            yaxis=dict(
                title='',  # Set title to empty string
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                showbackground=False,
            ),
            zaxis=dict(
                title='',  # Set title to empty string
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                showbackground=False,
            ),
            bgcolor='white'  # Set background color to white
        ),
        paper_bgcolor='white',  # Set the paper background to white
        plot_bgcolor='white'  # Set the plot background to white
    )

    if camera_eye_dict is not None:
        fig.update_layout(scene_camera=camera_eye_dict)

    if save_plot:
        fig.write_html("my_plot.html", auto_open=True)
        # Save the plot as an image
        pio.write_image(fig, f'/users/kevin/desktop/{title}.png', scale=6, width=1080, height=1080)

    # Show the plot
    fig.show()

    # Return the figure object
    return fig


def plot_room_volume_mesh(points, adj_matrix, point_groups=None, title=None, save_plot=False, camera_eye_dict=None):
    # Ensure that points is a NumPy array
    points_array = np.array(points)

    # Ensure that adj_matrix is a NumPy array
    adj_matrix_array = np.array(adj_matrix)

    # Extract coordinates
    x = points_array[:, 0]
    y = points_array[:, 1]
    z = points_array[:, 2]

    # Create traces for edges
    edge_traces = []
    for i in range(adj_matrix_array.shape[0]):
        for j in range(i + 1, adj_matrix_array.shape[1]):
            if adj_matrix_array[i, j] > 0:
                edge_trace = go.Scatter3d(
                    x=[x[i], x[j]],
                    y=[y[i], y[j]],
                    z=[z[i], z[j]],
                    mode='lines',
                    line=dict(width=3.5, color='gray'),
                    showlegend=False
                )
                edge_traces.append(edge_trace)

    # Create trace for nodes
    node_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(size=3.5, color='orange', opacity=1),
        text=[f'Point {i}' for i in range(len(points))],
        showlegend=False
    )

    # New code - Added color mapping
    color_map = {
        '1': 'steelblue',
        '2': 'midnightblue',  # dark blue
        '3': 'darkgreen',  # deep green
        '4': 'indigo',  # rich purple-blue
        '5': 'orange',  # unchanged
        '6': 'cyan',  # unchanged
        '7': 'gold',  # unchanged
        '8': 'darkred',  # earthy dark brown
        '9': 'brown',  # unchanged
        '10': 'darkmagenta',  # bold dark pink
        '11': 'magenta',  # unchanged
        '12': 'lime',  # unchanged
        '13': 'teal'  # unchanged
    }

    # Add traces for the point groups (volumes)
    volume_traces = []
    if point_groups is not None:
        for group_id, indices_list in point_groups.items():
            # Get the color for this room type from our mapping
            color = color_map.get(group_id, 'lightgray')  # Default to light gray if not in map

            for indices in indices_list:
                volume_trace = go.Mesh3d(
                    x=points_array[indices, 0],
                    y=points_array[indices, 1],
                    z=points_array[indices, 2],
                    color=color,
                    opacity=0.8,
                    alphahull=0,
                    name=f'Room Type {group_id}'
                )
                volume_traces.append(volume_trace)

    # Create figure and add traces
    fig = go.Figure(data=edge_traces + [node_trace] + volume_traces)

    if title is None:
        title = '3D Graph Visualization'

    # Update layout to hide grid, zero lines, and axes
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title='',  # Set title to empty string
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                showbackground=False,
            ),
            yaxis=dict(
                title='',  # Set title to empty string
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                showbackground=False,
            ),
            zaxis=dict(
                title='',  # Set title to empty string
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                showbackground=False,
            ),
            bgcolor='white'  # Set background color to white
        ),
        paper_bgcolor='white',  # Set the paper background to white
        plot_bgcolor='white'  # Set the plot background to white
    )

    if camera_eye_dict is not None:
        fig.update_layout(scene_camera=camera_eye_dict)

    if save_plot:
        fig.write_html("my_plot.html", auto_open=True)
        # Save the plot as an image
        pio.write_image(fig, f'/users/kevin/desktop/{title}.png', scale=6, width=1080, height=1080)

    # Show the plot
    fig.show()


def plot_room_volume_point_cloud(points, adj_matrix, point_groups=None, building_footprint=None,
                                 points_per_cubic_meter=10, title=None, save_plot=False, camera_eye_dict=None):
    """
    Plot 3D building visualization with room volumes represented by color-coded points
    that don't extend beyond the building footprint. Points are distributed randomly but uniformly
    (constant number of points per cubic meter) regardless of room size.

    Parameters:
    -----------
    points : list or numpy.ndarray
        List of 3D points [x, y, z] representing the building structure
    adj_matrix : list or numpy.ndarray
        Adjacency matrix describing connections between points
    point_groups : dict, optional
        Dictionary mapping room type IDs to lists of point indices for each room
    building_footprint : list, optional
        List of [x, y] coordinates defining the building footprint polygon
    points_per_cubic_meter : float, optional
        Number of points per cubic meter for room volume visualization
    title : str, optional
        Title for the plot
    save_plot : bool, optional
        Whether to save the plot
    camera_eye_dict : dict, optional
        Camera positioning parameters

    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Ensure that inputs are NumPy arrays
    points_array = np.array(points)
    adj_matrix_array = np.array(adj_matrix)

    # Extract coordinates
    x = points_array[:, 0]
    y = points_array[:, 1]
    z = points_array[:, 2]

    # Create building footprint polygon if provided
    footprint_polygon = None
    if building_footprint is not None:
        footprint_polygon = Polygon(building_footprint)
        # Apply a minimal buffer to handle points that are exactly on the boundary
        footprint_polygon = footprint_polygon.buffer(0.001)

    # Create traces for edges
    edge_traces = []
    for i in range(adj_matrix_array.shape[0]):
        for j in range(i + 1, adj_matrix_array.shape[1]):
            if adj_matrix_array[i, j] > 0:
                edge_trace = go.Scatter3d(
                    x=[x[i], x[j]],
                    y=[y[i], y[j]],
                    z=[z[i], z[j]],
                    mode='lines',
                    line=dict(width=3.5, color='gray'),
                    showlegend=False
                )
                edge_traces.append(edge_trace)

    # Create trace for structural nodes
    node_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(size=3.5, color='orange', opacity=1),
        text=[f'Point {i}' for i in range(len(points))],
        showlegend=False
    )

    # Color mapping for different room types
    color_map = {
        '1': 'steelblue',
        '2': 'midnightblue',  # dark blue
        '3': 'darkgreen',  # deep green
        '4': 'indigo',  # rich purple-blue
        '5': 'orange',  # unchanged
        '6': 'cyan',  # unchanged
        '7': 'gold',  # unchanged
        '8': 'darkred',  # earthy dark brown
        '9': 'brown',  # unchanged
        '10': 'darkmagenta',  # bold dark pink
        '11': 'magenta',  # unchanged
        '12': 'lime',  # unchanged
        '13': 'teal'  # unchanged
    }

    # Create point cloud traces for each room volume
    volume_traces = []
    if point_groups is not None:
        for group_id, indices_lists in point_groups.items():
            # Get the color for this room type
            color = color_map.get(group_id, 'lightgray')

            for indices in indices_lists:
                if len(indices) < 4:  # Need at least 4 points to define a volume
                    continue

                # Extract points for this room
                room_points = points_array[indices]

                # Calculate bounding box for this room
                min_x, max_x = np.min(room_points[:, 0]), np.max(room_points[:, 0])
                min_y, max_y = np.min(room_points[:, 1]), np.max(room_points[:, 1])
                min_z, max_z = np.min(room_points[:, 2]), np.max(room_points[:, 2])

                # Calculate room volume
                room_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)

                # Calculate total number of points to generate based on density
                total_points = int(room_volume * points_per_cubic_meter)

                # Ensure a minimum number of points for visualization
                total_points = max(total_points, 50)

                volume_x = []
                volume_y = []
                volume_z = []

                # Create a convex hull or alpha shape for the room
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(room_points)

                    # Helper function to check if a point is inside the hull
                    def point_in_hull(point, hull, tolerance=1e-12):
                        from scipy.spatial import Delaunay
                        hull_points = room_points[hull.vertices]
                        hull_delaunay = Delaunay(hull_points[:, :3])  # Use all 3 dimensions
                        return hull_delaunay.find_simplex(point) >= 0

                    # Generate random points and check if they're within the hull
                    points_to_generate = total_points * 2  # Generate extra points as some will be rejected

                    # Random points within bounding box
                    random_x = np.random.uniform(min_x, max_x, points_to_generate)
                    random_y = np.random.uniform(min_y, max_y, points_to_generate)
                    random_z = np.random.uniform(min_z, max_z, points_to_generate)

                    # Check each point
                    for i in range(points_to_generate):
                        # Check if we already have enough points
                        if len(volume_x) >= total_points:
                            break

                        x_val, y_val, z_val = random_x[i], random_y[i], random_z[i]

                        # Check if point lies within building footprint (only considering x,y)
                        if footprint_polygon is not None:
                            point = Point(x_val, y_val)
                            if not footprint_polygon.contains(point):
                                continue

                        # Check if point is inside the hull
                        test_point = np.array([x_val, y_val, z_val])
                        if point_in_hull(test_point, hull):
                            volume_x.append(x_val)
                            volume_y.append(y_val)
                            volume_z.append(z_val)

                except (ImportError, ValueError, QhullError):
                    # Fallback method if ConvexHull fails or is not available
                    # Simple random points in the bounding box
                    for _ in range(total_points):
                        x_val = np.random.uniform(min_x, max_x)
                        y_val = np.random.uniform(min_y, max_y)

                        # Check if point lies within building footprint (only considering x,y)
                        if footprint_polygon is not None:
                            point = Point(x_val, y_val)
                            if not footprint_polygon.contains(point):
                                continue

                        z_val = np.random.uniform(min_z, max_z)
                        volume_x.append(x_val)
                        volume_y.append(y_val)
                        volume_z.append(z_val)

                # Create point cloud trace for this room
                if volume_x:  # Only add trace if we have points
                    volume_trace = go.Scatter3d(
                        x=volume_x,
                        y=volume_y,
                        z=volume_z,
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=color,
                            opacity=0.7
                        ),
                        name=f'Room Type {group_id}',
                        showlegend=True
                    )
                    volume_traces.append(volume_trace)

    # Create figure and add traces
    fig = go.Figure(data=edge_traces + [node_trace] + volume_traces)

    if title is None:
        title = '3D Building Visualization'

    # Update layout to hide grid, zero lines, and axes
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title='',
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                showbackground=False,
            ),
            yaxis=dict(
                title='',
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                showbackground=False,
            ),
            zaxis=dict(
                title='',
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                showbackground=False,
            ),
            bgcolor='white'
        ),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    if camera_eye_dict is not None:
        fig.update_layout(scene_camera=camera_eye_dict)

    if save_plot:
        fig.write_html("my_plot.html", auto_open=True)
        # Save the plot as an image
        pio.write_image(fig, f'/users/kevin/desktop/{title}.png', scale=6, width=1080, height=1080)

    # Show the plot
    fig.show()

def plot_images_side_by_side(left_image_path, right_image_path, left_title, right_title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Load left image
    left_img = Image.open(left_image_path)
    ax1.imshow(left_img)
    ax1.set_title(left_title)

    # Load right image
    right_img = Image.open(right_image_path)
    ax2.imshow(right_img)
    ax2.set_title(right_title)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Display the plot
    plt.show()
    # Close the figure after showing
    plt.close("all")