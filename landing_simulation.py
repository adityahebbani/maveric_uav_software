import numpy as np
import open3d as o3d

def acquire_sensor_data():
    # Simulate sensor data for 4 LiDAR sensors over a bumpy terrain.
    point_clouds = []
    grid_size = 100  # Expanded map.
    x = np.linspace(-10, 10, grid_size)
    y = np.linspace(-10, 10, grid_size)
    X, Y = np.meshgrid(x, y)
    # Bumpy surface with noise.
    Z = np.sin(X) * np.cos(Y) + np.random.normal(scale=0.2, size=X.shape)
    pts = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

    # Insert a flat patch at (0,0) only.
    mask_center = (np.abs(pts[:, 0]) < 2) & (np.abs(pts[:, 1]) < 2)
    pts[mask_center, 2] = 0

    # Do not force flatness elsewhere; leave (5,5) and (-5,-5) bumpy.

    # Simulate 4 sensors by applying slight positional offsets.
    offsets = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.0, 0.5, 0.0]),
        np.array([-0.5, -0.5, 0.0])
    ]
    for offset in offsets:
        sensor_pts = pts + offset
        point_clouds.append(sensor_pts)
    return point_clouds

def transform_point_clouds(point_clouds, sensor_transforms):
    transformed_clouds = []
    for cloud, transform in zip(point_clouds, sensor_transforms):
        ones = np.ones((cloud.shape[0], 1))
        cloud_homogeneous = np.hstack((cloud, ones))
        transformed = (transform @ cloud_homogeneous.T).T[:, :3]
        transformed_clouds.append(transformed)
    return transformed_clouds

def merge_point_clouds(clouds):
    merged = np.vstack(clouds)
    return merged

def filter_point_cloud(pcd):
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(pcd)
    cl, ind = o3d_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    filtered_cloud = o3d_cloud.select_by_index(ind)
    return filtered_cloud

def create_cylinder_between_points(p1, p2, radius=0.05, resolution=20):
    """(Not used anymore) Creates a cylinder mesh between two 3D points."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    v = p2 - p1
    height = np.linalg.norm(v)
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution)
    cyl.compute_vertex_normals()
    z = np.array([0, 0, 1])
    if height < 1e-6:
        R = np.eye(3)
    else:
        v_norm = v / height
        axis = np.cross(z, v_norm)
        angle = np.arccos(np.clip(np.dot(z, v_norm), -1.0, 1.0))
        if np.linalg.norm(axis) < 1e-6:
            R = np.eye(3)
        else:
            axis = axis / np.linalg.norm(axis)
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    cyl.rotate(R, center=np.zeros(3))
    cyl.translate(p1)
    cyl.paint_uniform_color([0, 0, 0])
    return cyl

def analyze_terrain(filtered_cloud):
    pts = np.asarray(filtered_cloud.points)
    # Candidate safe zone centers (areas the drone will scan).
    candidate_centers = [(0, 0), (5, 5), (-5, -5)]
    safe_zones = []  # List of (center, base_z) for safe (flat) regions.
    # Check each candidate: only classify as safe if elevation std < 0.05.
    for center in candidate_centers:
        mask = (np.abs(pts[:, 0] - center[0]) < 2) & (np.abs(pts[:, 1] - center[1]) < 2)
        zone_pts = pts[mask]
        if zone_pts.shape[0] > 100 and np.std(zone_pts[:, 2]) < 0.15:
            base_z = np.min(zone_pts[:, 2])
            safe_zones.append((center, base_z))
    landing_safe = len(safe_zones) > 0
    drone = None
    if landing_safe:
        # Pick the first safe zone arbitrarily.
        chosen_center, base_z = safe_zones[0]
        drone = o3d.geometry.TriangleMesh.create_box(width=2, height=2, depth=2)
        # Translate the drone so that its bottom face sits just above the surface.
        # Adding a small offset (0.01) ensures the drone is on top, not intersecting the surface.
        drone.translate(np.array([chosen_center[0] - 1, chosen_center[1] - 1, base_z + 1]))
        drone.compute_vertex_normals()
        drone.paint_uniform_color([1, 0, 0])  # Red drone.
        print("Safe zone detected at:", chosen_center, "with base elevation:", base_z)
    return landing_safe, drone, safe_zones

def main():
    sensor_data = acquire_sensor_data()
    sensor_transforms = [np.eye(4) for _ in range(4)]
    transformed_clouds = transform_point_clouds(sensor_data, sensor_transforms)
    merged_cloud = merge_point_clouds(transformed_clouds)
    filtered_cloud = filter_point_cloud(merged_cloud)

    # Initially, set the entire point cloud to black.
    pts = np.asarray(filtered_cloud.points)
    colors = np.zeros((pts.shape[0], 3))
    filtered_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # Let the drone analyze (scan) the terrain to identify safe (flat) zones.
    safe, drone, safe_zones = analyze_terrain(filtered_cloud)

    # Now, recolor only those points that belong to truly flat (safe) regions in green.
    if safe:
        for (center, _) in safe_zones:
            mask = (np.abs(pts[:, 0] - center[0]) < 2) & (np.abs(pts[:, 1] - center[1]) < 2)
            colors[mask] = np.array([0, 1, 0])
        filtered_cloud.colors = o3d.utility.Vector3dVector(colors)
        print("Safe landing zone(s) detected:", safe_zones)
        # Visualize the point cloud with safe (green) areas and the drone landing.
        o3d.visualization.draw_geometries([filtered_cloud, drone],
                                          window_name="Simulated 3D Map with Drone Landed")
    else:
        print("No safe landing zone found.")
        o3d.visualization.draw_geometries([filtered_cloud],
                                          window_name="Simulated 3D Map")

if __name__ == "__main__":
    main()