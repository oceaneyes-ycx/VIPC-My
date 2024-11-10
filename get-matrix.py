import numpy as np

def generate_camera_matrices(azimuth, elevation, in_plane_rotation, distance, field_of_view):
    # Convert angles from degrees to radians
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)
    in_plane_rotation = np.radians(in_plane_rotation)
    
    # Intrinsic matrix
    # Assuming square pixels and principal point at the center
    f = 1 / np.tan(np.radians(field_of_view) / 2)
    intrinsic_matrix = np.array([
        [f, 0, 0],
        [0, f, 0],
        [0, 0, 1]
    ])
    
    # Extrinsic matrix
    # Rotation matrices for azimuth, elevation, and in-plane rotation
    R_azimuth = np.array([
        [np.cos(azimuth), -np.sin(azimuth), 0],
        [np.sin(azimuth), np.cos(azimuth), 0],
        [0, 0, 1]
    ])
    
    R_elevation = np.array([
        [1, 0, 0],
        [0, np.cos(elevation), -np.sin(elevation)],
        [0, np.sin(elevation), np.cos(elevation)]
    ])
    
    R_in_plane = np.array([
        [np.cos(in_plane_rotation), 0, np.sin(in_plane_rotation)],
        [0, 1, 0],
        [-np.sin(in_plane_rotation), 0, np.cos(in_plane_rotation)]
    ])
    
    # Combined rotation matrix
    R = R_azimuth @ R_elevation @ R_in_plane
    
    # Translation vector
    t = np.array([0, 0, -distance])
    
    # Extrinsic matrix [R | t]
    extrinsic_matrix = np.hstack((R, t.reshape(-1, 1)))
    
    return intrinsic_matrix, extrinsic_matrix

if __name__ == "__main__":
    # Example line from the file
    line = "42.6092002399 26.194836774 0 0.854632067268 25"
    params = list(map(float, line.split()))
    
    azimuth = params[0]
    elevation = params[1]
    in_plane_rotation = params[2]
    distance = params[3]
    field_of_view = params[4]
    
    intrinsic_matrix, extrinsic_matrix = generate_camera_matrices(azimuth, elevation, in_plane_rotation, distance, field_of_view)
    
    print("Intrinsic Matrix:")
    print(intrinsic_matrix)
    print("\nExtrinsic Matrix:")
    print(extrinsic_matrix)