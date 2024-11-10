import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

def rotation_x(pc, angle):
    """Rotate point cloud around x-axis by the given angle."""
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]])
    return np.dot(pc, rotation_matrix)

def rotation_y(pc, angle):
    """Rotate point cloud around y-axis by the given angle."""
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    return np.dot(pc, rotation_matrix)

def load_and_process_point_cloud(pc_path, view_metadata_path, part_view_id, image_view_id):
    # Load point cloud data
    with open(pc_path, 'rb') as f:
        pc = pickle.load(f).astype(np.float32)

    # Load view metadata
    view_metadata = np.loadtxt(view_metadata_path)

    # Convert angles from degrees to radians
    theta_part = math.radians(view_metadata[int(part_view_id), 0])
    phi_part = math.radians(view_metadata[int(part_view_id), 1])
    theta_img = math.radians(view_metadata[int(image_view_id), 0])
    phi_img = math.radians(view_metadata[int(image_view_id), 1])

    # Rotate point cloud
    pc = rotation_y(rotation_x(pc, -phi_part), np.pi + theta_part)
    pc = rotation_x(rotation_y(pc, np.pi - theta_img), phi_img)

    # Normalize point cloud
    gt_mean = pc.mean(axis=0)
    pc = pc - gt_mean
    pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
    pc = pc / pc_L_max

    return pc

def visualize_point_cloud(pc):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
    plt.show()

def save_point_cloud_to_ply(pc, ply_path):
    # Convert numpy array to open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    # Save point cloud to .ply file
    o3d.io.write_point_cloud(ply_path, pcd)

if __name__ == "__main__":
    # Replace these paths with your local .dat file paths
    pc_path = './part/02.dat'
    view_metadata_path = './rendering/rendering_metadata.txt'
    part_view_id = '02'  # Replace with actual part view ID
    image_view_id = '02'  # Replace with actual image view ID

    pc = load_and_process_point_cloud(pc_path, view_metadata_path, part_view_id, image_view_id)
    visualize_point_cloud(pc)
    # Save point cloud to .ply file
    ply_path = './output/point_cloud.ply'
    save_point_cloud_to_ply(pc, ply_path)
    print(f"Point cloud saved to {ply_path}")