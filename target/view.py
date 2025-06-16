import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
import sys

def read_ply(filename):
    plydata = PlyData.read(filename)
    vertex = plydata['vertex']
    
    x = np.array(vertex['x'])
    y = np.array(vertex['y'])
    z = np.array(vertex['z'])

    return np.vstack((x, y, z)).T  # Shape: (N, 3)

def visualize_point_cloud(points, title='Point Cloud'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5, c=points[:, 2], cmap='viridis')
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python visualize_ply.py pointcloud.ply")
        sys.exit(1)

    filename = sys.argv[1]
    point_cloud = read_ply(filename)
    visualize_point_cloud(point_cloud)
