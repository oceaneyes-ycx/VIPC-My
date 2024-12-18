# VIPC Task Inference Code
# This script loads a trained model and performs inference on images and partial point clouds to generate completed point clouds.

import torch
import os
import numpy as np
import open3d as o3d
from PIL import Image
from model import Network
from decoder.utils.utils import farthest_point_sample

# Ensure result directory exists
result_dir = "results_inference"
os.makedirs(result_dir, exist_ok=True)
os.makedirs(os.path.join(result_dir, "partial_pc"), exist_ok=True)
os.makedirs(os.path.join(result_dir, "completed_pc"), exist_ok=True)
os.makedirs(os.path.join(result_dir, "input_images"), exist_ok=True)

def save_point_cloud_as_ply(filepath, point_cloud):
    """
    Save a point cloud to a PLY file.

    Args:
        filepath: Path to save the file.
        point_cloud: numpy array of shape [N, 3].
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(filepath, pc)

def inference(data_dir, model_path, output_dir):
    """
    Perform inference using the trained model.

    Args:
        data_dir: Path to the directory containing input images and partial point clouds.
        model_path: Path to the trained model checkpoint.
        output_dir: Path to save the completed point clouds and results.
    """
    # Load the trained model
    model = Network(pretrained_model_path=None)  # Do not load the pretrained weights again
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    model.eval()

    # Iterate through the data directory
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            # Load partial point cloud
            partial_pc_path = os.path.join(data_dir, file_name)
            partial_pc = np.loadtxt(partial_pc_path).astype(np.float32)

            # Load corresponding image
            img_name = file_name.replace(".txt", ".png")
            img_path = os.path.join(data_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()  # [H, W, C] -> [1, C, H, W]

            # Preprocess partial point cloud
            partial_pc = torch.from_numpy(partial_pc).unsqueeze(0).cuda()  # [N, 3] -> [1, N, 3]
            partial_pc = farthest_point_sample(partial_pc, 2048)  # Sample 2048 points

            # Perform inference
            with torch.no_grad():
                completed_pc = model(partial_pc, img)  # Generate completed point cloud

            # Save results
            base_name = os.path.splitext(file_name)[0]

            # Save partial point cloud
            partial_out_path = os.path.join(output_dir, "partial_pc", f"{base_name}.ply")
            save_point_cloud_as_ply(partial_out_path, partial_pc.cpu().squeeze(0).numpy())

            # Save completed point cloud
            completed_out_path = os.path.join(output_dir, "completed_pc", f"{base_name}.ply")
            save_point_cloud_as_ply(completed_out_path, completed_pc.cpu().squeeze(0).numpy())

            # Save input image
            img_out_path = os.path.join(output_dir, "input_images", f"{base_name}.png")
            img_save = (img.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # [C, H, W] -> [H, W, C]
            Image.fromarray(img_save).save(img_out_path)

if __name__ == "__main__":
    data_directory = "./inference_data"  # Directory containing input images and partial point clouds
    model_checkpoint = "./checkpoints/ckpt-best.pth"  # Path to the trained model checkpoint
    output_directory = result_dir  # Directory to save the results

    inference(data_directory, model_checkpoint, output_directory)