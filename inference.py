# VIPC Task Inference Code
# This script loads a trained model and performs inference on images and partial point clouds to generate completed point clouds.

import torch
import os
import numpy as np
import open3d as o3d
from PIL import Image
from model_pre import Network
from decoder.utils.utils import farthest_point_sample
from Symm.symm import SymmCompletion, SymmCompletion2, SymmCompletion3
import config_vipc

import os
import math
import pickle
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Ensure result directory exists
result_dir = "./target"
os.makedirs(result_dir, exist_ok=True)
os.makedirs(os.path.join(result_dir, "partial_pc"), exist_ok=True)
os.makedirs(os.path.join(result_dir, "completed_pc"), exist_ok=True)
os.makedirs(os.path.join(result_dir, "input_images"), exist_ok=True)
os.makedirs(os.path.join(result_dir, "keypoints"), exist_ok=True)
os.makedirs(os.path.join(result_dir, "symmetry_points"), exist_ok=True)
os.makedirs(os.path.join(result_dir, "coarse"), exist_ok=True)
os.makedirs(os.path.join(result_dir, "fine"), exist_ok=True)
os.makedirs(os.path.join(result_dir, "GT_PC"), exist_ok=True)

def rotation_x(points, angle):
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, cosval, -sinval],
                                [0, sinval, cosval]])
    return np.dot(points, rotation_matrix.T)

def rotation_y(points, angle):
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    return np.dot(points, rotation_matrix.T)




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
    cfg = config_vipc.cfg 
    model = SymmCompletion3(cfg)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    model.eval()

    meta_path = os.path.join(data_dir, 'rendering_metadata.txt')
    view_metadata = np.loadtxt(meta_path)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".dat") and not file_name.startswith("gt"):
            base_name = os.path.splitext(file_name)[0]
            view_idx = int(base_name)  # e.g., '03' -> 3

            # --- Load .dat point clouds ---
            partial_pc_path = os.path.join(data_dir, f"{base_name}.dat")
            with open(partial_pc_path, 'rb') as f:
                pc_part = pickle.load(f).astype(np.float32)

            gt_pc_path = os.path.join(data_dir, f"gt_{base_name}.dat")
            with open(gt_pc_path, 'rb') as f:
                pc_gt = pickle.load(f).astype(np.float32)

            # --- Load image and metadata ---
            img_path = os.path.join(data_dir, f"{base_name}.png")
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image)[:3, :, :]  # Only RGB
            image_tensor = image_tensor.unsqueeze(0).cuda()

            theta_part = math.radians(view_metadata[view_idx, 0])
            phi_part = math.radians(view_metadata[view_idx, 1])
            theta_img = theta_part
            phi_img = phi_part

            # --- Rotate and normalize point clouds ---
            pc_part = rotation_y(rotation_x(pc_part, -phi_part), math.pi + theta_part)
            pc_part = rotation_x(rotation_y(pc_part, math.pi - theta_img), phi_img)

            pc_gt = rotation_y(rotation_x(pc_gt, -phi_part), math.pi + theta_part)
            pc_gt = rotation_x(rotation_y(pc_gt, math.pi - theta_img), phi_img)

            gt_mean = pc_gt.mean(axis=0)
            pc_gt = pc_gt - gt_mean
            pc_L_max = np.max(np.sqrt(np.sum(pc_gt ** 2, axis=-1)))
            pc_gt = pc_gt / pc_L_max

            pc_part = (pc_part - gt_mean) / pc_L_max

            # --- Sample and convert ---
            pc_part_tensor = torch.from_numpy(pc_part).unsqueeze(0).float().cuda()
            pc_part_tensor = farthest_point_sample(pc_part_tensor, 2048)

            pc_gt_tensor = torch.from_numpy(pc_gt).unsqueeze(0).float().cuda()
            pc_gt_tensor = farthest_point_sample(pc_gt_tensor, 2048)

            with torch.no_grad():
                _, completed_pc, _, ret = model(pc_part_tensor, image_tensor)

            # Save results
            save_point_cloud_as_ply(os.path.join(output_dir, "partial_pc", f"{base_name}.ply"),
                                    pc_part_tensor.squeeze(0).cpu().numpy())
            save_point_cloud_as_ply(os.path.join(output_dir, "GT_PC", f"{base_name}.ply"),
                                    pc_gt_tensor.squeeze(0).cpu().numpy())
            save_point_cloud_as_ply(os.path.join(output_dir, "completed_pc", f"{base_name}.ply"),
                                    completed_pc.squeeze(0).cpu().numpy())
            image.save(os.path.join(output_dir, "input_images", f"{base_name}.png"))

            save_point_cloud_as_ply(os.path.join(output_dir, "keypoints", f"{base_name}.ply"),
                                    ret[0].permute(0, 2, 1).squeeze(0).cpu().numpy())
            save_point_cloud_as_ply(os.path.join(output_dir, "symmetry_points", f"{base_name}.ply"),
                                    ret[1].permute(0, 2, 1).squeeze(0).cpu().numpy())
            save_point_cloud_as_ply(os.path.join(output_dir, "coarse", f"{base_name}.ply"),
                                    ret[2].permute(0, 2, 1).squeeze(0).cpu().numpy())
            save_point_cloud_as_ply(os.path.join(output_dir, "fine", f"{base_name}.ply"),
                                    ret[3].squeeze(0).cpu().numpy())
            


            print(f"[✓] Processed {file_name}")

if __name__ == "__main__":
    data_directory = "./target"  # Directory containing input images and partial point clouds
    model_checkpoint = "../checkpoints/VIPC/ckpt-best.pth"  # Path to the trained model checkpoint
    output_directory = result_dir  # Directory to save the results

    inference(data_directory, model_checkpoint, output_directory)