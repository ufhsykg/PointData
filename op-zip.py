import open3d as o3d
import numpy as np


def read_xyz_file(file_path):
    # 从.xyz文件读取点云数据
    points = np.loadtxt(file_path, delimiter=" ")
    return points


def save_xyz(point_cloud, filename="output.xyz"):
    """将 Open3D 点云对象保存到.xyz文件"""
    # 将点云对象的点转换为 NumPy 数组
    points = np.asarray(point_cloud.points)
    if points.size == 0:
        print("没有点要保存。")
        return
    # 保存到文件
    np.savetxt(filename, points, fmt="%.6f", comments="")


def adjust_voxel_size(point_cloud, target_points, tolerance=10, max_iterations=100):
    voxel_size_min = 0.001
    voxel_size_max = 10.0
    iteration = 0

    while iteration < max_iterations:
        voxel_size = (voxel_size_min + voxel_size_max) / 2
        downsampled_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
        num_points = len(downsampled_cloud.points)

        print(f"Iteration {iteration}: Voxel size {voxel_size}, Points {num_points}")

        if abs(num_points - target_points) <= tolerance:
            return downsampled_cloud, voxel_size
        elif num_points > target_points:
            voxel_size_min = voxel_size
        else:
            voxel_size_max = voxel_size

        iteration += 1

    return downsampled_cloud, voxel_size


def visualize_point_cloud(point_cloud):
    print(point_cloud)
    save_xyz(point_cloud)
    # 可视化点云
    # o3d.visualization.draw_geometries([point_cloud])


target_points = 200
input_points = read_xyz_file("retest.xyz")

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(input_points)

downsampled_cloud, final_voxel_size = adjust_voxel_size(point_cloud, target_points)
print("Final voxel size:", final_voxel_size)

visualize_point_cloud(downsampled_cloud)
