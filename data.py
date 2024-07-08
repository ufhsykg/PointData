import numpy as np
import math
import os

def generate_cube(num_points=200):
    """ 生成立方体点云 """
    points = np.random.rand(num_points, 3)  # 在0到1之间随机生成点
    return points

def generate_sphere(num_points=200):
    """ 生成球体点云 """
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2*np.pi, num_points)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    points = np.stack((x, y, z), axis=1)
    return points

def add_gaussian_noise(points, mean=0, std=0.01):
    """ 向点云添加高斯噪声 """
    noise = np.random.normal(mean, std, points.shape)
    noisy_points = points + noise
    return noisy_points

def save_to_xyz(points, filename, dataset_type='train'):
    """ 保存点云到XYZ文件，根据数据集类型选择目录 """
    path = f'data/{dataset_type}/{filename}'
    np.savetxt(path, points, fmt='%.6f', header='x y z', comments='')


def generate_cylinder(num_points=200):
    """ 生成圆柱体点云 """
    height = 1  # 高度
    radius = 0.5  # 半径
    heights = np.random.uniform(0, height, num_points)
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = heights
    return np.stack((x, y, z), axis=-1)

def generate_cone(num_points=200):
    """ 生成锥体点云 """
    height = 1  # 高度
    max_radius = 0.5  # 底面最大半径
    heights = np.random.uniform(0, height, num_points)
    radii = (1 - heights / height) * max_radius
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    z = heights
    return np.stack((x, y, z), axis=-1)

def rotate_points(points, angle, axis='z'):
    """ 对点云应用旋转 """
    if axis == 'z':
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        rotation_matrix = np.array([[cos_val, -sin_val, 0],
                                    [sin_val, cos_val, 0],
                                    [0, 0, 1]])
    rotated_points = np.dot(points, rotation_matrix.T)
    return rotated_points

def create_directories():
    """ 创建数据存储目录 """
    if not os.path.exists('data/train'):
        os.makedirs('data/train')
    if not os.path.exists('data/test'):
        os.makedirs('data/test')

# 更新批量生成和保存点云部分
shapes = {'cube': generate_cube, 'sphere': generate_sphere, 'cylinder': generate_cylinder, 'cone': generate_cone}
num_variants = 75
angles = np.linspace(0, 2 * np.pi, 5)
test_ratio = 0.2
num_tests = int(num_variants * test_ratio)
num_trains = num_variants - num_tests

create_directories()  # 确保文件夹存在

for shape_name, generator in shapes.items():
    for variant in range(num_variants):
        dataset_type = 'test' if variant < num_tests else 'train'
        for angle in angles:
            points = generator()
            rotated_points = rotate_points(points, angle)
            noisy_points = add_gaussian_noise(rotated_points)
            filename_prefix = f'{shape_name}_variant{variant}_angle{int(math.degrees(angle))}'
            save_to_xyz(rotated_points, f'{filename_prefix}_clean.xyz', dataset_type)
            save_to_xyz(noisy_points, f'{filename_prefix}_noisy.xyz', dataset_type)