import subprocess
import cv2
import os
import glob
import numpy as np
import time
import serial
import threading
import struct
import random

NUM_POINTS = 300


class Point3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def create_sphere(radius):
    points = []
    for i in range(NUM_POINTS):
        theta = np.arccos(1 - 2 * i / (NUM_POINTS - 1))
        phi = np.sqrt(NUM_POINTS * np.pi) * theta

        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)

        points.append(Point3D(x, y, z))
    return points


def read_xyz(file_path):
    # 从.xyz文件读取点云数据
    points = np.loadtxt(file_path, delimiter=" ")
    return points


def read_xyz_file(filename):
    points = []
    with open(filename, "r") as file:
        lines = file.readlines()
        for line in lines:
            if len(points) >= NUM_POINTS:
                break
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            x, y, z = map(float, parts)
            points.append(Point3D(x, y, z))

    if len(points) > NUM_POINTS:
        points = points[:NUM_POINTS]

    elif len(points) < NUM_POINTS:
        while len(points) < NUM_POINTS:
            points.append(random.choice(points))

    return points


def normalize_points(points, scale_factor=100.0):
    # 计算点云中所有点到原点的最大距离
    max_distance = max(np.sqrt(p.x**2 + p.y**2 + p.z**2) for p in points)

    # 计算缩放因子
    scale = scale_factor / max_distance

    # 等比例缩放每个点的坐标
    for p in points:
        p.x *= scale
        p.y *= scale
        p.z *= scale

    return points


def send_points_via_serial(points, ser):
    try:
        for p in points:
            data = f"{p.x:.2f} {p.y:.2f} {p.z:.2f}\n".encode("utf-8")
            i = 0
            while i < len(data):
                if data[i] == ord("\n"):
                    ser.write(data[i : i + 1])
                    print(f"Sent: {data[i:i+1]}")
                    i += 1
                    time.sleep(0.02)
                else:
                    ser.write(data[i : i + 2])
                    print(f"Sent: {data[i:i+2]}")
                    i += 2
                    time.sleep(0.02)
        end_message = b"END\n"
        for i in range(0, len(end_message)):
            ser.write(end_message[i : i + 1])
            print(f"Sent: {end_message[i:i+1]}")
            time.sleep(0.01)
    finally:
        pass
        # ser.close()


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


def receive_data_via_serial(ser):
    try:
        while True:
            if ser.in_waiting > 0:
                data = ser.readline().decode("utf-8").strip()
                print(f"Received: {data}")
    except KeyboardInterrupt:
        print("Serial communication stopped.")
    finally:
        ser.close()


def save_frame_from_stream(
    cap, save_directory="ZeroShape/t", image_filename="frame.jpg"
):
    """
    从视频流中获取一帧并保存到指定目录。

    参数:
    - cap: cv2.VideoCapture 对象，视频流。
    - save_directory: str, 图片保存的目录。
    - image_filename: str, 保存图片的文件名。
    """
    if not cap.isOpened():
        print("无法打开视频流或文件")
        return

    # 确保保存目录存在
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 从视频流中读取一帧
    ret, frame = cap.read()
    if ret:
        # 删除该目录下所有旧图片
        files = glob.glob(f"{save_directory}/*")
        for f in files:
            os.remove(f)

        # 保存新图片
        img_path = os.path.join(save_directory, image_filename)
        cv2.imwrite(img_path, frame)
        print(f"图片已保存至 {img_path}")
    else:
        print("无法从视频流中获取帧")


def run_zero_shape():
    # 保存当前目录路径
    main_directory = os.getcwd()

    # 切换到 ZeroShape 文件夹
    os.chdir("ZeroShape")

    # 运行 preprocess.py
    print("Running preprocess.py...")
    subprocess.run(["python", "preprocess.py", "t"], check=True)

    # 运行 demo.py
    print("Running demo.py...")
    subprocess.run(
        [
            "python",
            "demo.py",
            "--yaml=options/shape.yaml",
            "--task=shape",
            "--datadir=my_examples",
            "--eval.vox_res=128",
            "--ckpt=weights/shape.ckpt",
        ],
        check=True,
    )

    # 返回到主目录
    os.chdir(main_directory)
    print("Returned to main directory.")

    # 在主目录中执行其他任务
    print("Performing other tasks in main directory...")
    # 这里可以添加其他操作


if __name__ == "__main__":
    # 配置视频流
    ip_address = "192.168.10.90"
    port = "554"
    username = "admin"
    password = "123456"
    stream_path = "/stream1"
    stream_url = f"rtsp://{username}:{password}@{ip_address}:{port}{stream_path}"
    cap = cv2.VideoCapture(stream_url)

    # 调用函数
    save_frame_from_stream(cap)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    # run_zero_shape() 重要
