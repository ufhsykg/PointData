import torch
import numpy as np
import torch.nn as nn

class PointCloudAutoencoder(nn.Module):
    def __init__(self):
        super(PointCloudAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(200 * 3, 512),  # Flatten input and pass through FC
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # Latent space representation
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 200 * 3)  # Output the same number of points as the input
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the input
        z = self.encoder(x)  # Latent space representation
        out = self.decoder(z)
        out = out.view(batch_size, 200, 3)  # Reshape output to match input shape
        return out


def load_xyz(filename):
    """ 读取.xyz文件 """
    points = np.loadtxt(filename, skiprows=1)  # 假设文件的第一行是头信息
    return points

def save_xyz(points, filename):
    """ 将点云保存到.xyz文件 """
    np.savetxt(filename, points, fmt='%.6f', header='x y z', comments='')

def denoise_point_cloud(model_path, input_xyz, output_xyz, device):
    # 加载模型
    model = PointCloudAutoencoder()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # 读取并处理输入点云
    points = load_xyz(input_xyz)
    points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # 增加批量维度
    points_tensor = points_tensor.to(device)

    # 模型去噪
    with torch.no_grad():
        output_points_tensor = model(points_tensor)
    output_points = output_points_tensor.squeeze(0).cpu().numpy()  # 去除批量维度并转回numpy

    # 保存去噪后的点云
    save_xyz(output_points, output_xyz)

# 使用示例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
denoise_point_cloud('model_final.pth', 'test.xyz', 'output_denoised.xyz', device)
