import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset import PointCloudDataset
import numpy as np
import os
import torch.optim as optim
from dist_chamfer import chamfer_3DDist
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def chamfer_distance(p1, p2):
    """
    计算两个点云之间的Chamfer距离
    :param p1: 第一个点云 [B, N, 3]
    :param p2: 第二个点云 [B, N, 3]
    """
    B, N, _ = p1.size()

    # 扩展p1和p2至[B, N, N, 3]
    p1 = p1.unsqueeze(2).expand(B, N, N, 3)
    p2 = p2.unsqueeze(1).expand(B, N, N, 3)

    # 计算所有点对的欧式距离
    dist = torch.norm(p1 - p2, dim=3)  # [B, N, N]

    # 对于p1中的每个点，找到p2中的最近点，然后计算平均距离
    min_dist_p1_p2 = dist.min(dim=2)[0]  # [B, N]
    min_dist_p2_p1 = dist.min(dim=1)[0]  # [B, N]

    # 计算Chamfer距离
    chamfer_dist = torch.mean(min_dist_p1_p2 + min_dist_p2_p1, dim=1)  # [B]
    return chamfer_dist.mean()  # 标量


chamfer_dist = chamfer_3DDist().to(device)


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
            nn.Linear(128, 64),  # Latent space representation
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 200 * 3),  # Output the same number of points as the input
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the input
        z = self.encoder(x)  # Latent space representation
        out = self.decoder(z)
        out = out.view(batch_size, 200, 3)  # Reshape output to match input shape
        return out


# 更新DataLoader和数据转换函数，确保数据在加载时被传送到正确的设备
def to_tensor(sample):
    noisy, clean = sample["noisy"], sample["clean"]
    return {
        "noisy": torch.tensor(noisy, dtype=torch.float32).to(device),
        "clean": torch.tensor(clean, dtype=torch.float32).to(device),
    }


train_dataset = PointCloudDataset("data/train", transform=to_tensor)
test_dataset = PointCloudDataset("data/test", transform=to_tensor)
val_dataset = PointCloudDataset("data/val", transform=to_tensor)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)


model = PointCloudAutoencoder().cuda()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
save_path = "model"


def validate(model, test_loader, device):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            noisy_points = batch["noisy"].to(device)
            clean_points = batch["clean"].to(device)
            outputs = model(noisy_points)
            loss = chamfer_dist(outputs, clean_points)
            total_loss += loss.item()
    return total_loss / len(test_loader)


def train(num_epochs, model, train_loader, test_loader, optimizer, device, save_path):
    model.train()
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            noisy_points = batch["noisy"].to(device)
            clean_points = batch["clean"].to(device)
            optimizer.zero_grad()
            outputs = model(noisy_points)
            loss = chamfer_dist(outputs, clean_points)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 计算验证集上的损失
        avg_val_loss = validate(model, test_loader, device)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, num_epochs + 1),
        train_losses,
        marker="o",
        linestyle="-",
        color="b",
        label="Training Loss",
    )
    plt.plot(
        range(1, num_epochs + 1),
        val_losses,
        marker="x",
        linestyle="--",
        color="r",
        label="Validation Loss",
    )
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 保存最终模型为.pth格式
    torch.save(model.state_dict(), f"{save_path}_final.pth")


def export_to_onnx(model, input_size, save_path):
    model.eval()  # 确保模型处于评估模式
    dummy_input = torch.randn(input_size, device=device)  # 创建一个假的输入
    torch.onnx.export(
        model,
        dummy_input,
        f"{save_path}.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )


# 假定 train_loader 已经定义
train(30, model, train_loader, val_loader, optimizer, device, "save")

# 转换模型为ONNX
export_to_onnx(model, (1, 200, 3), "model_final")
