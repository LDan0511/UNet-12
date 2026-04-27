import csv
import os

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import MyDataset
from model.UNet import UNetMich  # 确保你用的是 U-Net 模型
from utils import load_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'

# 设置路径和数据
BASE_DIR = r"C:\Users\hanst\Desktop\DeepLearning\PaperRevise"
TEST_DIR_LIST = [os.path.join(BASE_DIR, "test")]
REAL_MICH_SIZE = (155, 78)  # 图像大小
VIRTUAL_MICH_SIZE = (155, 78)

# 模型加载
CHECKPOINT_DIR = "checkpoints_unet/"
CHECKPOINT_NAME = "unet_mich_best.pth.tar"  # 最佳模型路径（在训练时保存的）

# 初始化 U-Net 模型
net = UNetMich(in_ch=1, out_ch=1, base_ch=32, residual=True).to(DEVICE)
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
net.eval()  # 设置为评估模式
load_checkpoint(CHECKPOINT_DIR + CHECKPOINT_NAME, net)  # 加载最佳模型

# 准备测试集
real_area = []
virtural_area = []
virtual_mask = []
total = False  # 根据实际情况修改为 True 或 False

test_dataset = MyDataset(
    root_dir=TEST_DIR_LIST[0],
    real_size=REAL_MICH_SIZE,
    virtual_size=VIRTUAL_MICH_SIZE,
    total=total,
    real_area=real_area,
    virtural_area=virtural_area,
    virtual_mask=virtual_mask
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# 评估函数：计算 PSNR 和 SSIM
def evaluation(y_fake, y):
    # 确保输入和输出图像的像素值在 [0, 1] 或 [0, 255] 之间
    y_fake = np.clip(y_fake, 0, 1)  # 或者 0 到 255，取决于你的图像范围
    y = np.clip(y, 0, 1)

    # 计算 PSNR
    psnr = peak_signal_noise_ratio(y, y_fake, data_range=1.0)

    # 计算 SSIM
    win_size = min(y.shape[0], y.shape[1], 7)  # 设置 win_size，确保不超过图像尺寸
    ssim = structural_similarity(y, y_fake, data_range=1.0, gaussian_weights=True,
                                 use_sample_covariance=False, sigma=1.5, K1=0.01, K2=0.03, win_size=win_size)

    return psnr, ssim

# 保存结果的函数
import os

def save_test(image_name, output, input, label):
    # 确保每个文件夹存在
    input_dir = 'inputs/'
    output_dir = 'outputs/'
    label_dir = 'labels/'

    # 创建文件夹（如果不存在的话）
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # 转换为 NumPy 数组后保存
    output = output.cpu().numpy()  # 转换为 NumPy 数组
    input = input.cpu().numpy()    # 转换为 NumPy 数组
    label = label.cpu().numpy()    # 转换为 NumPy 数组

    # 保存为 .mich 文件，分别保存在不同的文件夹中
    output.tofile(output_dir + image_name.split('/')[-1] + '_output.mich')
    input.tofile(input_dir + image_name.split('/')[-1] + '_input.mich')
    label.tofile(label_dir + image_name.split('/')[-1] + '_label.mich')

# 测试过程
def test():
    results = []

    # 对每一张图片进行测试
    with open("evaluation_results.csv", mode="w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "PSNR", "SSIM"])  # 写入表头

        for idx, (xT, x0) in enumerate(test_loader):
            with torch.no_grad():
                image_name = test_dataset.data_name_list[idx].split(os.sep)[-1]
                print(f"Inferencing: {image_name}")

                # 将输入和标签放到设备上
                xT = Variable(xT).to(DEVICE)  # 输入（无 DOI）
                x0 = Variable(x0).to(DEVICE)  # 标签（有 DOI）

                # 使用 U-Net 模型进行推理
                output = net(xT)  # 得到输出

                # 计算 PSNR 和 SSIM
                psnr, ssim = evaluation(output.cpu().numpy(), x0.cpu().numpy())

                # 保存测试结果
                save_test(f"results/{image_name}", output, xT, x0)

                # 写入 CSV 文件
                writer.writerow([image_name, psnr, ssim])
                results.append((image_name, psnr, ssim))

    # 打印整体结果
    for result in results:
        print(f"Image: {result[0]} | PSNR: {result[1]:.4f} | SSIM: {result[2]:.4f}")

if __name__ == "__main__":
    test()