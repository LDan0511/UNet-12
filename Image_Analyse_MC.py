from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import numpy as np
import glob
import os
import csv
# from piq import ms_ssim
import torch
import matplotlib.pyplot as plt

# 注意修改：recon_path，

shape = (208, 208)
shape = (160, 160)
small_shape = (158, 158)
start_pixel = round((shape[0] - small_shape[0]) / 2)
end_pixel = round(start_pixel + small_shape[0])

start_it = 50
end_it = 50

Train_Datasets = ["PhantomHot", "ROBYC"]
Test_Datasets = ["PhantomHot", "PhantomHotMC", "PhantomHotMC-P", "PhantomHotMC-P-A", "PhantomHotMC-PRS",
                 "PhantomHotMC-PRS-A",
                 "ROBYC", "ROBYCMC", "ROBYCMC-P", "ROBYCMC-P-A", "ROBYCMC-PRS", "ROBYCMC-PRS-A"]

Train_Dataset = "PhantomHot"
Test_Dataset = "ROBYCMC-PRS"
Noise_Type = "P20"
Attn_Flag = False
Attn_Flag = True
SRM_Type = "MC"
SRM_Type = "RT"
DOI = 4
# DOI = "output"

MICHs = ["input", "label", "output1", "output2"]
# MICHs = ["output1", "output2"]
MICHs = ["input", "label", "output"]
# MICHs = ["output"]

POISSON = False
POISSON = True
TOTAL = False
# TOTAL = True
SIMULATION = True
TRANSFER = False
TRANSFER = True

if Test_Dataset in ["PaperRevise", "PaperReviseMC"]:
    true_path = "J:\\image_true\\PaperRevise"  # 真值图像
elif Test_Dataset in ["PhantomHotMC", "PhantomHotMC-P-A", "PhantomHotMC-PRS-A"]:
    true_path = "J:\\image_true\\PhantomHotMC"  # 真值图像
elif Test_Dataset in ["ROBYCMC", "ROBYCMC-P-A", "ROBYCMC-PRS-A"]:
    true_path = "J:\\image_true\\ROBYCMC"  # 真值图像
else:
    true_path = "J:\\image_true\\" + Test_Dataset + "\\test"  # 真值图像

if TOTAL:
    NAME = "total-"
else:
    NAME = "seg-"

# Slice_Compact2
if POISSON:
    if TRANSFER:
        recon_path = "J:\\Real_Slice\\image\\" + NAME + Train_Dataset + "P_" + Test_Dataset + "P_qianyi"
    else:
        recon_path = "J:\\Real_Slice\\image\\" + NAME + Train_Dataset + "P_" + Test_Dataset + "P"
else:
    if TRANSFER:
        recon_path = "J:\\Real_Slice\\image\\" + NAME + Train_Dataset + "_" + Test_Dataset + "_qianyi"  # 重建图像
    else:
        recon_path = "J:\\Real_Slice\\image\\" + NAME + Train_Dataset + "_" + Test_Dataset  # 重建图像

recon_input_path = r"C:\Users\hanst\Desktop\DeepLearning\newcode\UNet\rebuild_input"
recon_label_path = r"C:\Users\hanst\Desktop\DeepLearning\newcode\UNet\rebuild_label"
recon_output_path = r"C:\Users\hanst\Desktop\DeepLearning\newcode\UNet\rebuild_output"

csv_output_path = r"C:\Users\hanst\Desktop\DeepLearning\newcode\UNet\metric_result"
os.makedirs(csv_output_path, exist_ok=True)
true_path = r"C:\Users\hanst\Desktop\DeepLearning\movetest\region\test"

print(true_path)
print(recon_input_path)
print(recon_label_path)
print(recon_output_path)


def Calculation_PSNR_SSIM(y, y_fake, data_range):
    psnr = peak_signal_noise_ratio(y, y_fake, data_range=data_range)

    # 调用M-SSIM
    ssim = structural_similarity(y, y_fake, data_range=data_range, gaussian_weights=False, use_sample_covariance=False,
                                 K1=0.01, K2=0.03)

    # 调用MS-SSIM
    # y = torch.from_numpy(y)
    # y = y.unsqueeze(0).unsqueeze(0)
    # y_fake = torch.from_numpy(y_fake)
    # y_fake = y_fake.unsqueeze(0).unsqueeze(0)
    # ssim = ms_ssim.multi_scale_ssim(y, y_fake)

    # 按照公式写的SSIM
    # c1 = (0.01 * data_range) * (0.01 * data_range)
    # c2 = (0.03 * data_range) * (0.03 * data_range)
    #
    # y = y.flatten()
    # y_fake = y_fake.flatten()
    #
    # Mean_y_fake = np.mean(y_fake)
    # Mean_y = np.mean(y)
    # Value_COV = np.cov(y, y_fake)
    #
    # ssim = (2 * Mean_y * Mean_y_fake + c1) * (2 * Value_COV[0, 1] + c2) / (
    #         Mean_y * Mean_y + Mean_y_fake * Mean_y_fake + c1) / (Value_COV[0, 0] + Value_COV[1, 1] + c2)

    return psnr, ssim


# 获取真值图像
true_images = glob.glob(os.path.join(true_path, '*.true'))
true_images.sort()

# 计算重建图像域真值图像间的PSNR与SSIM
for mich_Type in MICHs:
    Parameter = np.zeros((10, 3))
    for j in range(start_it, end_it + 1, 50):  # 按照迭代次数读重建图像
        Row_ID = int(j / 50 - 1)
        Parameter[Row_ID, 0] = j
        PSNR_MEAN = 0
        SSIM_MEAN = 0

        csv_name = os.path.join(csv_output_path, f"{Test_Dataset}-{mich_Type}-PSNR-SSIM-{j}.csv")

        with open(csv_name, mode="w", newline='') as f:
            for i in range(len(true_images)):
                # 读取真值图像
                name_true_image = true_images[i]
                # print(name_true_image)

                if name_true_image == "J:\image_true\PhantomHotMC\Derenzo1804.true" or name_true_image == "J:\image_true\PhantomHotMC\Derenzo1803.true":
                    continue

                true_image = np.fromfile(name_true_image, dtype=np.float32)
                true_image = true_image / np.max(true_image)
                true_image = true_image.reshape(shape)

                # 对真值图像进行抠图，提取位于中间的156×156的图像块
                true_image = true_image[start_pixel:end_pixel, start_pixel:end_pixel]

                # 根据真值图像名解析重建图像文件名
                phantom = os.path.basename(name_true_image)
                phantom = phantom[:phantom.find('.')]

                max_psnr = 0
                max_ssim = 0
                ID_recon_psnr = 0
                ID_recon_ssim = 0

                phantom = os.path.basename(name_true_image)
                phantom = phantom[:phantom.find('.')]  # CSlice5401

                idx = phantom.replace("CSlice", "")  # 5401

                if mich_Type == "input":
                    recon_dir = recon_input_path
                    recon_name = f"PCSlice{idx}.da_MC_input_{j}.dat"
                elif mich_Type == "label":
                    recon_dir = recon_label_path
                    recon_name = f"PCSlice{idx}.im_MC_label_{j}.dat"
                elif mich_Type == "output":
                    recon_dir = recon_output_path
                    recon_name = f"PCSlice{idx}.dat_MC_output_{j}.dat"
                else:
                    print("未知 mich_Type:", mich_Type)
                    continue

                name_recon_image = os.path.join(recon_dir, recon_name)

                if not os.path.exists(name_recon_image):
                    print("找不到文件:", name_recon_image)
                    continue

                print(name_recon_image)
                recon_image = np.fromfile(name_recon_image, dtype=np.float32)
                recon_max = np.max(recon_image)
                if recon_max == 0:
                    print("全0图:", name_recon_image)
                    continue
                recon_image = recon_image / recon_max
                recon_image = recon_image.reshape(shape)

                # 对重建图像进行抠图，提取位于中间的156×156的图像块
                recon_image = recon_image[start_pixel:end_pixel, start_pixel:end_pixel]

                # 计算PSNR和SSIM
                data_range = np.max(true_image)
                psnr, ssim = Calculation_PSNR_SSIM(true_image, recon_image, data_range)
                PSNR_MEAN += psnr / (len(true_images))
                SSIM_MEAN += ssim / (len(true_images))

                # print(phantom, j, psnr, ssim)

                # 计算最大psnr和ssim
                # if psnr > max_psnr:
                #     max_psnr = psnr
                #     ID_recon_psnr = j
                #
                # if ssim > max_ssim:
                #     max_ssim = ssim
                #     ID_recon_ssim = j

                writer = csv.writer(f)
                writer.writerow([phantom, psnr, ssim])

            # print(phantom, ID_recon_psnr, max_psnr, ID_recon_ssim, max_ssim)

            writer = csv.writer(f)
            writer.writerow(["mean", PSNR_MEAN, SSIM_MEAN])
            Parameter[Row_ID, 1] = PSNR_MEAN
            Parameter[Row_ID, 2] = SSIM_MEAN

    # 将Parameter输出为.csv文件
    csv_name = os.path.join(csv_output_path, f"{Test_Dataset}-{mich_Type}-Mean.csv")

    with open(csv_name, mode="w", newline='') as f:
        writer = csv.writer(f)
        for i in range(Parameter.shape[0]):
            writer.writerow([Parameter[i, 0], Parameter[i, 1], Parameter[i, 2]])
