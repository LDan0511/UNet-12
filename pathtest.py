import os
import numpy as np  # 确保导入 numpy

# 设置文件路径
directory = r"C:\Users\hanst\Desktop\DeepLearning\newcode\UNet\rebuild_output"
filename = "PCSlice5519.dat_MC_output_50.dat"
full_path = os.path.join(directory, filename)

# 打印文件的完整路径
print("尝试读取的完整路径:", full_path)

# 检查文件是否存在并读取
if os.path.exists(full_path):
    print(f"文件路径存在: {full_path}")
    try:
        # 尝试读取文件数据
        recon_image = np.fromfile(full_path, dtype=np.float32)
        print(f"文件读取成功: {full_path}")
        print(f"文件数据（前10个元素）：{recon_image[:10]}")  # 打印前10个数据
    except Exception as e:
        print(f"文件读取失败，错误信息: {e}")
else:
    print(f"文件路径不存在: {full_path}")