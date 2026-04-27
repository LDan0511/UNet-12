import glob
import os
import Setting
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image

DATASET = "MultiShape"
REAL_MICH_SIZE = (155, 78)
VIRTUAL_MICH_SIZE = (155, 78)


class MyDataset(Dataset):
    def __init__(self, root_dir, real_size, virtual_size, real_area, virtural_area, virtual_mask, total):
        data_name_list = glob.glob(os.path.join(root_dir, 'data', '*.dat'))
        lab_name_list = glob.glob(os.path.join(root_dir, 'label', '*.img'))
        assert len(data_name_list) == len(lab_name_list)

        self.data_name_list = sorted(data_name_list)
        self.label_name_list = sorted(lab_name_list)

        self.total = total
        self.real_length = real_size[1] // 2
        self.virtual_length = virtual_size[1] // 2
        self.real_size = real_size
        self.virtual_size = virtual_size

        self.real_area = real_area
        self.virtual_area = virtural_area
        self.virtual_mask = virtual_mask

    def __len__(self):
        return len(self.data_name_list)

    def __getitem__(self, idx):
        # ---- 文件名匹配（.png 与 .true）----
        if not Setting.SIMULATION:
            data_stem = os.path.splitext(os.path.basename(self.data_name_list[idx]))[0]
            label_stem = os.path.splitext(os.path.basename(self.label_name_list[idx]))[0]
            assert data_stem == label_stem, f"Pair mismatch: {data_stem} vs {label_stem}"

        # ---- 1) 无条件读取：先把 data/label 定义出来 ----
        data1d = np.fromfile(self.data_name_list[idx], dtype="<f4")

        label1d = np.fromfile(self.label_name_list[idx], dtype="<f4")  # 1D

        # 清理 NaN/Inf
        data1d = np.nan_to_num(data1d, nan=0.0, posinf=0.0, neginf=0.0)
        label1d = np.nan_to_num(label1d, nan=0.0, posinf=0.0, neginf=0.0)

        # 归一化（防止除0）
        TARGET_H, TARGET_W = 78, 155
        DATA_H, DATA_W = 78, 155

        if data1d.size != DATA_H * DATA_W:
            raise ValueError(f"Data length={data1d.size}, expected {DATA_H * DATA_W} for ({DATA_H},{DATA_W}).")
        data2d = data1d.reshape((DATA_H, DATA_W)).astype(np.float32, copy=False)

        if label1d.size != TARGET_H * TARGET_W:
            raise ValueError(
                f"Label length={label1d.size}, expected {TARGET_H * TARGET_W} for ({TARGET_H},{TARGET_W}).")
        label2d = label1d.reshape((TARGET_H, TARGET_W)).astype(np.float32, copy=False)

        if (not Setting.POISSON) or Setting.SIMULATION:
            m = float(data2d.mean())
            if abs(m) > 1e-12:
                data2d = data2d / m
            ml = float(label2d.mean())
            if abs(ml) > 1e-12:
                label2d = label2d / ml

        # ---- 2) 只支持 total=True：输出固定 [1,104,207] ----
        # 你现在用 png 做输入，不可能再走 real_area 那套 1D 切块逻辑
        # if not self.total:
        #     raise RuntimeError("png input mode requires total=True. Please set Setting.TOTAL=True.")
        #
        # TARGET_H, TARGET_W = 160, 160
        #
        # # # png resize 对齐
        # # if data2d.shape != (TARGET_H, TARGET_W):
        # #     data2d = np.asarray(Image.fromarray(data2d).resize((TARGET_W, TARGET_H)), dtype=np.float32)
        #
        # # true reshape 对齐
        # if label1d.size != TARGET_H * TARGET_W:
        #     raise ValueError(f"Label length={label1d.size}, expected {TARGET_H * TARGET_W} for (104,207).")
        #
        # label2d = label1d.reshape((TARGET_H, TARGET_W)).astype(np.float32, copy=False)

        # ---- 3) 变成 [1,H,W] 返回 ----
        data = np.expand_dims(data2d, axis=0)
        label = np.expand_dims(label2d, axis=0)
        return data, label


if __name__ == "__main__":
    blocks1 = [0, 1, 2, 3, 0, 1, 2, 3]
    blocks2 = [1, 2, 3, 0, 2, 3, 0, 1]

    real_area = []
    virtual_area = []
    virtual_mask = []

    # for block1, block2 in zip(blocks1, blocks2):
    #     real_file = open(f'area/real_block{block1}_block{block2}.dat', 'rb')
    #     real_data = np.fromfile(real_file, '<i4', -1)
    #     real_data = torch.LongTensor(real_data)
    #     real_data.unsqueeze(0)
    #     real_file.close()
    #     real_area.append(real_data)
    #
    #     virtual_file = open(f'area/virtual_block{block1}_block{block2}.dat', 'rb')
    #     virtual_data = np.fromfile(virtual_file, '<i4', -1)
    #     virtual_data = torch.LongTensor(virtual_data)
    #     virtual_data.unsqueeze(0)
    #     virtual_file.close()
    #     virtual_area.append(virtual_data)

    # file_name = r'area/virtual.dat'
    # with open(file_name, 'rb') as f:
    #     virtual_mask = np.fromfile(f, '<f4', -1).reshape((104, 207))

    model_name = DATASET
    dataset_dir = os.path.join('datasets', model_name, 'test' + os.sep)
    print(dataset_dir)
    train_dataset = MyDataset(root_dir=dataset_dir, real_size=REAL_MICH_SIZE,
                              virtual_size=VIRTUAL_MICH_SIZE, real_area=real_area, virtural_area=virtual_area,
                              virtual_mask=virtual_mask, total=False)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    for idx, (area_data, area_label) in enumerate(train_dataloader):
        if idx == 0:
            print(train_dataset.data_name_list[idx])
            # save_image(area_data[:, 1, :, :], f"x{idx}-1.png", nrow=1, normalize=True, range=(0, 1), cmap='gray')
