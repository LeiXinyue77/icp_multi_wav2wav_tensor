import os
import numpy as np
import matplotlib.pyplot as plt


def load_data(folders, root_dir, seed=42):
    icp_data = []
    abp_data = []
    ppg_data = []
    ecg_data = []
    info = []

    # 获取所有文件路径
    for folder in folders:
        for root, dirs, files in os.walk(os.path.join(root_dir, folder)):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    info.append(file_path)

    info.sort()
    np.random.seed(seed)

    # 按照固定顺序加载数据
    for file_path in info:
        try:
            npy_data = np.load(file_path)
            if npy_data.shape[1] < 4:
                raise ValueError(f"Data shape {npy_data.shape} is invalid, expecting at least 4 columns.")

            # Normalize data
            npy_data_min = np.min(npy_data, axis=0)
            npy_data_max = np.max(npy_data, axis=0)
            npy_data_range = np.where(npy_data_max - npy_data_min == 0, 1, npy_data_max - npy_data_min)
            normalized_data = (npy_data - npy_data_min) / npy_data_range

            # Extract columns as separate data
            icp_data.append(normalized_data[:, 0])  # First column: ICP
            abp_data.append(normalized_data[:, 1])  # Second column: ABP
            ppg_data.append(normalized_data[:, 2])  # Third column: PPG
            ecg_data.append(normalized_data[:, 3])  # Fourth column: ECG

        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    # Convert to numpy arrays
    icp_data = np.array(icp_data, dtype=object)
    abp_data = np.array(abp_data, dtype=object)
    ppg_data = np.array(ppg_data, dtype=object)
    ecg_data = np.array(ecg_data, dtype=object)
    info = np.array(info, dtype=object)

    return icp_data, abp_data, ppg_data, ecg_data, info


# 测试
if __name__ == "__main__":
    # folders = ["folder2", "folder3", "folder4", "folder5"]
    folders = ["folder1"]
    root_dir = "data"

    icp_data, abp_data, ppg_data, ecg_data, info = load_data(folders, root_dir)

    # 打印结果
    print(f"ICP Data Shape: {icp_data.shape}")
    print(f"ABP Data Shape: {abp_data.shape}")
    print(f"PPG Data Shape: {ppg_data.shape}")
    print(f"ECG Data Shape: {ecg_data.shape}")
    print(f"Info Shape: {info.shape}")

    # 遍历所有文件数据并绘制图像
    for idx in range(len(info)):
        plt.figure(figsize=(10, 12))  # 设置画布大小

        # 绘制 ICP 数据
        plt.subplot(4, 1, 1)
        plt.plot(icp_data[idx], label="ICP", color='b')
        plt.title("ICP Data")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()

        # 绘制 ABP 数据
        plt.subplot(4, 1, 2)
        plt.plot(abp_data[idx], label="ABP", color='g')
        plt.title("ABP Data")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()

        # 绘制 PPG 数据
        plt.subplot(4, 1, 3)
        plt.plot(ppg_data[idx], label="PPG", color='r')
        plt.title("PPG Data")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()

        # 绘制 ECG 数据
        plt.subplot(4, 1, 4)
        plt.plot(ecg_data[idx], label="ECG", color='m')
        plt.title("ECG Data")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()

        # 添加整体标题和调整布局
        plt.suptitle(f"File: {info[idx]}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 保存或显示图像
        # plt.savefig(f"{info[idx].split('/')[-1]}_plot.png")  # 保存为文件
        plt.show()  # 或显示图像
        print(f"{info[idx]}")


