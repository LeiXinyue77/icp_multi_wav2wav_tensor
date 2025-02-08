import csv
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

            # # Normalize data
            # npy_data_min = np.min(npy_data, axis=0)
            # npy_data_max = np.max(npy_data, axis=0)
            # npy_data_range = np.where(npy_data_max - npy_data_min == 0, 1, npy_data_max - npy_data_min)
            # normalized_data = (npy_data - npy_data_min) / npy_data_range

            # Extract columns as separate data
            icp_data.append(npy_data[:, 0])  # First column: ICP
            abp_data.append(npy_data[:, 1])  # Second column: ABP
            ppg_data.append(npy_data[:, 2])  # Third column: PPG
            ecg_data.append(npy_data[:, 3])  # Fourth column: ECG

        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    # Convert to numpy arrays
    icp_data = np.array(icp_data, dtype=object)
    abp_data = np.array(abp_data, dtype=object)
    ppg_data = np.array(ppg_data, dtype=object)
    ecg_data = np.array(ecg_data, dtype=object)
    info = np.array(info, dtype=object)

    return icp_data, abp_data, ppg_data, ecg_data, info


# 找到 ICP 小于 0 或大于 55 的文件并写入 CSV
def save_outliers_to_csv(icp_data, info, output_csv):
    outlier_files = []

    # 遍历所有数据，检查 ICP 是否小于 0 或大于 55
    for idx, icp in enumerate(icp_data):
        if np.any((icp < 0) | (icp > 55)):  # ICP 小于 0 或大于 55
            outlier_files.append(info[idx])  # 保存文件名

    # 将文件名保存到 CSV 文件
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File Path'])  # 写入表头
        for file_path in outlier_files:
            writer.writerow([file_path])  # 写入每个文件的路径

    print(f"Outlier files saved to {output_csv}")



if __name__ == "__main__":

    folders = ["p095235"]
    root_dir = "../data-v1/folder5"
    icp_data, abp_data, ppg_data, ecg_data, info = load_data(folders, root_dir)

    fold_no = 5

    # save_outliers_to_csv(icp_data, info, f"outliers_icp_folder{fold_no}.csv")

    # # 统计icp_mean
    # icp_means = []
    # icp_labels = []
    # icp_infos = []
    # for i in range(len(icp_data)):
    #     beat_means, label, new_info = CalBeatAve(icp_data[i], info[i], 125)
    #     print(f"icp_mean[{i}]: {beat_means}")
    #     icp_means.extend(beat_means)
    #     icp_labels.append(label)
    #     icp_infos.append(new_info)
    #
    # # 保存 icp_mean 到 CSV 文件
    # save_icp_mean_to_csv(icp_means, "p061877", root_dir)
    #
    # # 保存直方图
    # save_histogram(icp_means, "p061877", root_dir)
    #
    # # 保存 [new_info, label]
    # with open(os.path.join(root_dir, "p061877_info_label.csv"), mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['File Path', 'Label'])
    #     for i in range(len(icp_infos)):
    #         writer.writerow([icp_infos[i], icp_labels[i]])
    #
    # # 统计不同 label (a, b, c) 的占比并输出
    # abel_distribution = compute_label_distribution(icp_labels)



    print(f"ICP Data Shape: {icp_data.shape}")
    print(f"ABP Data Shape: {abp_data.shape}")
    print(f"PPG Data Shape: {ppg_data.shape}")
    print(f"ECG Data Shape: {ecg_data.shape}")
    print(f"Info Shape: {info.shape}")
    print(f"folder{fold_no}/{folders}: icp_max = {np.max(icp_data)}, icp_min = {np.min(icp_data)}")
    print(f"folder{fold_no}/{folders}: abp_max = {np.max(abp_data)}, abp_min = {np.min(abp_data)}")
    print(f"folder{fold_no}/{folders}: ppg_max = {np.max(ppg_data)}, ppg_min = {np.min(ppg_data)}")
    print(f"folder{fold_no}/{folders}: ecg_max = {np.max(ecg_data)}, ecg_min = {np.min(ecg_data)}")



    # # 找到 ICP 小于 0 或大于 55 的数据行
    outlier_indices = []
    # for idx, icp in enumerate(icp_data):
    #     if np.any((icp < 0) | (icp > 30)):
    #         outlier_indices.append(idx)
    # #
    # for idx, abp in enumerate(abp_data):
    #     if np.any((abp < 0) | (abp > 200)):
    #         outlier_indices.append(idx)

    # 遍历所有符合条件的文件数据并绘制图像
    for idx in outlier_indices:
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



