import csv
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocess.cal_beat_ave import CalBeatAve


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


def save_icp_mean_to_csv(icp_means, folder_name, root_dir):
    # 创建输出文件路径
    output_csv = os.path.join(root_dir, f'{folder_name}_icp_mean.csv')

    # 将 icp_mean 转换为 DataFrame 并保存为 CSV
    df_icp_mean = pd.DataFrame(icp_means, columns=["icp_mean"])
    df_icp_mean.to_csv(output_csv, index=False)
    print(f"ICP Mean saved to {output_csv}")


def save_histogram(icp_means, folder_name, root_dir):
    # 创建输出图片路径
    output_histogram = os.path.join(root_dir, f'{folder_name}_icp_histogram.png')

    # 绘制直方图并保存为图片
    plt.hist(icp_means, bins=50)
    plt.title(f"ICP Mean Histogram - {folder_name}")
    plt.xlabel("ICP Mean Value")
    plt.ylabel("Frequency")
    plt.savefig(output_histogram)
    plt.close()  # 关闭图形，避免重复绘图时内存占用
    print(f"Histogram saved to {output_histogram}")


def compute_label_distribution(icp_labels):
    """
    计算 icp_labels 数组中不同 label (a, b, c) 的占比，并输出。

    :param icp_labels: 包含分类标签 ('a', 'b', 'c') 的列表或数组
    :return: label 统计字典
    """
    # 统计每个标签的数量
    label_counts = Counter(icp_labels)

    # 计算占比
    total = sum(label_counts.values())  # 总数
    label_distribution = {label: count / total * 100 for label, count in label_counts.items()}  # 计算百分比

    # 输出统计信息
    print("Label Distribution:")
    for label, percentage in label_distribution.items():
        print(f"  {label}: {label_counts[label]} ({percentage:.2f}%)")

    return label_distribution



if __name__ == "__main__":

    folders = ["folde"]
    root_dir = "../data"
    icp_data, abp_data, ppg_data, ecg_data, info = load_data(folders, root_dir)

    fold_no = 1


    # 统计icp_mean
    icp_means = []
    icp_labels = []
    icp_infos = []
    for i in range(len(icp_data)):
        beat_means, label, new_info = CalBeatAve(icp_data[i], info[i], 125)
        print(f"icp_mean[{i}]: {beat_means}")
        icp_means.extend(beat_means)
        icp_labels.append(label)
        icp_infos.append(new_info)

    # 保存 icp_mean 到 CSV 文件
    save_icp_mean_to_csv(icp_means, "p061877", root_dir)

    # 保存直方图
    save_histogram(icp_means, "p061877", root_dir)

    # 保存 [new_info, label]
    with open(os.path.join(root_dir, "p061877_info_label.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File Path', 'Label'])
        for i in range(len(icp_infos)):
            writer.writerow([icp_infos[i], icp_labels[i]])

    # 统计不同 label (a, b, c) 的占比并输出
    abel_distribution = compute_label_distribution(icp_labels)



    print(f"ICP Data Shape: {icp_data.shape}")
    print(f"ABP Data Shape: {abp_data.shape}")
    print(f"PPG Data Shape: {ppg_data.shape}")
    print(f"ECG Data Shape: {ecg_data.shape}")
    print(f"Info Shape: {info.shape}")
    print(f"folder_{fold_no}: icp_max = {np.max(icp_data)}, icp_min = {np.min(icp_data)}")
    print(f"folder_{fold_no}: abp_max = {np.max(abp_data)}, abp_min = {np.min(abp_data)}")
    print(f"folder_{fold_no}: ppg_max = {np.max(ppg_data)}, ppg_min = {np.min(ppg_data)}")
    print(f"folder_{fold_no}: ecg_max = {np.max(ecg_data)}, ecg_min = {np.min(ecg_data)}")


    """
    # 找到 ICP 小于 0 或大于 55 的数据行
    outlier_indices = []
    for idx, icp in enumerate(icp_data):
        if np.any((icp < 0) | (icp > 30)):
            outlier_indices.append(idx)
    print(f"Number of files with ICP < 0 or ICP > 55: {len(outlier_indices)}")

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
    """


