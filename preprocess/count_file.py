import csv
import os
from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt

from preprocess.cal_beat_ave import CalBeatAve
from preprocess.load_data import load_data


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
    folders = ["p095235"]
    patient = "p095235"
    root_dir = "../data-v1/folder5"
    icp_data, abp_data, ppg_data, ecg_data, info = load_data(folders, root_dir)

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
    save_icp_mean_to_csv(icp_means, f"{patient}", root_dir)

    # 保存直方图
    save_histogram(icp_means, f"{patient}", root_dir)

    # 保存 [new_info, label]
    with open(os.path.join(root_dir, f"{patient}_info_label.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File Path', 'Label'])
        for i in range(len(icp_infos)):
            writer.writerow([icp_infos[i], icp_labels[i]])

    # 统计不同 label (a, b, c) 的占比并输出
    abel_distribution = compute_label_distribution(icp_labels)
