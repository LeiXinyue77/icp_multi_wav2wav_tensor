import os
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from helpers import get_timestamp


def plot_results(refer_icp, pred_icp, abp, ppg, ecg, info, idx, save_dir):
    """
    绘制结果图，包括参考ICP、预测ICP、ABP、PPG、ECG信号，并保存。
    """
    time = np.arange(0, refer_icp.shape[0] * 0.008, 0.008)

    plt.figure(figsize=(10, 12))

    # 绘制参考ICP和预测ICP
    plt.subplot(4, 1, 1)
    plt.plot(time, refer_icp, label="Reference ICP", linestyle='-', color='b')
    plt.plot(time, pred_icp, label="Predicted ICP", linestyle='--', color='r')
    plt.title("ICP Signal", fontsize=16)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("ICP (mmHg)", fontsize=12)
    plt.legend()
    plt.grid()

    # 绘制 ABP 信号
    plt.subplot(4, 1, 2)
    plt.plot(time, abp, label="ABP", color='g')
    plt.title("ABP Signal", fontsize=16)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Normalized Value", fontsize=12)
    plt.legend()
    plt.grid()

    # 绘制 PPG 信号
    plt.subplot(4, 1, 3)
    plt.plot(time, ppg, label="PPG", color='m')
    plt.title("PPG Signal", fontsize=16)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Normalized Value", fontsize=12)
    plt.legend()
    plt.grid()

    # 绘制 ECG 信号
    plt.subplot(4, 1, 4)
    plt.plot(time, ecg, label="ECG", color='c')
    plt.title("ECG Signal", fontsize=16)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Normalized Value", fontsize=12)
    plt.legend()
    plt.grid()

    plt.tight_layout()

    # 使用 info 作为文件名，提取路径并格式化为合法文件名
    filename = info  # 获取 info 中的路径
    # 从路径中提取所需的部分并构建文件名
    filename = filename.strip().replace('/', '_').replace(' ', '')  # 去除空格、替换斜杠为下划线

    # 去掉文件扩展名（如果有的话）
    filename = filename.split('.')[0]  # 以 '.' 为分隔符，保留 '.' 前的部分

    filename = f"{filename}.png"  # 添加 .png 扩展名

    # 保存文件到指定文件夹
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()  # 保存后关闭图像
    print(f"Plot saved for sample {idx} - Info: {info}")


def save_plots_in_batches(refer_icps, pred_icps, abps, ppgs, ecgs, infos, batch_size=100):
    """
    遍历所有数据并将图像按批次保存，每批次保存 100 张图像到一个子文件夹。
    """
    total_samples = len(infos)

    for i in range(total_samples):
        # 每 100 个样本创建一个新的文件夹
        batch_num = i // batch_size + 1
        save_dir = f"save_png/batch_{batch_num}"

        # 创建目录（如果不存在）
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 获取当前样本数据
        refer_icp = refer_icps[i]
        pred_icp = pred_icps[i]
        abp = abps[i]
        ppg = ppgs[i]
        ecg = ecgs[i]

        # 绘制并保存结果图
        plot_results(refer_icp, pred_icp, abp, ppg, ecg, infos[i], i, save_dir)


if __name__ == "__main__":
    # timestamp = get_timestamp()
    # timestamp = "20250209"
    fold_no = 1

    pred_icp_file = f"pred_icp_5_{fold_no}.mat"
    data = io.loadmat(pred_icp_file)
    pred_icps = data['pred_icp']

    refer_icp_file = f"refer_icp_5_{fold_no}.mat"
    data = io.loadmat(refer_icp_file)
    refer_icps = data['refer_icp']

    abp_mat_file = f"abp_5_{fold_no}.mat"
    data = io.loadmat(abp_mat_file)
    abps = data['abp']

    ppg_mat_file = f"ppg_5_{fold_no}.mat"
    data = io.loadmat(ppg_mat_file)
    ppgs = data['ppg']

    ecg_mat_file = f"ecg_5_{fold_no}.mat"
    data = io.loadmat(ecg_mat_file)
    ecgs = data['ecg']

    info_mat_file = f"info_5_{fold_no}.mat"
    data = io.loadmat(info_mat_file)
    infos = data['info']

    # 每 100 张图像保存到一个新的文件夹
    # save_plots_in_batches(refer_icps, pred_icps, abps, ppgs, ecgs, infos, batch_size=100)
    save_dir = "save_png"
    for idx in range(len(pred_icps)):
        plot_results(refer_icps[idx], pred_icps[idx], abps[idx], ppgs[idx], ecgs[idx], infos[idx], idx, save_dir)
        print(f"{infos[idx]}")

