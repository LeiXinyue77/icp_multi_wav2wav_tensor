import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt


def plot_results(refer_icp, pred_icp, abp, ppg, ecg, info, idx):
    """
    绘制结果图，包括参考ICP、预测ICP、ABP、PPG、ECG信号。
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
    plt.savefig(f"result_plot_{idx}.png", dpi=300)
    plt.close()  # 保存后关闭图像
    print(f"Plot saved for sample {idx} - Info: {info}")


if __name__ == "__main__":
    pred_icp_file = "20250122_pred_icp_1.mat"
    data = io.loadmat(pred_icp_file)
    pred_icps = data['pred_icp_1']

    refer_icp_file = "20250122_refer_icp_1.mat"
    data = io.loadmat(refer_icp_file)
    refer_icps = data['refer_icp_1']

    abp_mat_file = "20250122_abp_1.mat"
    data = io.loadmat(abp_mat_file)
    abps = data['abp_1']

    info_mat_file = "20250122_info_1.mat"
    data = io.loadmat(info_mat_file)
    infos = data['info_1']

    # 遍历测试数据并绘制前 5 张图像
    max_plots = 5
    for i in range(min(len(infos), max_plots)):  # 限制最多绘制 max_plots 张图像
        refer_icp = refer_icps[i]
        pred_icp = pred_icps[i]
        abp = abps[i]
        ppg = np.zeros_like(abp)  # 使用占位符数据模拟 PPG（可替换为实际数据）
        ecg = np.zeros_like(abp)  # 使用占位符数据模拟 ECG（可替换为实际数据）
        plot_results(refer_icp, pred_icp, abp, ppg, ecg, infos[i], i)


