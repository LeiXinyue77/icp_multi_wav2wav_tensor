from wfdb.processing import find_local_peaks
import numpy as np
import os


def rename_file_with_label(info, label):
    """
    根据 label 修改 info 对应的文件名，并重命名文件。

    :param info: 原始文件路径，例如 '../data/folder1/p061877/3320699_0046/21223_2_3320699_0046_46.npy'
    :param label: 需要添加的分类标签（如 'a', 'b', 'c'）
    :return: 新文件路径
    """
    # 确保 info 是字符串，避免 numpy 类型错误
    info = str(info).strip()

    # 获取目录路径和原始文件名
    dir_path, original_filename = os.path.split(info)  # 分离目录和文件名
    filename_no_ext, ext = os.path.splitext(original_filename)  # 分离文件名和扩展名 (.npy)

    # 生成新文件名，插入 label
    new_filename = f"{filename_no_ext}_{label}{ext}"

    # 生成新文件完整路径
    new_file_path = os.path.join(dir_path, new_filename)

    # 进行文件重命名
    try:
        os.rename(info, new_file_path)
        print(f"Renamed: {info} → {new_file_path}")
    except FileNotFoundError:
        print(f"File not found: {info}")
    except Exception as e:
        print(f"Error renaming file {info}: {e}")

    return new_file_path


def period_autocorrelation(sig, freq):
    """
    利用自相关函数计算ICP信号的主周期

    :param icp: 输入的ICP信号（numpy.ndarray）
    :param fs: 采样频率（Hz）
    :return period: 信号的主周期（秒）
    """

    # 1. 计算自相关
    sig = sig - np.mean(sig)  # 去均值
    autocorr = np.correlate(sig, sig, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # 保留非负延迟部分

    # 2. 寻找主周期
    # 找到自相关的第一个局部最大值（跳过0延迟点）
    peaks = np.where((autocorr[1:-1] > autocorr[:-2]) & (autocorr[1:-1] > autocorr[2:]))[0] + 1
    if len(peaks) == 0:
        # print("未找到周期性峰值")
        return None

    # 第一个峰值对应的延迟时间
    dominant_lag = peaks[0]
    period = dominant_lag / freq  # 将延迟转换为时间

    # 可视化自相关函数
    # plt.figure(figsize=(10, 5))
    # plt.plot(autocorr, label="Autocorrelation")
    # plt.axvline(dominant_lag, color='r', linestyle='--', label=f"Dominant Lag: {dominant_lag} samples")
    # plt.title("Autocorrelation Function")
    # plt.xlabel("Lag (samples)")
    # plt.ylabel("Autocorrelation")
    # plt.legend()
    # plt.grid()
    # plt.show()

    return period


def CalBeatAve(icp, info, fs):
    icp_period = period_autocorrelation(icp, fs)
    # 检测峰值和谷值
    radius = int(0.6*icp_period * fs)
    peaks = find_local_peaks(icp, radius)
    troughs = find_local_peaks(-icp, radius)

    valid_peaks = []
    valid_troughs = []
    trough_ptr = 0  # 谷值的指针

    for peak_idx in peaks:
        # 找到第一个谷值索引大于当前峰值索引的谷值
        while trough_ptr < len(troughs) and troughs[trough_ptr] <= peak_idx:
            trough_ptr += 1

        # 如果找到的谷值满足条件
        if trough_ptr < len(troughs):
            trough_idx = troughs[trough_ptr]

            # 检查 (troughs - peaks)
            if trough_idx - peak_idx > 1.2 * icp_period * fs:
                continue  # 跳过当前峰值

            # 保存当前峰值和谷值
            valid_peaks.append(peak_idx)
            valid_troughs.append(trough_idx)

    # 计算 ave_icp
    beat_means = []
    for j in range(len(valid_peaks)):
        peak_idx = valid_peaks[j]
        trough_idx = valid_troughs[j]

        peak_icp = icp[peak_idx]
        trough_icp = icp[trough_idx]

        # 按公式计算 ave_icp
        ave_icp = (peak_icp - trough_icp) / 3 + trough_icp
        beat_means.append(ave_icp)

    # beat_means中存储了每个心跳的平均ICP值, 如果每个心跳的值均小于20，则标记为"normal"类(a)，
    # beat_means中某个心跳大于等20小于40，则标记为“high”类（b）
    # beat_means中某个心跳的值大于等于40，则标记为“ultra-high”类（c）
    # 初始化默认标签
    label = "a"
    # 如果存在大于等于 40 的值，直接归为 "ultra-high"（c）
    if any(ave_icp >= 40 for ave_icp in beat_means):
        label = "c"
    # 如果没有大于等于 40 的值，但有 20 <= ICP < 40 的值，归为 "high"（b）
    elif any(20 <= ave_icp < 40 for ave_icp in beat_means):
        label = "b"
    # 如果所有 ICP 值均小于 20，归为 "normal"（a）
    else:
        label = "a"

    # 根据 label 修改 info 对应的文件名
    # info = '../data/folder1/p061877/3320699_0046/21223_2_3320699_0046_46.npy'
    new_info = rename_file_with_label(info, label)

    return beat_means, label, new_info





