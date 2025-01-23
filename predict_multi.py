import os
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from generate import DataGenerator
from model.unet2 import unet2
from helpers import setup_gpu


if __name__ == "__main__":
    # 初始化 GPU 设置
    setup_gpu()

    fold_no = 1
    print('fold_no = ', fold_no)

    # 定义文件夹和根目录
    test_folders = ["folder1"]  # fold1 的测试数据文件夹
    root_dir = "data"
    batch_size = 32

    # 创建测试数据生成器
    test_gen = DataGenerator(test_folders, root_dir, batch_size=batch_size, shuffle=False, mode='test')

    # 构建模型
    myModel = unet2()

    # 加载训练好的模型权重
    checkpoint_save_path = "./20250123_unet2_checkpoint5_1/unet2_icp.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        myModel.load_weights(checkpoint_save_path)
    else:
        print("Checkpoint not found. Please train the model and save the weights before prediction.")
        exit()

    # 执行批量预测
    print("Starting batch prediction...")
    y_pred = [] # 用于存储所有预测结果
    y_refer = []
    infos = []  # 用于存储文件信息

    abp_data = []
    ppg_data = []
    ecg_data = []

    for x_batch, y_batch, batch_info in test_gen:
        # 批量预测当前 batch
        batch_pred = myModel.predict(x_batch, batch_size=batch_size, verbose=1)
        y_pred.append(batch_pred)  # 将当前批次预测结果添加到总预测列表
        y_refer.append(y_batch)
        # x_data.append(x_batch)
        abp_data.append(x_batch[:, :, 0])
        ppg_data.append(x_batch[:, :, 1])
        ecg_data.append(x_batch[:, :, 2])
        infos.extend(batch_info)  # 保存对应的文件信息

    # 将所有预测结果拼接成完整的数组
    y_pred = np.concatenate(y_pred, axis=0)  # 拼接所有批次结果
    y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])  # 调整预测结果的形状

    y_refer = np.concatenate(y_refer, axis=0)
    y_refer = y_refer.reshape(y_refer.shape[0], y_refer.shape[1])

    abp_data = np.concatenate(abp_data, axis=0)
    abp_data = abp_data.reshape(abp_data.shape[0], abp_data.shape[1])

    ppg_data = np.concatenate(ppg_data, axis=0)
    ppg_data = ppg_data.reshape(ppg_data.shape[0], ppg_data.shape[1])

    ecg_data = np.concatenate(ecg_data, axis=0)
    ecg_data = ecg_data.reshape(ecg_data.shape[0], ecg_data.shape[1])

    # 保存预测结果为 MAT 文件
    pred_icp_file = "20250123_pred_icp_1.mat"
    io.savemat(pred_icp_file, {'pred_icp_1': y_pred})
    print(f"saved to {pred_icp_file}")

    refer_icp_file = "20250123_refer_icp_1.mat"
    io.savemat(refer_icp_file, {'refer_icp_1': y_refer})
    print(f"saved to {refer_icp_file}")

    abp_mat_file = "20250123_abp_1.mat"
    io.savemat(abp_mat_file, {'abp_1': abp_data})
    print(f"saved to {abp_mat_file}")

    ppg_mat_file = "20250123_ppg_1.mat"
    io.savemat(ppg_mat_file, {'ppg_1': ppg_data})
    print(f"saved to {ppg_mat_file}")

    ecg_mat_file = "20250123_ecg_1.mat"
    io.savemat(ecg_mat_file, {'ecg_1': ecg_data})
    print(f"saved to {ecg_mat_file}")

    info_mat_file = "20250123_info_1.mat"
    io.savemat(info_mat_file, {'info_1': infos})
    print(f"saved to {info_mat_file}")


