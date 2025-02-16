import datetime
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from matplotlib import pyplot as plt
import os
import pandas as pd


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, checkpoint_freq=10, checkpoint_save_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_save_path = checkpoint_save_path
        self.best_val_loss = float('inf')  # 初始化最佳验证损失
        self.best_epoch = 0  # 初始化最佳 epoch

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get("val_loss")

        # 如果当前的 val_loss 更好，更新最佳模型
        if current_val_loss and current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_epoch = epoch + 1
            # 保存最佳模型
            self.model.save_weights(self.filepath)
            print(f"Best model saved at epoch {self.best_epoch} with val_loss: {self.best_val_loss}")

        # 每隔 10 个 epoch 保存一次模型
        if (epoch + 1) % self.checkpoint_freq == 0:
            checkpoint_path = self.checkpoint_save_path.format(epoch=epoch + 1)
            self.model.save_weights(checkpoint_path)

        # 调用父类的 on_epoch_end 方法来进行常规处理
        super().on_epoch_end(epoch, logs)


class LossLogger(Callback):
    def __init__(self, log_file, best_epoch_file):
        super().__init__()
        self.log_file = log_file
        self.best_epoch_file = best_epoch_file
        self.best_val_loss = float('inf')  # 初始化最佳验证损失
        self.best_epoch = 0  # 初始化最佳 epoch

        # 初始化日志文件，若文件不存在，则创建
        if not os.path.exists(self.log_file):
            df = pd.DataFrame(columns=["Epoch", "Train Loss", "Val Loss"])
            df.to_csv(self.log_file, index=False)

        # 初始化最佳模型文件，若文件不存在，则创建
        if not os.path.exists(self.best_epoch_file):
            with open(self.best_epoch_file, "w") as f:
                f.write("Best Epoch, Best Val Loss\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_train_loss = logs.get('loss', float('inf'))
        current_val_loss = logs.get('val_loss', float('inf'))

        # 记录当前训练损失和验证损失
        log_data = pd.DataFrame([[epoch + 1, current_train_loss, current_val_loss]],
                                columns=["Epoch", "Train Loss", "Val Loss"])
        log_data.to_csv(self.log_file, mode='a', header=False, index=False)

        # 如果当前的验证损失更好，更新最佳模型的轮次
        if current_val_loss and current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_epoch = epoch + 1

            # 更新最佳模型轮数文件
            with open(self.best_epoch_file, "a") as f:
                f.write(f"{self.best_epoch}, {self.best_val_loss}\n")

            # print(f"New best model saved at epoch {self.best_epoch} with val_loss: {self.best_val_loss}")

    def on_train_end(self, logs=None):
        """训练结束时输出总结信息"""
        print(f"Training finished! Best model at epoch {self.best_epoch} with Val Loss: {self.best_val_loss:.6f}")


def setup_gpu(seed=42):
    """
    设置 GPU 配置并确保随机种子一致性。

    Args:
        seed (int): 随机种子，默认值为 42。
    """
    # 固定随机种子
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 设置环境变量以确保操作的确定性
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # GPU 配置
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)  # 动态分配显存
            print(f"Using GPU: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found. Using CPU.")



def plot_signals(abp, ppg, ecg, icp, file_info, idx):
    """
    Plot multichannel signals and the corresponding ICP target.
    """
    time = np.arange(0, abp.shape[0] * 0.008, 0.008)  # Assume 0.008s per time step

    plt.figure(figsize=(12, 10))

    # ABP Signal
    plt.subplot(4, 1, 1)
    plt.plot(time, abp, label="ABP", color='g')
    plt.title("ABP Signal", fontsize=16)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Normalized Value", fontsize=12)
    plt.grid()
    plt.legend()

    # PPG Signal
    plt.subplot(4, 1, 2)
    plt.plot(time, ppg, label="PPG", color='m')
    plt.title("PPG Signal", fontsize=16)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Normalized Value", fontsize=12)
    plt.grid()
    plt.legend()

    # ECG Signal
    plt.subplot(4, 1, 3)
    plt.plot(time, ecg, label="ECG", color='c')
    plt.title("ECG Signal", fontsize=16)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Normalized Value", fontsize=12)
    plt.grid()
    plt.legend()

    # ICP Signal
    plt.subplot(4, 1, 4)
    plt.plot(time, icp, label="ICP (Target)", color='b')
    plt.title("ICP Signal", fontsize=16)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Normalized Value", fontsize=12)
    plt.grid()
    plt.legend()

    plt.tight_layout()
    # plt.savefig(f"test_signal_plot_{idx}.png", dpi=300)
    plt.show()
    # print(f"Plot saved for sample {idx} - File info: {file_info}")



def get_timestamp():
    # 获取当前的日期和时间
    current_time = datetime.datetime.now()
    # 格式化为 'YYYYMMDD_HHMMSS' 格式的时间戳
    timestamp = current_time.strftime('%Y%m%d')

    return timestamp


def plot_loss(history, save_path):
    """
    绘制训练和验证的损失曲线，并保存为图片。

    Args:
    - history: 训练过程中返回的 history 对象
    - save_path: 保存路径
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 创建图形
    plt.figure()
    plt.plot(loss, linewidth=1, label='Training Loss')
    plt.plot(val_loss, linewidth=1, label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend()

    # 保存图像
    plt.savefig(save_path, dpi=600)

