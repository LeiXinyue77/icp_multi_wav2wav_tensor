import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os
from tensorflow.keras.callbacks import Callback


class LossAndCheckpointLogger(Callback):
    def __init__(self, log_file, checkpoint_file):
        super().__init__()
        self.log_file = log_file
        self.checkpoint_file = checkpoint_file
        self.best_val_loss = float('inf')
        self.best_epoch = None

        # 初始化日志文件
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as file:
                file.write("Epoch,Loss,Val Loss\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get('loss', None)
        current_val_loss = logs.get('val_loss', None)

        # 追加日志
        with open(self.log_file, "a") as file:
            file.write(f"{epoch + 1},{current_loss},{current_val_loss}\n")

        # 更新最佳模型
        if current_val_loss and current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.best_epoch = epoch + 1
            with open(self.checkpoint_file, "a") as ckpt_file:
                ckpt_file.write(f"Best Model at Epoch {self.best_epoch}, Val Loss: {self.best_val_loss}\n")


    def on_train_end(self, logs=None):
        """
        训练结束时输出总结信息。
        """
        if self.best_epoch is not None:
            print(f"Training finished! Best model at epoch {self.best_epoch} with Val Loss: {self.best_val_loss:.6f}")



def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found. Using CPU.")


def plot_signals(abp, ppg, ecg, icp, file_info, idx):
    """
    Plot multi-channel signals and the corresponding ICP target.
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
    print(f"Plot saved for sample {idx} - File info: {file_info}")
