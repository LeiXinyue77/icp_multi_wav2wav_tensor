import os
import numpy as np
import tensorflow as tf
from helpers import plot_signals


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, folders, root_dir, batch_size=32, shuffle=True, mode='train', seed=42):
        """
        Args:
            folders (list): List of folders containing the `.npy` files.
            root_dir (str): Root directory where the data folders are stored.
            batch_size (int): Batch size for training.
            shuffle (bool): Whether to shuffle the data (applied only to train mode).
            split_ratio (float): Ratio for splitting data into train and val sets.
            mode (str): 'train', 'val', or 'test' to specify which data to load.
            seed (int): Random seed for reproducibility.
        """
        self.file_paths = []
        for folder in folders:
            folder_path = os.path.join(root_dir, folder)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.npy'):
                        self.file_paths.append(os.path.join(root, file))

        self.mode = mode
        # Sort file paths for consistency
        self.file_paths = sorted(self.file_paths)

        # Set random seed for reproducibility
        np.random.seed(seed)
        # if self.mode == 'train' or self.mode == 'val':
        if self.mode == 'train':
            np.random.shuffle(self.file_paths)

        # # Handle mode-specific file selection
        # split_index = int(len(self.file_paths) * split_ratio)
        # if self.mode == 'train':
        #     self.file_paths = self.file_paths[:split_index]
        # elif self.mode == 'val':
        #     self.file_paths = self.file_paths[split_index:]
        # elif self.mode == 'test':
        #     # Use all files for test
        #     pass
        # else:
        #     raise ValueError("Invalid mode. Use 'train', 'val', or 'test'.")

        self.batch_size = batch_size
        self.shuffle = shuffle and mode == 'train'  # Shuffle only for train mode
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        batch_files = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
        x_batch = []
        y_batch = []
        batch_info = []  # Store file info for test mode

        for file_path in batch_files:
            try:
                npy_data = np.load(file_path)
                if npy_data.shape[1] < 4:
                    raise ValueError(f"Data shape {npy_data.shape} is invalid, expecting at least 4 columns.")

                # Normalize data
                npy_data_min = np.min(npy_data, axis=0)
                npy_data_max = np.max(npy_data, axis=0)
                npy_data_range = np.where(npy_data_max - npy_data_min == 0, 1, npy_data_max - npy_data_min)
                normalized_data = (npy_data - npy_data_min) / npy_data_range

                # # Append ICP (y) and ABP (x)
                # x_batch.append(normalized_data[:, 1].reshape(-1, 1, 1))  # ABP
                # y_batch.append(normalized_data[:, 0].reshape(-1, 1, 1))  # ICP

                # Combine ABP, PPG, and ECG into multi-channel input
                multi_channel_input = np.stack([
                    normalized_data[:, 1],  # ABP
                    normalized_data[:, 2],  # PPG
                    normalized_data[:, 3]  # ECG
                ], axis=-1)  # Shape: [1024, 3]

                x_batch.append(multi_channel_input)  # Add multi-channel input
                y_batch.append(normalized_data[:, 0].reshape(-1, 1, 1))  # ICP as targe


                # Add file info for test mode
                if self.mode == 'val':
                    batch_info.append(file_path)

            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

        # Convert to numpy arrays and reshape
        x_batch = np.array(x_batch).reshape(-1, 1024, 3, 1)  # Shape: [batch_size, 1024, 3, 1]
        y_batch = np.array(y_batch)  # Shape: [batch_size, 1024, 1, 1]

        if self.mode == 'val':
            return x_batch, y_batch, batch_info
        else:
            return x_batch, y_batch

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch (only for train mode)."""
        if self.shuffle:
            np.random.shuffle(self.file_paths)


# Example usage
if __name__ == "__main__":
    folders = ["folder1"]
    root_dir = "data"

    # 固定随机种子，确保一致性
    seed = 42

    # 创建测试数据生成器 (不分割)
    val_gen = DataGenerator(folders, root_dir, batch_size=32, shuffle=False, mode='val', seed=seed)

    for x_batch, y_batch, info in val_gen:
        print(f"Test batch: x shape {x_batch.shape}, y shape {y_batch.shape}")
        print(f"File info: {info[:5]}")  # Print first 5 file paths for this batch

        for i in range(min(len(x_batch), 10)):  # 绘制前 5 个样本
            abp = x_batch[i, :, 0, 0]
            ppg = x_batch[i, :, 1, 0]
            ecg = x_batch[i, :, 2, 0]
            icp = y_batch[i]
            icp = np.squeeze(icp)
            plot_signals(abp, ppg, ecg, icp, info[i], i)
        break
