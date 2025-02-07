import os
import numpy as np


def load_data(fold_no, folders, root_dir, save_dir):
    global_min = None
    global_max = None

    # 获取所有文件路径
    for folder in folders:
        for root, dirs, files in os.walk(os.path.join(root_dir, folder)):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    try:
                        npy_data = np.load(file_path)
                        if npy_data.shape[1] < 4:
                            raise ValueError(f"Data shape {npy_data.shape} is invalid, expecting at least 4 columns.")

                        # 计算当前文件的最小值和最大值
                        local_min = np.min(npy_data, axis=0)
                        local_max = np.max(npy_data, axis=0)

                        # 更新全局最大值和最小值
                        if global_min is None:
                            global_min = local_min
                            global_max = local_max
                        else:
                            global_min = np.minimum(global_min, local_min)
                            global_max = np.maximum(global_max, local_max)

                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, f"folder{fold_no}","global_min.npy"), global_min)
    np.save(os.path.join(save_dir, f"folder{fold_no}", "global_max.npy"), global_max)

    return global_min, global_max


# 测试
if __name__ == "__main__":
    fold_no = 1
    folders = ["folder2", "folder3", "folder4", "folder5"]
    root_dir = "data"
    save_dir = "global_stats"


    global_min, global_max = load_data(fold_no, folders, root_dir, save_dir)

    print(f"Global Min: {global_min}")
    print(f"Global Max: {global_max}")
