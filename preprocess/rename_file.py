import os
import re
import shutil


def rename_and_organize_files(root_directory):
    """
    遍历 root_directory 下的所有 `folderi/pxxnnnn` 目录，并对其中的 .npy 文件进行重命名和归类。
    """
    # 遍历 `../data/folderi/pxxnnnn` 结构
    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)

        # 确保是一个目录
        if not os.path.isdir(folder_path):
            continue

        # 进入 `folderi` 目录，查找 `pxxnnnn` 子目录
        for sub_folder_name in os.listdir(folder_path):
            sub_folder_path = os.path.join(folder_path, sub_folder_name)

            # 确保是 `pxxnnnn` 目录
            if not os.path.isdir(sub_folder_path):
                continue

            print(f"Processing folder: {sub_folder_path}")

            # 获取该目录下所有 .npy 文件
            files = os.listdir(sub_folder_path)

            # 使用正则表达式提取文件的相关信息
            file_info = []
            for file in files:
                match = re.match(r'(\d+_\d+)_(\d+)_(\d+)_(\d+)\.npy', file)
                if match:
                    patient_id = match.group(1)  # 病人标识
                    index = match.group(2)  # index
                    start_sample = int(match.group(3))  # 采样点数
                    icp_ecg_shift = match.group(4)  # ICP 相对于 ECG 信号的相移
                    file_info.append((file, patient_id, index, start_sample, icp_ecg_shift))

            # 按照采样点数排序文件
            file_info.sort(key=lambda x: x[3])  # 按采样点数排序

            # 为文件重新命名并组织到病人标识的文件夹中
            for i in range(0, len(file_info)):
                curr_file = file_info[i]

                # 根据新的命名规则生成文件名
                new_name = f"{curr_file[3]}_{curr_file[2]}_{curr_file[1]}_{curr_file[4]}.npy"

                # 获取病人标识目录路径
                patient_folder = os.path.join(sub_folder_path, curr_file[1])  # 病人标识文件夹
                if not os.path.exists(patient_folder):  # 如果文件夹不存在，创建文件夹
                    os.makedirs(patient_folder)

                # 构造旧文件路径和新文件路径
                old_file_path = os.path.join(sub_folder_path, curr_file[0])
                new_file_path = os.path.join(patient_folder, new_name)

                # 重命名并移动文件到相应的文件夹
                shutil.move(old_file_path, new_file_path)
                print(f"Moved and Renamed: {curr_file[0]} -> {new_name} in folder {curr_file[1]}")


def rename_files_in_folder(root_dir):
    """
    遍历指定文件夹及其子文件夹中的 .npy 文件，并重命名去掉末尾的 '_a'、'_b' 或 '_c'。

    :param root_dir: 根目录路径，指定文件夹
    """
    # 遍历指定文件夹及子文件夹
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.npy'):
                # 获取文件路径
                file_path = os.path.join(root, file)

                # 检查文件名是否以 '_a', '_b', '_c' 结尾，且确保文件末尾是 '.npy'
                if file[-6:] in ['_a.npy', '_b.npy', '_c.npy']:  # 检查倒数5个字符（_a.npy, _b.npy, _c.npy）
                    new_name = file[:-6] + '.npy'  # 去掉最后的 '_a'、'_b' 或 '_c' 和 '.npy'

                else:
                    continue  # 如果文件名没有 '_a', '_b', '_c' 结尾，则跳过

                # 获取新的文件路径
                new_file_path = os.path.join(root, new_name)

                # 重命名文件
                os.rename(file_path, new_file_path)

                print(f"Renamed: {file_path} → {new_file_path}")


if __name__ == "__main__":
    # rename_and_organize_files("../data-v1")
    # 设置根目录路径
    root_dir = "../data-v1/folder2/p087913"
    # 调用函数进行文件重命名
    rename_files_in_folder(root_dir)
    print("===================================== finished !!! ================================================")
