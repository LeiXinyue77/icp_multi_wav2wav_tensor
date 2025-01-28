import os
import re
import shutil


def rename_and_organize_files(directory):
    files = os.listdir(directory)

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
    for i in range(1, len(file_info)):
        curr_file = file_info[i]

        # 根据新的命名规则生成文件名
        new_name = f"{curr_file[3]}_{curr_file[2]}_{curr_file[1]}_{curr_file[4]}.npy"

        # 获取病人标识目录路径
        patient_folder = os.path.join(directory, curr_file[1])  # 病人标识文件夹
        if not os.path.exists(patient_folder):  # 如果文件夹不存在，创建文件夹
            os.makedirs(patient_folder)

        # 构造旧文件路径和新文件路径
        old_file_path = os.path.join(directory, curr_file[0])
        new_file_path = os.path.join(patient_folder, new_name)

        # 重命名并移动文件到相应的文件夹
        shutil.move(old_file_path, new_file_path)
        print(f"Moved and Renamed: {curr_file[0]} -> {new_name} in folder {curr_file[1]}")


rename_and_organize_files(directory="data/folder1/p061877")
