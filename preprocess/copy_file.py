import os
import shutil


def copy_all_files_and_folders(source_folder, target_folder):
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    # 遍历源文件夹的所有文件和子目录
    for root, dirs, files in os.walk(source_folder):
        # 计算目标路径
        relative_path = os.path.relpath(root, source_folder)
        target_path = os.path.join(target_folder, relative_path)

        # 创建对应的目标文件夹
        os.makedirs(target_path, exist_ok=True)

        # 复制文件
        for file in files:
            source_file = os.path.join(root, file)
            destination_file = os.path.join(target_path, file)
            shutil.copy2(source_file, destination_file)

    print(f"所有文件和子目录已拷贝到 {target_folder}")


if __name__ == "__main__":
    source_folder = "../data-v0"
    target_folder = "../data-v1"
    copy_all_files_and_folders(source_folder, target_folder)
