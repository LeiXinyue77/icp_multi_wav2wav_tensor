import os
import pandas as pd


def delete_file(target_file_path):
    """
    删除指定文件。
    :param target_file_path: 文件的完整路径
    """
    try:
        if os.path.exists(target_file_path):  # 检查文件是否存在
            os.remove(target_file_path)  # 删除文件
            print(f"File deleted: {target_file_path}")
        else:
            print(f"File not found: {target_file_path}")
    except Exception as e:
        print(f"Error occurred while deleting file: {e}")


def delete_files_from_csv(csv_files):
    """
    读取多个 CSV 文件，删除其中列出的 .npy 文件，并记录成功或失败的文件。

    :param csv_files: 包含文件路径的 CSV 文件列表
    :return: (成功删除的文件列表, 失败的文件列表)
    """
    deleted_files = []
    failed_files = []

    # 遍历每个 CSV 文件
    for csv_file in csv_files:
        try:
            # 读取 CSV 文件，假设第一列包含文件路径
            df = pd.read_csv(csv_file, header=None)  # 没有表头
            file_paths = df.iloc[:, 0].dropna().tolist()  # 获取文件路径列表

            # 遍历文件路径并删除
            for file_path in file_paths:
                file_path = file_path.strip()  # 移除空格和换行符
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        deleted_files.append(file_path)
                        print(f"File deleted: {file_path}")
                    except Exception as e:
                        failed_files.append((file_path, str(e)))
                        print(f"Error deleting {file_path}: {e}")
                else:
                    failed_files.append((file_path, "File not found"))
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    print(f"Total deleted files: {len(deleted_files)}")
    print(f"Total failed deletions: {len(failed_files)}")

    # 可选：输出失败的文件路径（仅查看前 10 个）
    if failed_files:
        print("Sample failed files:", failed_files[:10])

    return deleted_files, failed_files


def delete_all_files_and_folders(path):
    """
    删除指定路径下的所有文件和文件夹

    参数:
    path (str): 要删除的文件夹路径
    """
    # 检查路径是否存在
    if os.path.exists(path):
        # 遍历指定目录下的所有内容
        for root, dirs, files in os.walk(path, topdown=False):
            # 删除所有文件
            for name in files:
                os.remove(os.path.join(root, name))
            # 删除所有子目录
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        # 删除根文件夹本身
        os.rmdir(path)
        print(f"所有文件和文件夹已删除：{path}")
    else:
        print(f"路径不存在: {path}")


if __name__ == "__main__":
    # 从 CSV 文件中读取要删除的文件
    # csv_files = ["outliers_icp_folder1.csv", "outliers_icp_folder2.csv", "outliers_icp_folder3.csv",
    #              "outliers_icp_folder4.csv", "outliers_icp_folder5.csv"]
    #
    # deleted_files, failed_files = delete_files_from_csv(csv_files)

    # 删除指定文件
    file_to_delete = "../data-v1/folder1/p061877/3320699_0089/8296058_1_3320699_0089_43.npy"
    delete_file(file_to_delete)

    # delete_all_files_and_folders("../data")
    # delete_all_files_and_folders("../data-folder1-record")


    print("===================================== finished !!! ================================================")
