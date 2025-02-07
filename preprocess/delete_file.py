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


# 指定文件路径
# file_to_delete = "../data/folder1/p061877/3320699_0089/8296058_1_3320699_0089_43.npy"
# file_to_delete = "../data/folder1/p061877/3320699_0052/4200440_42_3320699_0052_61.npy"


# file_to_delete = "../data/folder2/p086300/3721988_0001/1183980_1_3721988_0001_46.npy"


# file_to_delete = "../data/folder3/p068564/3951979_0023/659255_1_3951979_0023_46.npy"


# file_to_delete = "../data/folder4/p079358/3781713_0009/1423962_81_3781713_0009_40.npy"
# file_to_delete = "../data/folder4/p045703/3825747_0005/4246649_1_3825747_0005_45.npy"
# file_to_delete = "../data/folder4/p045703/3825747_0023/23344_1_3825747_0023_42.npy"
# file_to_delete = "../data/folder4/p079358/3781713_0004/2333322_1_3781713_0004_43.npy"
# file_to_delete = "../data/folder4/p045703/3825747_0005/4247524_2_3825747_0005_44.npy"
# file_to_delete = "../data/folder4/p045703/3662063_0060/142509_1_3662063_0060_37.npy"
# file_to_delete = "../data/folder4/p045703/3662063_0102/4629429_1_3662063_0102_40.npy"
# file_to_delete = "../data/folder4/p045703/3825747_0004/1943993_1_3825747_0004_39.npy"
# file_to_delete = "../data/folder4/p045703/3825747_0021/457559_40_3825747_0021_159.npy"


# file_to_delete = "../data/folder5/p043571/3841089_0010/305021_1_3841089_0010_42.npy"
# file_to_delete = "../data/folder5/p027162/3363366_0024/589405_1_3363366_0024_46.npy"
# file_to_delete = "../data/folder5/p043571/3665139_0013/168887_1_3665139_0013_41.npy"
# file_to_delete = "../data/folder5/p043571/3665139_0024/1291178_1_3665139_0024_35.npy"
# file_to_delete = "../data/folder5/p043571/3841089_0014/3319608_71_3841089_0014_45.npy"
# file_to_delete = "../data/folder5/p043571/3841089_0017/1356853_1_3841089_0017_41.npy"
# file_to_delete = "../data/folder5/p027162/3699470_0052/0_1_3699470_0052_54.npy"



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

# 使用方法




if __name__ == "__main__":
    # 从 CSV 文件中读取要删除的文件
    # csv_files = ["outlier_icp_fold1.csv", "outlier_icp_fold2.csv", "outlier_icp_fold3.csv",
    #              "outlier_icp_fold4.csv", "outlier_icp_fold5.csv"]
    #
    # deleted_files, failed_files = delete_files_from_csv(csv_files)

    # 删除指定文件
    # file_to_delete = "../data/folder1/p061877/3320699_0089/8296058_1_3320699_0089_43.npy"
    # delete_file(file_to_delete)

    delete_all_files_and_folders("../save_png")


    print("===================================== finished !!! ================================================")
