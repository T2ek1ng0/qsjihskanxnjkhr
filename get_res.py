import pickle
import os
import pandas as pd
import numpy as np

def read_sgbest_from_pickle(filename="test_results.pkl"):
    """
    读取 pickle 文件并打印 'sgbest' 键的内容，
    使得每个子数组（sub-array）单独占一行。

    Args:
        filename (str): 要读取的 pickle 文件名。
    """
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # 检查 'sgbest' 键是否存在
        if 'sgbest' in data:
            sgbest_data = data['sgbest']

            # 遍历 sgbest 中的每一个测试项目
            for problem_name, result_dict in sgbest_data.items():
                print(problem_name)

                # 结果本身也是一个字典，遍历它
                for algo_name, list_of_subarrays in result_dict.items():
                    print(f"agent: {algo_name}")

                    # 检查子数组列表是否为空
                    if not list_of_subarrays:
                        print("(无数据)")
                        continue

                    # 遍历子数组列表，并逐行打印每个子数组
                    for sub_array in list_of_subarrays:
                        if isinstance(sub_array, list):
                            converted = [
                                float(x) if isinstance(x, np.ndarray) and x.size == 1 else x
                                for x in sub_array
                            ]
                            print(converted)
                        else:
                            print(sub_array)

                print("\n")  # 在不同测试项目之间加一个空行，方便阅读
        else:
            print(f"错误：在文件 '{filename}' 中未找到 'sgbest' 键。")

    except FileNotFoundError:
        print(f"错误：文件 '{filename}' 不存在。请确保文件在正确的路径下。")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")

# 运行函数
if __name__ == "__main__":
    pkl_file = r"output\test\20251017T174609_bbob-10D_easy\test_results.pkl"
    read_sgbest_from_pickle(pkl_file)
