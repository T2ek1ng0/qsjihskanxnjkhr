import pickle
import numpy as np


def read_sgbest_from_pickle(filename="test_results.pkl"):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        if 'sgbest' in data:
            sgbest_data = data['sgbest']

            for problem_name, result_dict in sgbest_data.items():
                print(problem_name)

                for algo_name, list_of_subarrays in result_dict.items():
                    print(f"agent: {algo_name}")

                    if not list_of_subarrays:
                        print("(无数据)")
                        continue

                    for sub_array in list_of_subarrays:
                        if isinstance(sub_array, list):
                            converted = [
                                float(x) if isinstance(x, np.ndarray) and x.size == 1 else x
                                for x in sub_array
                            ]
                            print(converted)
                        else:
                            print(sub_array)

                    print("\n")
        else:
            print(f"错误：在文件 '{filename}' 中未找到 'sgbest' 键。")

    except FileNotFoundError:
        print(f"错误：文件 '{filename}' 不存在。请确保文件在正确的路径下。")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")


if __name__ == "__main__":
    pkl_file = r"output\test\20251018T012244_bbob-10D_easy\test_results.pkl"
    read_sgbest_from_pickle(pkl_file)
