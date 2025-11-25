import pickle


def parse_pkl_to_txt(input_pkl_file, output_txt_file):
    """
    解析 .pkl 文件并将其内容保存为 .txt 文件。

    参数:
        input_pkl_file (str): 输入的 .pkl 文件路径
        output_txt_file (str): 输出的 .txt 文件路径
    """
    try:
        # 打开并加载 .pkl 文件
        with open(input_pkl_file, 'rb') as pkl_file:
            data = pickle.load(pkl_file)

        # 将数据保存为 .txt 文件
        with open(output_txt_file, 'w', encoding='utf-8') as txt_file:
            # 如果数据是字典或列表，逐行写入
            if isinstance(data, dict):
                for key, value in data.items():
                    txt_file.write(f"{key}: {value}\n")
            elif isinstance(data, list):
                for item in data:
                    txt_file.write(f"{item}\n")
            else:
                # 如果是其他类型，直接写入
                txt_file.write(str(data))

        print(f"数据已成功保存到 {output_txt_file}")
    except Exception as e:
        print(f"发生错误: {e}")


# 示例用法
input_pkl_file = r'D:\GDPZero-master\outputs\gpt-4o-mini_SR_nrpa_ESC_test.pkl'  # 替换为你的 .pkl 文件路径
output_txt_file = r'D:\GDPZero-master\outputs/output.txt'  # 替换为你想要保存的 .txt 文件路径

parse_pkl_to_txt(input_pkl_file, output_txt_file)