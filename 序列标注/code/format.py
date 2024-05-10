def adjust_output_format(source_file, output_file, adjusted_output_file):
    # 读取源文件
    with open(source_file, 'r', encoding='utf-8') as file:
        source_data = file.readlines()

    # 读取输出文件
    with open(output_file, 'r', encoding='utf-8') as file:
        output_data = [line.strip() for line in file.readlines()]

    # 准备调整格式后的输出
    adjusted_output = []
    output_index = 0

    for line in source_data:
        source_line = line.strip()
        if not source_line:
            # 如果源数据行是空的，直接添加一个空行到调整后的输出
            adjusted_output.append('\n')
        else:
            # 否则，根据源数据行中的元素数量调整输出数据
            tokens = source_line.split()
            adjusted_line = ' '.join(output_data[output_index:output_index + len(tokens)])
            adjusted_output.append(adjusted_line + '\n')
            output_index += len(tokens)

    # 写入调整后的输出到新文件
    with open(adjusted_output_file, 'w', encoding='utf-8') as file:
        file.writelines(adjusted_output)


# 使用示例
source_file_path = '../data/dev_TAG.txt'  # 源数据文件路径
output_file_path = 'predictions_epoch_1_batch_6000.txt'  # 当前输出数据文件路径
adjusted_output_file_path = 'adjusted_output.txt'  # 调整后的输出数据文件路径

adjust_output_format(source_file_path, output_file_path, adjusted_output_file_path)
