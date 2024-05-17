def adjust_format(test_file_path, output_file_path, adjusted_output_path):
    # 读取测试集文件，以确定每一行的长度
    with open(test_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        line_lengths = [len(line.split()) for line in lines]

    # 读取输出文件
    with open(output_file_path, 'r', encoding='utf-8') as file:
        output_labels = [line.strip() for line in file]

    # 调整输出格式，匹配测试集的行结构
    adjusted_output = []
    index = 0
    for length in line_lengths:
        adjusted_output.append(" ".join(output_labels[index:index + length]))
        index += length

    # 将调整后的输出写入新文件
    with open(adjusted_output_path, 'w', encoding='utf-8') as file:
        for line in adjusted_output:
            file.write(line + '\n')


# 调用函数，需要提供测试集路径、输出文件路径和调整后的输出文件路径
adjust_format('../data/dev.txt', 'predictions.txt', 'adjusted.txt')
