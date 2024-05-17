def calculate_accuracy(pred_file_path, gold_file_path):
    # 读取预测文件和标准答案文件
    with open(pred_file_path, 'r', encoding='utf-8') as pred_file, \
            open(gold_file_path, 'r', encoding='utf-8') as gold_file:
        pred_lines = pred_file.readlines()
        gold_lines = gold_file.readlines()

    total_tokens = 0
    correct_tokens = 0

    # 比较每一行的预测结果和标准答案
    for pred_line, gold_line in zip(pred_lines, gold_lines):
        pred_tags = pred_line.strip().split()
        gold_tags = gold_line.strip().split()

        # 比较每个标记的预测和真实标签
        for pred_tag, gold_tag in zip(pred_tags, gold_tags):
            if pred_tag == gold_tag:
                correct_tokens += 1
            total_tokens += 1

    # 计算准确率
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    return accuracy


# 调用函数
pred_file_path = 'adjusted.txt'
gold_file_path = '../data/dev_TAG.txt'
accuracy = calculate_accuracy(pred_file_path, gold_file_path)
print(f"Accuracy: {accuracy:.2%}")
