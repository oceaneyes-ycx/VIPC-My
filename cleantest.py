import random

def reduce_txt_file(input_file, output_file, keep_ratio=0.1):
    # 读取所有非空行
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    total_lines = len(lines)
    keep_count = max(1, int(total_lines * keep_ratio))  # 至少保留1行

    # 随机选择要保留的行
    reduced_lines = random.sample(lines, keep_count)

    # 写入新的文件
    with open(output_file, 'w') as f:
        for line in reduced_lines:
            f.write(line + '\n')

    print(f"原始行数: {total_lines}, 保留行数: {keep_count}")
    print(f"输出文件已保存至: {output_file}")

# 示例用法
if __name__ == "__main__":
    reduce_txt_file("train_list.txt", "train_reduced.txt")
