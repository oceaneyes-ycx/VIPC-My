# 筛选特定类别行并保存为新文件

# 允许的类别列表（第一栏）
allowed_categories = {'03211117', '03691459', '04401088', '02828884'}

# 输入输出文件路径
input_file = 'test_list.txt'
output_file = 'test_onlyunseen.txt'

# 读取和筛选行
with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    for line in fin:
        parts = line.strip().split('/')
        if parts and parts[0] in allowed_categories:
            fout.write(line)
