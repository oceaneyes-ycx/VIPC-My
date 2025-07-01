# 定义需要过滤的第一个字段的列表
filter_list = ['03211117', '03691459', '04090263', '02828884','04401088']

# 输入文件路径
input_file = 'test_list.txt'
# 输出文件路径
output_file = 'test_list_shot.txt'

# 打开输入文件和输出文件
with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    # 遍历每一行
    for line in fin:
        # 分割行为字段
        fields = line.strip().split('/')
        # 检查第一个字段是否在过滤列表中
        if fields[0] not in filter_list:
            # 如果不在过滤列表中，则写入到输出文件中
            fout.write(line)
