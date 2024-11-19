import random

def process_dataset(input_file, output_file):
    # 用于存储物体的分类信息
    object_dict = {}

    # 读取文件内容
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # 按照类别和物体编号整理数据
    for line in lines:
        line = line.strip()
        if not line:
            continue
        category, object_id, angle = line.split('/')
        key = f"{category}/{object_id}"
        if key not in object_dict:
            object_dict[key] = []
        object_dict[key].append(angle)

    # 随机选择一个角度
    selected_lines = []
    for key, angles in object_dict.items():
        selected_angle = random.choice(angles)
        selected_lines.append(f"{key}/{selected_angle}")

    # 写入新的文件
    with open(output_file, 'w') as file:
        file.write("\n".join(selected_lines))
    
    print(f"随机选择完成，新文件已保存为 {output_file}")

# 文件路径
input_file = "train_list.txt"
output_file = "train_list_contrasive.txt"

# 处理数据集
process_dataset(input_file, output_file)
