# -*- coding: utf-8 -*-
import os
import shutil

# 定义原始数据集路径和训练集、验证集的子文件夹路径
original_dataset_path = '/home/ai/gsh/firebox_withonline/images'
train_dataset_path = os.path.join(original_dataset_path, 'train')
val_dataset_path = os.path.join(original_dataset_path, 'val')

# 如果训练集和验证集文件夹不存在，则创建它们
os.makedirs(train_dataset_path, exist_ok=True)
os.makedirs(val_dataset_path, exist_ok=True)

# 获取原始数据集中所有图片文件的列表，并按文件名排序
image_files = [f for f in os.listdir(original_dataset_path) if f.endswith('.jpg')]
image_files.sort()

# 遍历排序后的文件列表，每十张图片分组
for i in range(0, len(image_files), 10):
    # 计算每组图片的索引范围
    start_index = i
    end_index = min(i + 9, len(image_files))  # 确保不会超出列表索引

    # 复制前9张图片到训练集文件夹
    for j in range(start_index, end_index):
        shutil.copy(os.path.join(original_dataset_path, image_files[j]), train_dataset_path)

    # 如果存在第10张图片，复制到验证集文件夹
    if end_index < len(image_files):
        shutil.copy(os.path.join(original_dataset_path, image_files[end_index]), val_dataset_path)

print("训练集和验证集图片已保存到指定文件夹。")