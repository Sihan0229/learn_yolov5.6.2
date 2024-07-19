import os

# 指定要更改文件扩展名的目录
directory = '/home/ai/gsh/firebox_withonline/images'

# 支持的图片格式列表
supported_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp']

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    # 检查文件扩展名是否在支持的列表中
    if any(filename.lower().endswith(ext) for ext in supported_extensions):
        # 构造原始文件的完整路径
        old_file = os.path.join(directory, filename)
        # 构造新的文件名，只保留.jpg扩展名
        new_filename = os.path.splitext(filename)[0] + '.jpg'
        new_file = os.path.join(directory, new_filename)
        # 重命名文件
        os.rename(old_file, new_file)
        print(f'Renamed "{old_file}" to "{new_file}"')

print('All supported images have been renamed to .jpg extension.')