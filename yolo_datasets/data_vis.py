import matplotlib.pyplot as plt

# 你的数据
data = {
    'pt': [17.9, 16.2, 16.8, 16.6],
    'onnx': [14.4, 14.6, 14.5, 14.4]
}

# 文本标签
categories = ['a', 'b', 'c', 'd']

# 绘制柱状图
plt.figure(figsize=(10, 6))  # 设置图形大小
bar_width = 0.35  # 柱子的宽度
index = range(len(categories))  # 索引数组

# 绘制pt和onnx的柱状图
plt.bar([i for i in index], data['pt'], bar_width, label='PT')
plt.bar([i + bar_width for i in index], data['onnx'], bar_width, label='ONNX')

# 添加一些文本标签
plt.xlabel('Category', fontsize=15)
plt.ylabel('Scores', fontsize=15)
plt.title('Scores Comparison between PT and ONNX', fontsize=20)
plt.xticks(index, categories)  # 使用文本标签

# 添加图例
plt.legend()

# 保存图形到文件
plt.savefig('bar_chart.png', format='png')  # 保存图形

# # 关闭图形，释放资源
# plt.close()