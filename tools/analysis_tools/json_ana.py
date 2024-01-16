import re
import matplotlib.pyplot as plt

# 定义JSON文件的文件名列表
json_files = ['C:/Users/MSI/Desktop/duibi/20230228_191757.json', 'C:/Users/MSI/Desktop/duibi/20230303_094638.json', 'C:/Users/MSI/Desktop/duibi/20230305_112227.json', 'C:/Users/MSI/Desktop/duibi/20230306_212611.json']

# 初始化存储数据的列表
loss_data = []
step_data = []

# 定义正则表达式来匹配loss和step的值
loss_pattern = re.compile(r'"loss": (\d+\.\d+)')
step_pattern = re.compile(r'"step": (\d+)')

# 读取每个JSON文件
for json_file in json_files:
    with open(json_file, 'r') as file:
        # 读取JSON文件内容
        data_str = file.read()

        # 使用正则表达式匹配loss和step的值
        loss_match = loss_pattern.search(data_str)
        step_match = step_pattern.search(data_str)

        # 提取loss和step数据
        if loss_match and step_match:
            loss = float(loss_match.group(1))
            step = int(step_match.group(1))

            # 存储数据
            loss_data.append(loss)
            step_data.append(step)

# 绘制曲线图
plt.plot(step_data, loss_data, '-o')
plt.title('Loss vs. Step')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid(True)
plt.legend(json_files)
plt.show()