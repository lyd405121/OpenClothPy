import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

price = [0.31, 0.85, 0.32]
"""
绘制水平条形图方法barh
参数一：y轴
参数二：x轴
"""
plt.barh(range(3), price, height=0.7, color='steelblue', alpha=0.8)      # 从下往上画
plt.yticks(range(3), ['explicit', 'implicit', 'verlet'])
plt.xlim(0.0,2.0)
plt.xlabel("second")
plt.title("different algtithm compute time")
for x, y in enumerate(price):
    plt.text(y + 0.2, x - 0.1, '%s' % y)
plt.show()