import numpy as np
import matplotlib.pyplot as plt
 
#定义x、y散点坐标
x    = [0.1, 1.0, 10.0, 50.0, 100.0]
num1 = [0.02,0.0005,0.0005,0.0005,0.0005]
num2 = [0.2,0.2,0.2,0.2,0.2]
num3 = [0.2,0.2,0.1,0.05,0.02]

y1 = np.array(num1)
y2 = np.array(num2)
y3 = np.array(num3)

#用3次多项式拟合
f1 = np.polyfit(x, y1, 3)
p1 = np.poly1d(f1)
yvals1 = p1(x)  #拟合y值
 
f2 = np.polyfit(x, y2, 3)
p2 = np.poly1d(f2)
yvals2 = p2(x)

f3 = np.polyfit(x, y3, 3)
p3 = np.poly1d(f3)
yvals3 = p3(x)

fig1, ax1 = plt.subplots()
ax1.set_xscale("log")
ax1.set_xlim(1e-1, 1e2)



#绘图
plot1 = plt.plot(x, y1, 's')
plot2 = plt.plot(x, yvals1, 'r')
plot3 = plt.plot(x, y2, 's')
plot4 = plt.plot(x, yvals2, 'g')
plot5 = plt.plot(x, y3, 's')
plot6 = plt.plot(x, yvals3, 'b')

plt.text(0.2,0.02,'explicit',fontsize=10)
plt.text(15.0,0.205,'implicit',fontsize=10)
plt.text(10,0.12,'verlet',fontsize=10)

plt.xlabel('spring factor')
plt.ylabel('max △t')
plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
plt.title('polyfitting')


plt.show()
