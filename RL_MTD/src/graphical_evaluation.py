import numpy as np
import matplotlib.pyplot as plt


x_log = np.array([3, 4, 5, 6, 7])
y = np.array([3000, 4000, 4600, 4800, 4900])

exp_10 = np.vectorize(lambda xi: 10**xi)

x = exp_10(x_log)

plt.plot(x, y, 'o')
plt.show()

"""
step = 0.5
x = np.array([1, 7, 20, 50, 79])
X = np.arange(x[0], x[-1], step)
y = np.array([10, 19, 30, 35, 51])
[a, b] = np.polyfit(np.log2(x), y, 1)
f = np.vectorize(lambda xi: a + b * np.log2(xi))
Y = f(X)

plt.plot(x, y, 'o')
plt.plot(X, Y, '-')
plt.show()
"""