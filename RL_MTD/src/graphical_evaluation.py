import numpy as np
import matplotlib.pyplot as plt

x = [3, 4, 5, 6, 7]
y = [3, 4, 4.6, 4.8, 4.9]

x = np.array(x)
y = np.array(y)

step = 0.01
X = np.arange(x[0], x[-1], step)

exp_10 = np.vectorize(lambda xi: 10**xi)
x = exp_10(x)
X = exp_10(X)

[a, b] = np.polyfit(np.log(x), y, 1)

f = np.vectorize(lambda xi: a + b * np.log(xi))

Y = f(X)

plt.plot(x, y, 'o')
plt.plot(X, Y, '-')
plt.show()
