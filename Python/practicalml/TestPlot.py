import numpy as np
import  matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

X = np.array([[[1, 11, 111, 1111],
               [2, 22, 222, 2222],
               [3, 33, 333, 3333]],
              [[4, 44, 444, 4444],
               [5, 55, 555, 5555],
               [6, 66, 666, 6666]]])
print(X.shape)
print(X[:,-1,1])
exit()

print(plt.get_backend())
print(plt.get_backend())
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)
plt.plot(t, s)

plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid(True)
plt.show()
plt.savefig('plt.png')