import numpy as np
import  matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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