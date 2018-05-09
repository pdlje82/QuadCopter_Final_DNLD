import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!


def reward(x, z):
    reward = 1. - .3 * (abs(np.array([x, z]) - np.array([0, 20])).sum())
    return reward

print(reward(0, 20))

fig = plt.figure()
ax = Axes3D(fig)
n = 10
xs = [i for i in range(n) for _ in range(n)]
zs = range(n) * n
rs = [reward(x, z) for x,z in zip(xs,zs)]

ax.scatter(xs, zs, rs)

ax.set_xlabel('X Label')
ax.set_ylabel('Z Label')
ax.set_zlabel('Reward Label')

plt.show()
