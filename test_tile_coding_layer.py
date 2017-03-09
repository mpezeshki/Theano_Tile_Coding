import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from layers import get_tile_coder


# x.shape: (num_features)
def target_function(x):
    return np.sin(x[0]) + np.cos(x[1]) + 0.01 * np.random.randn()


def get_dataset(num_samples, min_val, max_val):
    # num_features is assumed to be 2 for this example.
    # However it can take any value in general.
    dataset = []
    for i in range(num_samples):
        x = np.array([np.random.random() * 7, np.random.random() * 7])
        y = target_function(x)
        dataset.append((x, y))
    return dataset


def plot_function(ax, function, text, hold=False):
    ax.cla()
    x_0 = np.linspace(0, 7, 100)
    x_1 = np.linspace(0, 7, 100)
    z = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            z[j, i] = function(np.array([x_0[i], x_1[j]]))
    X_0, X_1 = np.meshgrid(x_0, x_1)
    ax.plot_surface(X_0, X_1, z, rstride=8, cstride=8, alpha=0.3)
    ax.contourf(X_0, X_1, z, zdir='z', offset=-3, cmap=cm.coolwarm)
    ax.contourf(X_0, X_1, z, zdir='x', offset=-1, cmap=cm.coolwarm)
    ax.contourf(X_0, X_1, z, zdir='y', offset=-1, cmap=cm.coolwarm)
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 8)
    ax.set_zlim(-3, 3)
    ax.view_init(45, 45)
    ax.set_title(text)
    if hold:
        plt.show()
    else:
        plt.draw()
        plt.savefig(text + '.png')
        plt.pause(.0001)

train, eval_ = get_tile_coder(
    min_val=0, max_val=7.0, num_tiles=10,
    num_tilings=10, num_features=2,
    learning_rate=0.1)

dataset = get_dataset(10000, 0, 7)
fig = plt.figure(figsize=np.array([12, 5]))
ax_0 = fig.add_subplot(1, 2, 1, projection='3d')
ax_1 = fig.add_subplot(1, 2, 2, projection='3d')
plot_function(ax_0, target_function, 'Target function')
# import ipdb; ipdb.set_trace()

print 'Training'
for i, datapoint in enumerate(dataset):
    x, y = datapoint
    train(x.astype('float32'), y.astype('float32'))
    if i <= 20:
        step = 5
    elif i <= 100:
        step = 20
    elif i <= 1500:
        step = 100
    else:
        step = 1000
    if i % step == 0:
        plot_function(ax_1, eval_, 'Seen points: ' + str(i))

    if i == len(dataset) - 1:
        plot_function(ax_1, eval_, 'Learned Function', True)
