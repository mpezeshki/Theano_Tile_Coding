import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from layers import get_tile_coder


def get_dataset():
    # x = np.linspace(0, 8 * np.pi, 830)
    # y = np.sin(x)
    y = np.load('data.npy')
    y = y - np.min(y)
    y = y / np.max(y)
    # import ipdb; ipdb.set_trace()
    feature_1 = y[:400]
    feature_2 = y[5:405]
    target = y[10:410]

    features = [feature_1, feature_2]
    targets = [target]
    return features, targets


def plot_function(ax, input, target, text, hold=False):
    ax.cla()
    ax.plot(np.arange(len(input)), input, c='b')
    ax.plot(np.arange(len(target)), target, c='g')
    # ax.set_xlim(0.0, 8 * np.pi)
    # ax.set_ylim(-1.0, 1.0)
    if hold:
        plt.show()
    else:
        plt.draw()
        plt.savefig(text + '.png')
        plt.pause(.0001)


def plot_function2(ax, input, target, text, hold=False):
    ax.cla()
    ax.plot(np.arange(len(input)), input, c='g')
    ax.plot(np.arange(len(target)), target, c='r', ls='-', lw=2)
    # ax.set_xlim(0.0, 8 * np.pi)
    # ax.set_ylim(-1.0, 1.0)
    if hold:
        plt.show()
    else:
        plt.draw()
        plt.savefig(text + '.png')
        plt.pause(.0001)

train, eval_ = get_tile_coder(
    min_val=0, max_val=1.0, num_tiles=10,
    num_tilings=10, num_features=2,
    learning_rate=0.07)

features, targets = get_dataset()
fig = plt.figure(figsize=np.array([12, 5]))
ax_0 = fig.add_subplot(2, 1, 1)
ax_1 = fig.add_subplot(2, 1, 2)
plot_function(ax_0, features[1], targets[0], 'Target function')
# import ipdb; ipdb.set_trace()

print 'Training'
rn = range(features[0].shape[0])
np.random.shuffle(rn)
len_ = len(rn)
for k in range(20):
    for i in rn:
        feat_1 = features[0][i]
        feat_2 = features[1][i]
        tar = targets[0][i]
        train([feat_1.astype('float32'),
               feat_2.astype('float32')],
              tar.astype('float32'))

        # if i <= 20:
        #     step = 5
        # elif i <= 100:
        #     step = 20
        # elif i <= 1500:
        #     step = 100
        # else:
        step = 1000

        if i % step == 0 or i == len_ - 1:
            res = []
            for j in range(len_):
                feat_1 = features[0][j]
                feat_2 = features[1][j]
                pred, grad_ = eval_(
                    [feat_1.astype('float32'),
                     feat_2.astype('float32')])
                res += [pred]
            if i == len_ - 1 and k == 19:
                plot_function2(ax_1, targets[0], np.array(res), 'Seen points: ' + str(i), True)
            else:
                plot_function2(ax_1, targets[0], np.array(res), 'Seen points: ' + str(i))
