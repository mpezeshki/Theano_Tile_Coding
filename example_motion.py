import numpy as np
import matplotlib.pyplot as plt
from layers import get_tile_coder
num_used_features = 10


def get_dataset():
    y = np.load('files/data.npy')
    y = y - np.min(y, axis=0, keepdims=True)
    y = y / np.max(y, axis=0, keepdims=True)
    features = y[:600, :num_used_features]
    targets = y[10:610, 9]

    return (np.array(features).astype('float32'),
            np.array(targets).astype('float32'))

train, eval_ = get_tile_coder(
    min_val=0, max_val=1.0, num_tiles=20,
    num_tilings=20, num_features=num_used_features,
    learning_rate=0.05)

features, targets = get_dataset()
fig = plt.figure(figsize=np.array([12, 2]))
ax_0 = fig.add_subplot(1, 1, 1)

print 'Training'
prediction = []
for i in range(600):
    train(features[i], targets[i].astype('float32'))
    pred = eval_(features[i])
    prediction += [pred]
    ax_0.cla()
    ax_0.set_ylim(0.0, 1.0)
    ax_0.set_xlim(0.0, 600.0)
    ax_0.scatter([i], features[i, 9], c='g', s=10, edgecolors='g')
    ax_0.plot(np.arange(len(features[:i + 1, 9])), features[:i + 1, 9], c='g')
    ax_0.scatter([i + 10], pred, c='r', s=10, edgecolors='r')
    x = np.arange(len(prediction))
    x = [l + 10 for l in x]
    ax_0.plot(x, prediction, c='r')
    plt.draw()
    plt.pause(.0001)
