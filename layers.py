import numpy as np
import theano
import theano.tensor as T


class TileCodingLayer:
    def __init__(self, min_val, max_val, num_tiles,
                 num_tilings, num_features):
        self.min_val = min_val
        self.max_val = max_val
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings
        self.num_features = num_features

        # The range of each feature is divided into num_tile tiles.
        # But we keep one more tile. The reason is that since we shift
        # tilings by an offset, we want to make sure that every point
        # after shifting maps to one tile.
        # Shape: (num_tilings, (num_tiles+1) * num_features)
        self.weight = theano.shared(np.zeros(
            (num_tilings,) + (num_tiles + 1,) * num_features))

    def quantize(self, x):
        len_tile = float(self.max_val - self.min_val) / self.num_tiles
        offsets = (len_tile * T.arange(self.num_tilings).astype('float32') /
                   self.num_tilings)
        # shapes: (features, num_tilings) = (features, 1) + (1, num_tilings)
        mapped_x = x.dimshuffle(0, 'x') + offsets.dimshuffle('x', 0)
        # Since we have an extra tile:
        new_max_val = self.max_val + len_tile
        # Mapping into range [0, num_tiles + 1] to be able to do quantization
        # by casting to int.
        mapped_x = ((self.num_tiles + 1) * mapped_x /
                    (new_max_val - self.min_val))
        # shape: (num_tilings, num_features)
        q_x = mapped_x.astype('int32').T
        return q_x

    def approximate(self, q_x):
        indices = [T.arange(self.num_tilings)]
        for f in range(self.num_features):
            indices += [q_x[:, f]]
        self.indices = indices
        # shape: (num_tilings,)
        selected = self.weight.__getitem__(tuple(indices))
        y_hat = T.sum(selected)
        return y_hat

    def update_rule(self, y, y_hat, learning_rate):
        # grad is only used to locate the weights which
        # are used for computing y_hat. Since
        # "y_hat = T.sum(selected)", T.grad will returns
        # 1 for corresponding weights and 0 for others.
        grad_ = T.grad(y_hat, self.weight)
        learning_rate = learning_rate / self.num_tilings
        upd = grad_ * learning_rate * (y - y_hat)
        updates = [(self.weight, self.weight + upd)]
        return updates


def get_tile_coder(min_val, max_val, num_tiles, num_tilings,
                   num_features, learning_rate):
    # x.shape: (num_features), y.shape: ()
    x = T.fvector('x')
    y = T.fscalar('y')

    tile_coding_layer = TileCodingLayer(
        min_val=min_val, max_val=max_val,
        num_tiles=num_tiles, num_tilings=num_tilings,
        num_features=num_features)

    # quantized_x
    q_x = tile_coding_layer.quantize(x)
    y_hat = tile_coding_layer.approximate(q_x)
    updates = tile_coding_layer.update_rule(y, y_hat, 0.1)

    train = theano.function([x, y], y_hat, updates=updates, allow_input_downcast=True)
    eval_ = theano.function([x], y_hat, allow_input_downcast=True)

    return train, eval_
