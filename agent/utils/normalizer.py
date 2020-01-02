import numpy as np


class Normalizer(object):
    def __init__(self, input_dims, init_mean, init_var, scale_factor=1, epsilon=1e-2, clip_range=5):
        self.input_dims = input_dims
        self.sample_count = 0
        self.history = []
        self.history_mean = init_mean
        self.history_var = init_var
        self.epsilon = epsilon*np.ones(self.input_dims)
        self.input_clip_range = (-clip_range*np.ones(self.input_dims), clip_range*np.ones(self.input_dims))
        self.scale_factor = scale_factor

    def store_history(self, *args):
        self.history.append(*args)

    # update mean and var for z-score normalization
    def update_mean(self):
        if len(self.history) == 0:
            return
        new_sample_num = len(self.history)
        new_history = np.array(self.history, dtype=np.float)
        new_mean = np.mean(new_history, axis=0)

        new_var = np.sum(np.square(new_history - new_mean), axis=0)
        new_var = (self.sample_count * self.history_var + new_var)
        new_var /= (new_sample_num + self.sample_count)
        self.history_var = np.sqrt(new_var)

        new_mean = (self.sample_count * self.history_mean + new_sample_num * new_mean)
        new_mean /= (new_sample_num + self.sample_count)
        self.history_mean = new_mean

        self.sample_count += new_sample_num
        self.history.clear()

    # pre-process inputs, currently using max-min-normalization
    def __call__(self, inputs):
        inputs = (inputs - self.history_mean) / (self.history_var+self.epsilon)
        inputs = np.clip(inputs, self.input_clip_range[0], self.input_clip_range[1])
        return self.scale_factor*inputs