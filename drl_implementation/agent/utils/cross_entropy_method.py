import torch as T
from torch.distributions import Normal


class CEM(object):
    def __init__(self, sample_dimension,
                 sample_num=100, selection_threshold=20, iteration_num=6,
                 initial_mean=0.0, initial_std=0.2):
        self.sample_dimension = sample_dimension
        self.sample_num = sample_num
        self.selection_threshold = selection_threshold
        self.initial_mean = initial_mean
        self.initial_std = initial_std
        self.iteration_num = iteration_num
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

    def sample(self, state, goal, evaluation_network, dimension=None):
        if dimension is None:
            dimension = self.sample_dimension
        actions = []
        num_samples_needed = state.size()[0]
        for sample_ind in range(num_samples_needed):
            state = state[sample_ind].repeat(self.sample_num, 1).to(self.device)
            goal = goal[sample_ind].repeat(self.sample_num, 1).to(self.device)
            mean = T.ones(dimension) * self.initial_mean
            std = T.ones(dimension) * self.initial_std
            mu = Normal(mean, std)
            samples = mu.rsample((self.sample_num,)).to(self.device)
            # evaluate and get the sorted indices of the first group of samples
            values = evaluation_network(state, goal, samples).view(1, self.sample_num).to(self.device)
            _, indices = T.sort(values, descending=True)
            for i in range(self.iteration_num):
                # gather the best samples with a selection threshold
                new_samples = samples[indices[0][:self.selection_threshold]]
                # calculate new mean & std, create new Gaussian and sample again
                mean = new_samples.mean(dim=0).view(dimension)
                std = T.sqrt(T.pow((new_samples - mean), 2).mean(dim=0)).view(dimension)
                mu = Normal(mean, std)
                samples = mu.rsample((self.sample_num,)).to(self.device)
                # evaluate and get the sorted indices of the new samples
                values = evaluation_network(state, goal, samples).to(self.device)
                _, indices = T.sort(values, descending=True)
            # store the top best action sample for one state-goal pair
            actions.append(samples[indices[0][0]])
        if len(actions) == 1:
            return actions[0].view(1, dimension)
        else:
            return T.cat(actions, dim=0).view(num_samples_needed, dimension)