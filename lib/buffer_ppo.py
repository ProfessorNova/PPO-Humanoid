import torch


class PPOBuffer:
    """
    Buffer for storing trajectories
    """

    def __init__(self, obs_dim, act_dim, size, num_envs, device, gamma=0.99, gae_lambda=0.95):
        # Initialize buffer
        self.capacity = size
        self.obs_buf = torch.zeros((size, num_envs, *obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((size, num_envs, *act_dim), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.val_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.term_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.trunc_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.logprob_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.ptr = 0

    def store(self, obs, act, rew, val, term, trunc, logprob):
        """
        Store one step of the trajectory. The step is in the first dimension of the tensors.
        In the second dimension are the different environments.
        """
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.term_buf[self.ptr] = term
        self.trunc_buf[self.ptr] = trunc
        self.logprob_buf[self.ptr] = logprob
        self.ptr += 1

    def calculate_advantages(self, last_vals, last_terminateds, last_truncateds):
        """
        Calculate advantages using Generalized Advantage Estimation (GAE).
        Should be called before get() to get the advantages and returns.
        :param last_vals: a tensor of shape (num_envs,) containing the value of the last state.
        :param last_terminateds: a tensor of shape (num_envs,) containing whether the last state was terminated.
        :param last_truncateds: a tensor of shape (num_envs,) containing whether the last state was truncated.
        :return: adv_buf, ret_buf
        """
        # Check if the Buffer is full
        assert self.ptr == self.capacity, "Buffer not full"
        # Calculate the advantages
        with torch.no_grad():
            adv_buf = torch.zeros_like(self.rew_buf)
            last_gae = 0.0
            for t in reversed(range(self.capacity)):
                # If the next state is the last state, use the last values and terminated flags
                next_vals = last_vals if t == self.capacity - 1 else self.val_buf[t + 1]
                term_mask = 1.0 - last_terminateds if t == self.capacity - 1 else 1.0 - self.term_buf[t + 1]
                trunc_mask = 1.0 - last_truncateds if t == self.capacity - 1 else 1.0 - self.trunc_buf[t + 1]

                # Only if the next state is marked as terminated the next value is 0
                # If the next state is marked as truncated, we still use the next value from the buffer
                # Last gae starts again from delta if the next state is terminated or truncated
                delta = self.rew_buf[t] + self.gamma * next_vals * term_mask - self.val_buf[t]
                last_gae = delta + self.gamma * self.gae_lambda * term_mask * trunc_mask * last_gae
                adv_buf[t] = last_gae
            ret_buf = adv_buf + self.val_buf
            return adv_buf, ret_buf

    def get(self):
        """
        Call this at the end of the sampling to get the stored trajectories needed for training.
        :return: obs_buf, act_buf, logprob_buf
        """
        assert self.ptr == self.capacity
        self.ptr = 0
        return self.obs_buf, self.act_buf, self.logprob_buf
