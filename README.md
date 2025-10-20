# PPO-Humanoid

This repository contains the implementation of a Proximal Policy Optimization (PPO) agent to control a humanoid in the
OpenAI Gymnasium Mujoco environment. The agent is trained to master complex humanoid locomotion using deep reinforcement
learning.

---

## Results

![Demo Gif](/docs/demo.gif)

The clip above showcases the performance of the PPO agent in the Humanoid-v5 environment after about 1000 epochs of
training.

---

## Getting Started

To get started with this project, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/ProfessorNova/PPO-Humanoid.git
    cd PPO-Humanoid
    ```

2. **Set Up Python Environment**:
   Make sure you have Python installed (tested with Python 3.10.11). It's recommended to create a virtual environment to
   avoid dependency conflicts. You can use `venv` or `conda` for this purpose.

3. **Install Dependencies**:
   Run the following command to install the required packages:
    ```bash
    pip install -r req.txt
    ```

   For proper PyTorch installation, visit [pytorch.org](https://pytorch.org/get-started/locally/) and follow the
   instructions based on your system configuration.

4. **Install Gymnasium Mujoco**:
   You need to install the Mujoco environment to simulate the humanoid:
    ```bash
    pip install gymnasium[mujoco]
    ```

5. **Train the Model**:
   To start training the model, run:
    ```bash
    python train_ppo.py
    ```
   This creates the folders `checkpoints`, `logs`, and `videos` in the root of the repository. The `checkpoints` folder
   will contain the model checkpoints, the `logs` folder will contain the TensorBoard logs, and the `videos` folder will
   contain the recorded videos of the agent's performance.

6. **Monitor Training Progress**:
   You can monitor the training progress by viewing the videos in the `videos` folder or by looking at the graphs in
   TensorBoard:
    ```bash
    tensorboard --logdir "logs"
    ```

---

## Usage

### Running pre-trained model

To run the pre-trained PPO model, execute the following command (make sure you followed the installation steps above):

```bash
python test_ppo.py
```

This will load the pre-trained model for the root of the repository (`model.pt`) and run it in the Humanoid-v5
environment. If git lfs is not installed or not working properly, you can download the model manually from the [release page](https://github.com/ProfessorNova/PPO-Humanoid/releases). Place the downloaded `model.pt` file in the root of the repository.

### Training with custom hyperparameters

You can customize the training by modifying the command-line arguments:

```bash
python train_ppo.py --n-envs <number_of_envs> --n-epochs <number_of_epochs> ...
```

All hyperparameters can be viewed either with `python train_ppo.py --help` or by looking at the
`parse_args_ppo` function in `lib/utils.py`.

---

## Structure

The training process mainly involves the following components:

- **lib/agent_ppo.py**: Contains the PPO agent implementation, including the policy and value networks and the necessary
  methods for sampling actions, getting log probabilities and entropy, as well as the values from the value network.
- **lib/buffer_ppo.py**: Implements the replay buffer to store experiences and sample batches for training. It also
  handles
  the GAE (Generalized Advantage Estimation) for calculating advantages.
- **lib/utils.py**: Contains utility functions for parsing command-line arguments, setting up the environment, and
  creating recordings of the agent's performance.
- **train_ppo.py**: The main script for training the PPO agent. It initializes the environment, agent, and buffer,
  and handles the training loop.

---

## Statistics

### Performance Metrics:

The following charts provide insights into the performance during training with the current default hyperparameters:

- **Reward**:
  ![Reward](/docs/reward_mean.svg)

The average reward per step basically indicates how fast the humanoid is moving.

The graph starts with a quick increase in the reward, which is expected as the agent learns to not instantly fall over.
After that, the reward stays relatively stable, with some fluctuations. After about 500 epochs, it starts to increase
significantly, indicating that the agent has learned to walk and tries to go faster and faster.

If the reward drops temporarily, it does not necessarily mean that the agent is performing worse. It can also be due to
the agent learning to stabilize and thus moving not as fast per step.

---

- **Policy Loss**:
  ![Policy Loss](/docs/loss_policy.svg)

- **Value Loss**:
  ![Value Loss](/docs/loss_value.svg)

- **Entropy**:
  ![Entropy](/docs/loss_entropy.svg)

---

- **Learning Rate**:
- ![Learning Rate](/docs/metrics_learning_rate.svg)

- **KL Divergence**:
- ![KL Divergence](/docs/metrics_kl.svg)

---

## References

The knowledge to implement this project was mainly acquired from the following book:

- [Deep Reinforcement Learning Hands-On](https://www.amazon.com/Deep-Reinforcement-Learning-Hands-Q-networks/dp/1788834240)
  by Maxim Lapan
