import datetime
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from lib.agent_ppo import PPOAgent
from lib.buffer_ppo import PPOBuffer
from lib.utils import parse_args_ppo, make_env, log_video


def ppo_update(agent, optimizer, scaler, batch_obs, batch_actions, batch_returns, batch_old_log_probs, batch_adv,
               clip_epsilon, vf_coef, ent_coef):
    agent.train()

    optimizer.zero_grad()
    with torch.amp.autocast(str(device)):
        # Get the new log probabilities, entropies and values
        _, new_log_probs, entropies, new_values = agent.get_action_and_value(batch_obs, batch_actions)
        ratio = torch.exp(new_log_probs - batch_old_log_probs)

        # Approximate mean KL divergence for this batch per action dimension
        kl = ((batch_old_log_probs - new_log_probs) / batch_actions.size(-1)).mean()

        # Surrogate objectives
        surr1 = ratio * batch_adv
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value function loss
        value_loss = nn.MSELoss()(new_values.squeeze(1), batch_returns)

        # Entropy bonus
        entropy = entropies.mean()

        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item(), kl.item()


if __name__ == "__main__":
    args = parse_args_ppo()
    device = torch.device("cuda" if args.cuda else "cpu")

    # Create the folders for logging
    current_dir = os.path.dirname(__file__)
    folder_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    videos_dir = os.path.join(current_dir, "videos", folder_name)
    os.makedirs(videos_dir, exist_ok=True)
    checkpoint_dir = os.path.join(current_dir, "checkpoints", folder_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create the tensorboard writer
    log_dir = os.path.join(current_dir, "logs", folder_name)
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Create the environments
    envs = gym.vector.AsyncVectorEnv(
        [lambda: make_env(args.env, reward_scaling=args.reward_scale) for _ in range(args.n_envs)])
    test_env = make_env(args.env, reward_scaling=args.reward_scale, render=True)
    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.shape

    # Create the agent and optimizer
    agent = PPOAgent(obs_dim[0], act_dim[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)


    def lr_lambda(epoch):
        warmup_epochs = 10
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            T_cur = epoch - warmup_epochs
            T_total = args.n_epochs - warmup_epochs
            return 0.5 * (1 + np.cos(np.pi * T_cur / T_total))


    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler(str(device))  # Scale the gradients for mixed precision training
    print(agent.actor_mu)
    print(agent.actor_logstd)
    print(agent.critic)

    # Create the buffer
    buffer = PPOBuffer(obs_dim, act_dim, args.n_steps, args.n_envs, device, args.gamma, args.gae_lambda)

    # Start the training
    global_step_idx = 0
    best_mean_reward = -np.inf
    start_time = time.time()
    next_obs = torch.tensor(np.array(envs.reset()[0], dtype=np.float32), device=device)
    next_terminateds = torch.zeros(args.n_envs, dtype=torch.float32, device=device)
    next_truncateds = torch.zeros(args.n_envs, dtype=torch.float32, device=device)
    reward_list = []

    try:
        for epoch in range(1, args.n_epochs + 1):

            # Collect trajectories
            for _ in tqdm(range(0, args.n_steps), desc=f"Epoch {epoch}: Collecting trajectories"):
                global_step_idx += args.n_envs
                obs = next_obs
                terminateds = next_terminateds
                truncateds = next_truncateds

                # Sample the actions
                with torch.no_grad():
                    actions, logprobs, _, values = agent.get_action_and_value(obs)
                    values = values.reshape(-1)

                # Step the environment
                next_obs, rewards, next_terminateds, next_truncateds, _ = envs.step(actions.cpu().numpy())

                # parse everything to tensors
                next_obs = torch.tensor(np.array(next_obs, dtype=np.float32), device=device)
                reward_list.extend(rewards)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_terminateds = torch.as_tensor(next_terminateds, dtype=torch.float32, device=device)
                next_truncateds = torch.as_tensor(next_truncateds, dtype=torch.float32, device=device)

                # Store the step in the buffer
                buffer.store(obs, actions, rewards, values, terminateds, truncateds, logprobs)

            # After the trajectories are collected, calculate the advantages and returns
            with torch.no_grad():
                # Finish the last step of the buffer with the value of the last state
                # and the terminated and truncated flags
                next_values = agent.get_value(next_obs).reshape(1, -1)
                next_terminateds = next_terminateds.reshape(1, -1)
                next_truncateds = next_truncateds.reshape(1, -1)
                traj_adv, traj_ret = buffer.calculate_advantages(next_values, next_terminateds, next_truncateds)

            # Get the stored trajectories from the buffer
            traj_obs, traj_act, traj_logprob = buffer.get()

            # Flatten the trajectories
            traj_obs = traj_obs.view(-1, *obs_dim)
            traj_act = traj_act.view(-1, *act_dim)
            traj_logprob = traj_logprob.view(-1)
            traj_adv = traj_adv.view(-1)
            traj_ret = traj_ret.view(-1)

            # Normalize the advantages
            traj_adv = (traj_adv - traj_adv.mean()) / (traj_adv.std() + 1e-8)

            # Create an array of indices to sample from the trajectories
            dataset_size = args.n_steps * args.n_envs
            traj_indices = np.arange(dataset_size)

            losses_policy = []
            losses_value = []
            entropies = []
            losses_total = []

            kl_list = []
            kl_early_stop = False

            for _ in tqdm(range(args.train_iters), desc=f"Epoch {epoch}: Training"):
                # Shuffle the indices
                np.random.shuffle(traj_indices)
                # Iterate over the batches
                for start_idx in range(0, dataset_size, args.batch_size):
                    end_idx = start_idx + args.batch_size
                    batch_indices = traj_indices[start_idx:end_idx]

                    batch_obs = traj_obs[batch_indices]
                    batch_actions = traj_act[batch_indices]
                    batch_returns = traj_ret[batch_indices]
                    batch_old_log_probs = traj_logprob[batch_indices]
                    batch_adv = traj_adv[batch_indices]

                    loss, policy_loss, value_loss, entropy, kl = ppo_update(agent, optimizer, scaler, batch_obs,
                                                                            batch_actions, batch_returns,
                                                                            batch_old_log_probs, batch_adv,
                                                                            args.clip_ratio, args.vf_coef,
                                                                            args.ent_coef)

                    losses_policy.append(policy_loss)
                    losses_value.append(value_loss)
                    entropies.append(entropy)
                    losses_total.append(loss)
                    kl_list.append(kl)

                    # Early stopping if KL divergence is too high
                    if kl > args.target_kl:
                        kl_early_stop = True
                        break

                if kl_early_stop:
                    break

            # Log the losses
            total_loss = np.mean(losses_total)
            policy_loss = np.mean(losses_policy)
            value_loss = np.mean(losses_value)
            entropy = np.mean(entropies)
            kl = np.mean(kl_list)
            writer.add_scalar(
                "loss/total", total_loss, epoch)
            writer.add_scalar(
                "loss/policy", policy_loss, epoch)
            writer.add_scalar(
                "loss/value", value_loss, epoch)
            writer.add_scalar(
                "loss/entropy", entropy, epoch)
            writer.add_scalar(
                "metrics/kl", kl, epoch)

            # Log learning rate
            writer.add_scalar(
                "metrics/learning_rate", scheduler.get_last_lr()[0], epoch)

            # Log the rewards
            mean_reward = float(np.mean(reward_list) / args.reward_scale)
            writer.add_scalar("reward/mean", mean_reward, epoch)
            reward_list = []
            print(f"Epoch {epoch} done in {time.time() - start_time:.2f}s, mean reward: {mean_reward:.2f}, "
                  f"total loss: {total_loss:.4f}, policy loss: {policy_loss:.4f}, value loss: {value_loss:.4f}, "
                  f"entropy: {entropy:.4f}, kl: {kl:.4f}, "
                  f"learning rate: {scheduler.get_last_lr()[0]:.2e}")
            start_time = time.time()

            # Save the model if the mean reward is better
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "best.pt"))
                print(f"New best model saved with mean reward: {mean_reward:.2f}")

            # Save the last model
            torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "last.pt"))

            # Every n epochs, log the video
            if epoch % args.render_epoch == 0:
                log_video(test_env, agent, device, os.path.join(videos_dir, f"epoch_{epoch}.mp4"))

            # Update the learning rate
            scheduler.step()

    finally:
        # Close the environments and tensorboard writer
        envs.close()
        test_env.close()
        writer.close()
