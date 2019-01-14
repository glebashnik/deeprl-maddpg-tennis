import copy
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from unityagents import UnityEnvironment

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OUNoise:
    def __init__(self, size, mu, theta, sigma):
        self.state = None
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc1_units, fc2_units):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        x = self.bn1(F.relu(self.fc1(state)))
        x = self.bn2(F.relu(self.fc2(x)))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_size, action_size, fc1_units, fc2_units):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        x = self.bn1(F.relu(self.fc1(state)))
        x = F.relu(self.fc2(torch.cat((x, action), dim=1)))
        return self.fc3(x)


class Replay:
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self):
        return random.sample(self.buffer, k=self.batch_size)

    def __len__(self):
        return len(self.buffer)


def transpose_to_tensor(tuples):
    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float).to(DEVICE)

    return list(map(to_tensor, zip(*tuples)))


def hard_update(target_model, source_model):
    for target_param, param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target_model, source_model, mix):
    for target_param, online_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - mix) + online_param.data * mix)


class SelfPlayAgent:
    def __init__(self, config):
        self.config = config

        self.online_actor = config.actor_fn().to(DEVICE)
        self.target_actor = config.actor_fn().to(DEVICE)
        self.online_actor_opt = config.actor_opt_fn(self.online_actor.parameters())

        self.online_critic = config.critic_fn().to(DEVICE)
        self.target_critic = config.critic_fn().to(DEVICE)
        self.online_critic_opt = config.critic_opt_fn(self.online_critic.parameters())

        self.noises = [config.noise_fn() for _ in range(config.num_agents)]
        self.replay = config.replay_fn()

        hard_update(self.target_actor, self.online_actor)
        hard_update(self.target_critic, self.online_critic)

    def act(self, states):
        state = torch.from_numpy(states).float().to(DEVICE)

        self.online_actor.eval()

        with torch.no_grad():
            action = self.online_actor(state).cpu().numpy()

        self.online_actor.train()

        action += [n.sample() for n in self.noises]
        return np.clip(action, -1, 1)

    def step(self, states, actions, rewards, next_states, dones):
        full_state = states.flatten()
        next_full_state = next_states.flatten()
        self.replay.add((states, full_state, actions, rewards, next_states, next_full_state, dones))

        if len(self.replay) > self.replay.batch_size:
            self.learn()

    def learn(self):
        # Sample a batch of transitions from the replay buffer
        transitions = self.replay.sample()
        states, full_state, actions, rewards, next_states, next_full_state, dones = transpose_to_tensor(transitions)

        # Update online critic model
        # Compute actions for next states with the target actor model
        with torch.no_grad():
            target_next_actions = [self.target_actor(next_states[:, i, :]) for i in range(self.config.num_agents)]

        target_next_actions = torch.cat(target_next_actions, dim=1)

        # Compute Q values for the next states and next actions with the target critic model
        with torch.no_grad():
            target_next_qs = self.target_critic(next_full_state.to(DEVICE), target_next_actions.to(DEVICE))

        # Compute Q values for the current states and actions with the Bellman equation
        target_qs = rewards.sum(1, keepdim=True)
        target_qs += self.config.discount * target_next_qs * (1 - dones.max(1, keepdim=True)[0])

        # Compute Q values for the current states and actions with the online critic model
        actions = actions.view(actions.shape[0], -1)
        online_qs = self.online_critic(full_state.to(DEVICE), actions.to(DEVICE))

        # Compute and minimize the online critic loss
        online_critic_loss = F.mse_loss(online_qs, target_qs.detach())
        self.online_critic_opt.zero_grad()
        online_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_critic.parameters(), 1)
        self.online_critic_opt.step()

        # Update online actor model
        # Compute actions for the current states with the online actor model
        online_actions = [self.online_actor(states[:, i, :]) for i in range(self.config.num_agents)]
        online_actions = torch.cat(online_actions, dim=1)
        # Compute the online actor loss with the online critic model
        online_actor_loss = -self.online_critic(full_state.to(DEVICE), online_actions.to(DEVICE)).mean()
        # Minimize the online critic loss
        self.online_actor_opt.zero_grad()
        online_actor_loss.backward()
        self.online_actor_opt.step()

        # Update target critic and actor models
        soft_update(self.target_actor, self.online_actor, self.config.target_mix)
        soft_update(self.target_critic, self.online_critic, self.config.target_mix)


def run(agent):
    config = agent.config
    scores_deque = deque(maxlen=100)
    scores = []
    mean_scores = []

    for episode in range(config.max_episodes):
        score = np.zeros(config.num_agents)
        env_info = config.env.reset(train_mode=True)[config.brain_name]
        states = env_info.vector_observations

        for step in range(config.max_steps):
            actions = agent.act(states)
            env_info = config.env.step(actions)[config.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, dones)

            score += rewards
            states = next_states

            if np.any(dones):
                break

        score = score.max()
        scores.append(score)
        scores_deque.append(score)
        mean_score = np.mean(scores_deque)
        mean_scores.append(mean_score)

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(episode, mean_score, score))

        if mean_score >= config.goal_score:
            break

    torch.save(agent.online_actor.state_dict(), config.actor_path)
    torch.save(agent.online_critic.state_dict(), config.critic_path)

    fig, ax = plt.subplots()
    x = np.arange(1, len(scores) + 1)
    ax.plot(x, scores)
    ax.plot(x, mean_scores)
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode #')
    fig.savefig(config.scores_path)
    plt.show()


class Config:
    def __init__(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

        self.env = None
        self.brain_name = None
        self.num_agents = None
        self.state_size = None
        self.action_size = None

        self.actor_fn = None
        self.actor_opt_fn = None
        self.critic_fn = None
        self.critic_opt_fn = None
        self.replay_fn = None
        self.noise_fn = None

        self.discount = None
        self.target_mix = None

        self.max_episodes = None
        self.max_steps = None

        self.actor_path = None
        self.critic_path = None
        self.scores_path = None


def main():
    config = Config(seed=6)

    config.env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")
    config.brain_name = config.env.brain_names[0]
    env_info = config.env.reset(train_mode=True)[config.brain_name]

    config.num_agents = len(env_info.agents)
    config.state_size = env_info.vector_observations.shape[1]
    config.action_size = config.env.brains[config.brain_name].vector_action_space_size

    config.actor_fn = lambda: Actor(config.state_size, config.action_size, 128, 128)
    config.actor_opt_fn = lambda params: optim.Adam(params, lr=1e-3)

    config.critic_fn = lambda: Critic(config.state_size * config.num_agents, config.action_size * config.num_agents, 128, 128)
    config.critic_opt_fn = lambda params: optim.Adam(params, lr=2e-3)

    config.replay_fn = lambda: Replay(config.action_size, buffer_size=int(1e6), batch_size=128)
    config.noise_fn = lambda: OUNoise(config.action_size, mu=0., theta=0.15, sigma=0.1)

    config.discount = 0.99
    config.target_mix = 3e-3

    config.max_episodes = 1000
    config.max_steps = int(1e6)
    config.goal_score = 1

    config.actor_path = 'actor.pth'
    config.critic_path = 'critic.pth'
    config.scores_path = 'scores.png'

    agent = SelfPlayAgent(config)
    run(agent)


if __name__ == '__main__':
    main()