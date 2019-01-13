import os
import random
import copy
from collections import deque, namedtuple

import numpy as np
import progressbar as pb
import matplotlib.pyplot as plt

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


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Agent:
    def __init__(self, config):
        self.online_actor = config.actor_fn().to(DEVICE)
        self.target_actor = config.actor_fn().to(DEVICE)
        self.actor_opt = config.actor_opt_fn(self.online_actor.parameters())

        self.online_critic = config.critic_fn().to(DEVICE)
        self.target_critic = config.critic_fn().to(DEVICE)
        self.critic_opt = config.critic_opt_fn(self.online_critic.parameters())

        self.noise = config.noise_fn()

        # initialize targets same as original networks
        hard_update(self.target_actor, self.online_actor)
        hard_update(self.target_critic, self.online_critic)

    def act(self, state):
        state = torch.from_numpy(state).float().to(DEVICE)

        self.online_actor.eval()

        with torch.no_grad():
            action = self.online_actor(state).cpu().numpy().flatten()

        self.online_actor.train()

        action += self.noise.sample()
        return action


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class MultiAgent:
    def __init__(self, config):
        self.config = config
        self.agents = [Agent(config) for _ in range(config.num_agents)]
        self.iter = 0

    def get_online_actors(self):
        return [agent.online_actor for agent in self.agents]

    def get_target_actors(self):
        return [agent.target_actor for agent in self.agents]

    def act(self, states):
        return np.array([agent.act(state) for agent, state in zip(self.agents, states)])

    def learn(self, transitions, agent_idx):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        states, full_state, actions, rewards, next_states, next_full_states, dones = transpose_to_tensor(transitions)

        full_state = torch.stack(full_state)
        next_full_states = torch.stack(next_full_states)

        agent = self.agents[agent_idx]
        agent.critic_opt.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_states)
        target_actions = torch.cat(target_actions, dim=1)
        target_critic_input = torch.cat((next_full_states.t(), target_actions), dim=1).to(DEVICE)

        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)

        y = rewards[agent_idx].view(-1, 1) + self.config.discount * q_next * (1 - dones[agent_idx].view(-1, 1))

        actions = torch.cat(actions, dim=1)
        critic_input = torch.cat((full_state.t(), actions), dim=1).to(DEVICE)
        q = agent.online_critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_opt.step()

        # update actor network using policy gradient
        agent.actor_opt.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [self.agents[i].online_actor(ob) if i == agent_idx else self.agents[i].online_actor(ob).detach()
                   for i, ob in enumerate(states)]

        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((full_state.t(), q_input), dim=1)

        # get the policy gradient
        actor_loss = -agent.online_critic(q_input2).mean()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_opt.step()

    def update_targets(self):
        """soft update targets"""
        self.iter += 1

        for agent in self.agents:
            soft_update(agent.target_actor, agent.online_actor, self.config.target_mix)
            soft_update(agent.target_critic, agent.online_critic, self.config.target_mix)


def random_play(config):
    for i in range(1, 6):                                      # play game for 5 episodes
        env_info = config.env.reset(train_mode=False)[config.brain_name]     # reset the environment
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(config.num_agents)                          # initialize the score (for each agent)

        while True:
            actions = np.random.randn(config.num_agents, config.action_size) # select an action (for each agent)
            actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            env_info = config.env.step(actions)[config.brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step

            print(states)
            if np.any(dones):                                  # exit loop if episode finished
                break
        # (2 + 2 + 2) * 3 = 24
        print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))

    config.env.close()


def run(config):
    magent = MultiAgent(config)
    replay = config.replay_fn()
    scores = [[]] * config.num_agents

    # use keep_awake to keep workspace from disconnecting
    for episode in range(config.max_episodes):
        env_info = config.env.reset(train_mode=True)[config.brain_name]
        states = env_info.vector_observations
        full_state = states.flatten()
        score = np.zeros(config.num_agents)

        for step in range(config.max_steps):
            actions = magent.act(states)
            env_info = config.env.step(actions)[config.brain_name]
            next_states = env_info.vector_observations
            next_full_state = next_states.flatten()
            rewards = env_info.rewards
            dones = env_info.local_done

            transition = (states, full_state, actions, rewards, next_states, next_full_state, dones)
            replay.add(transition)

            score += rewards
            states, full_state = next_states, next_full_state

            if len(replay) > replay.batch_size:
                for i in range(config.num_agents):
                    transitions = replay.sample()
                    magent.learn(transitions, i)

                magent.update_targets()

            if np.any(dones):
                break

        for i in range(config.num_agents):
            scores[i].append(score[i])

        print('\rEpisode {}\tAverage scores: {}'.format(episode, np.array_str(score, precision=2)))

    config.env.close()


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

    config.critic_fn = lambda: Critic(config.state_size + config.num_agents * config.action_size, 1, 128, 128)
    config.critic_opt_fn = lambda params: optim.Adam(params, lr=1e-3)

    config.replay_fn = lambda: Replay(config.action_size, buffer_size=int(1e6), batch_size=256)
    config.noise_fn = lambda: OUNoise(config.action_size, mu=0., theta=0.15, sigma=0.05)

    config.discount = 0.99
    config.target_mix = 1e-3

    config.max_episodes = int(1000)
    config.max_steps = int(1e6)
    config.goal_score = 30

    config.actor_path = 'actor.pth'
    config.critic_path = 'critic.pth'
    config.scores_path = 'scores.png'

    #random_play(config)
    # agent = Agent(config)
    run(config)


if __name__ == '__main__':
    main()