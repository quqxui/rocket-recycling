import random
import numpy as np
import torch
import torch.optim as optim
from rocket import Rocket
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import glob
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


task = 'landing'  # 'hover' or 'landing'
render = False
max_m_episode = 800000
max_steps = 800
GAMMA = 0.999
exploration = 0.001
LR = 5e-5
temperature = 1000

env = Rocket(task=task, max_steps=max_steps)
ckpt_folder = os.path.join('./', task + '_temperature' + '_ckpt')
if not os.path.exists(ckpt_folder):
    os.mkdir(ckpt_folder)



def calculate_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


class PositionalMapping(nn.Module):
    """
    Positional mapping Layer.
    This layer map continuous input coordinates into a higher dimensional space
    and enable the prediction to more easily approximate a higher frequency function.
    See NERF paper for more details (https://arxiv.org/pdf/2003.08934.pdf)
    """

    def __init__(self, input_dim, L=5, scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L*2 + 1)
        self.scale = scale

    def forward(self, x):

        x = x * self.scale

        if self.L == 0:
            return x

        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x)
            x_cos = torch.cos(2**i * PI * x)
            h.append(x_sin)
            h.append(x_cos)

        return torch.cat(h, dim=-1) / self.scale


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.mapping = PositionalMapping(input_dim=input_dim, L=7)

        h_dim = 128
        self.linear1 = nn.Linear(in_features=self.mapping.output_dim, out_features=h_dim, bias=True)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear3 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear4 = nn.Linear(in_features=h_dim, out_features=output_dim, bias=True)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # shape x: 1 x m_token x m_state
        x = x.view([1, -1])
        x = self.mapping(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class ActorCritic(nn.Module):
    """
    RL policy and update rules
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.output_dim = output_dim
        self.actor = MLP(input_dim=input_dim, output_dim=output_dim)
        self.critic = MLP(input_dim=input_dim, output_dim=1)
        self.softmax = nn.Softmax(dim=-1)

        self.optimizer = optim.RMSprop(self.parameters(), lr=LR)

    def forward(self, x,epoch):
        # shape x: batch_size x m_token x m_state
        y = self.actor(x)
        temp = temperature/(1+np.log(epoch+1)) 
        probs = self.softmax(y/temp)
        value = self.critic(x)

        return probs, value

    def get_action(self, state, epoch,deterministic=False):

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs, value = self.forward(state,epoch)
        probs = probs[0, :]
        value = value[0]

        if deterministic:
            action_id = np.argmax(np.squeeze(probs.detach().cpu().numpy()))
        else:
            if random.random() < exploration: 
                action_id = random.randint(0, self.output_dim - 1)
            else:
                
                action_id = np.random.choice(self.output_dim, p=np.squeeze(probs.detach().cpu().numpy()))

        log_prob = torch.log(probs[action_id] + 1e-9)

        return action_id, log_prob, value

    @staticmethod
    def update_ac(network, rewards, log_probs, values, masks, Qval, gamma=0.99):

        # compute Q values
        Qvals = calculate_returns(Qval.detach(), rewards, masks, gamma=gamma)
        Qvals = torch.tensor(Qvals, dtype=torch.float32).to(device).detach()

        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        # print('log_probs.shape',log_probs.shape)
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage.detach()).mean()
        # print(actor_loss.shape)
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss

        network.optimizer.zero_grad()
        ac_loss.backward()
        network.optimizer.step()



last_episode_id = 0
REWARDS = []

net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)


# if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
#     # load the last ckpt
#     checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1])
#     net.load_state_dict(checkpoint['model_G_state_dict'])
#     last_episode_id = checkpoint['episode_id']
#     REWARDS = checkpoint['REWARDS']

for episode_id in range(last_episode_id, max_m_episode):

    # training loop
    state = env.reset()
    rewards, log_probs, values, masks = [], [], [], []
    for step_id in range(max_steps):

        action, log_prob, value = net.get_action(state,episode_id)

        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)
        masks.append(1-done)
        if episode_id % 100 == 1 and render==True:
            env.render()

        if done or step_id == max_steps-1:
            _, _, Qval = net.get_action(state,episode_id)

            net.update_ac(net, rewards, log_probs, values, masks, Qval, gamma=GAMMA)
            break

    REWARDS.append(np.sum(rewards))
    if episode_id % 100 == 1:
        print('episode id: %d, episode reward: %.3f'
                % (episode_id, np.sum(rewards)))

    if episode_id % 1000 == 1:
        a=np.array(REWARDS)
        np.save(os.path.join(ckpt_folder, 'REWARDS.npy'),a)
        plt.figure()
        plt.plot(REWARDS), plt.plot(utils.moving_avg(REWARDS, N=50))
        plt.legend(['episode reward', 'moving avg'], loc=2)
        plt.grid()
        plt.xlabel('m episode')
        plt.ylabel('reward')
        plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(episode_id).zfill(8) + '.jpg'))
        plt.close()

    # if episode_id % 1000 == 1:
    #     torch.save({'episode_id': episode_id,
    #                 'REWARDS': REWARDS,
    #                 'model_G_state_dict': net.state_dict()},
    #                 os.path.join(ckpt_folder, 'ckpt_' + str(episode_id).zfill(8) + '.pt'))

