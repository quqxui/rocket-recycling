from cv2 import randShuffle
import torch                                   
import torch.nn as nn                         
import torch.nn.functional as F                 
import numpy as np                              
import gym    


import numpy as np
import torch
from rocket import Rocket
from a2c import ActorCritic
from pg import PG
import matplotlib.pyplot as plt
import utils
import os
import glob                                 
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 超参数
BATCH_SIZE = 32                                
LR = 0.01                                       
EPSILON = 0.9                               
GAMMA = 0.9                                  
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
MEMORY_CAPACITY = 2000                          # 记忆库容量
task = 'landing'  # 'hover' or 'landing'
render = False
max_m_episode = 800000
max_steps = 800
env = Rocket(task=task, max_steps=max_steps)      
N_ACTIONS = env.action_dims    #9
N_STATES = env.state_dims      #8


ckpt_folder = os.path.join('./', task + '_prioritizeddqn' +'_ckpt')
if not os.path.exists(ckpt_folder):
    os.mkdir(ckpt_folder)



class PositionalMapping(nn.Module):
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
        x = x.view([x.size(0), -1])
        x = self.mapping(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x


from collections import deque
import random



# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self):                                                         # 定义DQN的一系列属性
        self.eval_net, self.target_net = MLP(N_STATES,N_ACTIONS), MLP(N_STATES,N_ACTIONS)                           # 利用Net创建两个神经网络: 评估网络和目标网络
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))             # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()                                           # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.priorities = np.ones((MEMORY_CAPACITY,), dtype=np.float32)


    def choose_action(self, x):                                                 
        x = torch.unsqueeze(torch.FloatTensor(x), 0)                           
        if np.random.uniform() < EPSILON:                                    
            actions_value = self.eval_net.forward(x)                            
            action = torch.max(actions_value, 1)[1].data.numpy()               
            action = action[0]                                                  
        else:                                                               
            action = np.random.randint(0, N_ACTIONS)                           
        return action                                                          

    def store_transition(self, s, a, r, s_,sample_index,probe,flag):                            
        transition = np.hstack((s, [a, r], s_))                         
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY         # 这个巧妙，通过求余 只要覆盖最前面              
        self.memory[index, :] = transition                         
        self.memory_counter += 1
        if flag:
            self.priorities[sample_index] = probe.squeeze(1)

    def learn(self):                                                      
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                 
            self.target_net.load_state_dict(self.eval_net.state_dict())        
        self.learn_step_counter += 1                   

        p = self.priorities/np.sum(self.priorities)
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE,p=p)      

        b_memory = self.memory[sample_index, :]                              
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])


        
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()                                    
        loss.backward()                                                
        self.optimizer.step()                                          


        return sample_index,torch.abs(q_eval - q_target).detach().numpy()


last_episode_id = 0
REWARDS = []

net = DQN() 

# if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
#     checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1])
#     net.load_state_dict(checkpoint['model_G_state_dict'])
#     last_episode_id = checkpoint['episode_id']
#     REWARDS = checkpoint['REWARDS']

for episode_id in range(last_episode_id, max_m_episode):

    # training loop
    s = env.reset()
    rewards = []
    for step_id in range(max_steps):
        
        
        # env.render()     
        a = net.choose_action(s)                                        # 输入该步对应的状态s，选择动作
        s_, r, done, info = env.step(a)                                 # 执行动作，获得反馈
        rewards.append(r)

        #                # 存储样本

        s = s_                                                # 更新状态

        if net.memory_counter > MEMORY_CAPACITY:              # 如果累计的transition数量超过了记忆库的固定容量2000
            # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
            sample_index,probe = net.learn()
            net.store_transition(s, a, r, s_,sample_index,probe,1)
        else:
            net.store_transition(s, a, r, s_,None,None,0)
            

        if done:
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


