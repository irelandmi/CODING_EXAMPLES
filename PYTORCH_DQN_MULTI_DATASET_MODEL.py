#%%
#I've re-engineering a pre-built DQN model from a Conv2d network to a linear one to accomodate a stock trading environment rather than the gym 'cartpole' environment.

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
import random

#GLOBAL VARIABLES#
environment_size = 10_000
num_episodes = 500
BATCH_SIZE = 100_00_000
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5_000_000
TARGET_UPDATE = 10
memsize = 10_000

connection = psycopg2.connect(user="postgres",
                                  password="password",
                                  host="localhost",
                                  port="5432",
                                  database="datawarehouse")

# Create a cursor to perform database operations
cursor = connection.cursor()

symbols_query = '''
SELECT distinct symbol
	FROM public."K_LINES";
'''

symbols = pd.read_sql_query(symbols_query, connection)

df_dict = {}
for i, a in symbols.iterrows():
    postgres_query = """ SELECT *
	FROM public."K_LINES"
	where symbol = '{}'
	order by open_stamp ;""".format(a["symbol"])
    cursor.execute(postgres_query)
    val = pd.read_sql_query(postgres_query, connection)
    for ii in range(1,5):
        val["open-"+str(ii)] = val["open"].shift(ii)
        val["high-"+str(ii)] = val["high"].shift(ii)
        val["low-"+str(ii)] = val["low"].shift(ii)
    val = val.drop(["open_stamp","close_stamp","symbol"],axis=1)
    val = val.dropna()
    df_dict[a["symbol"]] = val


randenv1 = random.randrange(1000,len(df_dict["BTCUSDT"])-1100)
randenv2 = random.randrange(1000,len(df_dict["ETHUSDT"])-1100)
randenv3 = random.randrange(1000,len(df_dict["BNBUSDT"])-1100)
randenv4 = random.randrange(1000,len(df_dict["LTCUSDT"])-1100)
dataset_1 = df_dict["BTCUSDT"].iloc[randenv1:randenv1+1000]
dataset_2 = df_dict["ETHUSDT"].iloc[randenv2:randenv2+1000]
dataset_3 = df_dict["BNBUSDT"].iloc[randenv3:randenv3+1000]
dataset_4 = df_dict["LTCUSDT"].iloc[randenv4:randenv4+1000]


#run model
def my_process_data(df, window_size, frame_bound):
    start = frame_bound[0] - window_size
    end = frame_bound[1]
    prices = df.loc[:, 'close'].to_numpy()[start:end]
    signal_features = df.to_numpy()[start:end]
    return prices, signal_features


class MyStocksEnv(StocksEnv):
    
    def __init__(self, prices, signal_features, **kwargs):
        self._prices = prices
        self._signal_features = signal_features
        super().__init__(**kwargs)

    def _process_data(self):
        return self._prices, self._signal_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


prices, signal_features = my_process_data(df=dataset_1, window_size=1, frame_bound=(1, len(dataset_1)))
env = MyStocksEnv(prices, signal_features, df=dataset_1, window_size=1, frame_bound=(1, len(dataset_1)))

prices1, signal_features1 = my_process_data(df=dataset_1, window_size=1, frame_bound=(1, len(dataset_1)))
env1 = MyStocksEnv(prices1, signal_features1, df=dataset_1, window_size=1, frame_bound=(1, len(dataset_1)))

prices2, signal_features2 = my_process_data(df=dataset_2, window_size=1, frame_bound=(1, len(dataset_2)))
env2 = MyStocksEnv(prices2, signal_features2, df=dataset_2, window_size=1, frame_bound=(1, len(dataset_2)))

prices3, signal_features3 = my_process_data(df=dataset_3, window_size=1, frame_bound=(1, len(dataset_3)))
env3 = MyStocksEnv(prices3, signal_features3, df=dataset_3, window_size=1, frame_bound=(1, len(dataset_3)))

prices4, signal_features4 = my_process_data(df=dataset_4, window_size=1, frame_bound=(1, len(dataset_4)))
env4 = MyStocksEnv(prices4, signal_features4, df=dataset_4, window_size=1, frame_bound=(1, len(dataset_4)))

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self,num_inputs, outputs):
        super(DQN, self).__init__()
        #self.fc1 = nn.Linear()
        self.fc1 = nn.Linear(num_inputs, 120)
        self.fc2 = nn.Linear(120, 240)
        self.fc3 = nn.Linear(240, 120)
        self.fc4 = nn.Linear(120, 120)
        self.fc5 = nn.Linear(120, 2)

        

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #print(x)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

state = env.reset()

# Get number of actions from gym action space
n_actions = env.action_space.n
policy_net = DQN(18, n_actions).to(device)
target_net = DQN(18, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(memsize)
steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    #print(batch)
    #print(type(batch))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    #print("batch reward")
    #print(batch.reward)
    #print("batch action")
    #print(batch.action)
    #print(batch.action[0])
    #print("batch state")
    #print(batch.state)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    #reward_batch = batch.reward

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

episode_list = []
randenv1 = random.randrange(environment_size,len(df_dict["BTCUSDT"])-environment_size-100)
dataset_1 = df_dict["BTCUSDT"].iloc[randenv1:randenv1+environment_size]
prices1, signal_features1 = my_process_data(df=dataset_1, window_size=1, frame_bound=(1, len(dataset_1)))
env1 = MyStocksEnv(prices1, signal_features1, df=dataset_1, window_size=1, frame_bound=(1, len(dataset_1)))
env = env1

#%%
for i_episode in tqdm(range(num_episodes)):
    # Initialize the environment and state
    selection = 1
    if selection == 1:
        pass
    elif selection == 2:
        randenv2 = random.randrange(environment_size,len(df_dict["ETHUSDT"])-environment_size-100)
        dataset_2 = df_dict["ETHUSDT"].iloc[randenv2:randenv2+environment_size]
        prices2, signal_features2 = my_process_data(df=dataset_2, window_size=1, frame_bound=(1, len(dataset_2)))
        env2 = MyStocksEnv(prices2, signal_features2, df=dataset_2, window_size=1, frame_bound=(1, len(dataset_2)))
        env = env2
    elif selection == 3:
        randenv3 = random.randrange(environment_size,len(df_dict["BNBUSDT"])-100)
        dataset_3 = df_dict["BNBUSDT"].iloc[randenv3:randenv3+environment_size]
        prices3, signal_features3 = my_process_data(df=dataset_3, window_size=1, frame_bound=(1, len(dataset_3)))
        env3 = MyStocksEnv(prices3, signal_features3, df=dataset_3, window_size=1, frame_bound=(1, len(dataset_3)))
        env = env3
    else:
        env = env4

    templist = []
    state = env.reset()
    
    for t in count():
        
        #if t % 1000 == 0:
            #print(t)
        # Select and perform an action
        state = torch.tensor(state, dtype=torch.float32, device=device)
        action = select_action(state)

        next_state, reward, done, info = env.step(action.item())
        #print("Reward val:")
        #print(reward)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        reward = np.float32(reward)
        reward = torch.tensor([reward], dtype=torch.float32, device=device)
        #print(type(reward))
        #print(reward.shape)
        
        # Store the transition in memory
        #templist.append([state,action,next_state,reward])
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            #episode_durations.append(t + 1)
            #plot_durations()
            #tempdf = pd.DataFrame(templist)
            #tempdf.to_csv("Episode {}.csv".format(t))
            break
            
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    #print("info:", info)
    episode_list.append(info)
    if i_episode % 50 == 0:
        listy = []
        for i in episode_list:
            listy.append(i["total_profit"])
        profit_df = pd.DataFrame(listy)
        profit_df.plot(title="profit")

        #plt.cla()
        #env.render_all()
        #plt.show()
        pass
        
#env.render()
#env.close()
#plt.ioff()
#plt.show()


#GRAPH EPISODES REWARD AND PROFIT $$$
listy = []
print(len(episode_list))
for i in episode_list:
    listy.append(i["total_reward"])
reward_df = pd.DataFrame(listy)
reward_df.plot(title="reward")

listy = []
print(len(episode_list))
for i in episode_list:
    listy.append(i["total_profit"])
profit_df = pd.DataFrame(listy)
profit_df.plot(title="profit")

print(reward_df.sum())
#%%
#print("> max_possible_profit:", env1.max_possible_profit())
print("> max_possible_profit:", env.max_possible_profit())

# %%

print()
print("custom_env information:")
print("> shape:", env.shape)
print("> df.shape:", env.df.shape)
print("> prices.shape:", env.prices.shape)
print("> signal_features.shape:", env.signal_features.shape)
print("> max_possible_profit:", env.max_possible_profit())
# %%
