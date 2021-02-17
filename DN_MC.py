import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Tic_Tac_Toe import *
import matplotlib.pyplot as plt
from tqdm import tqdm

class DeepQNetwork(nn.Module):
    def __init__(self,lr):
        super(DeepQNetwork,self).__init__()
        self.fc1 = nn.Linear(27,32)
        self.dropout_1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32,16)
        self.dropout_2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(16,1)
        self.optimizer = optim.SGD(self.parameters(),lr=lr)
        self.loss = nn.SmoothL1Loss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        state = T.Tensor(state).to(self.device)
        x = F.relu(self.fc1(state))
        x = self.dropout_1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_2(x)
        x = F.tanh(self.fc3(x))
        return x


class Agent():
    def __init__(self,gamma,epsilon,lr,batch_size,epochs,state):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.state_episode = [initial_state()]
        self.Q_network = DeepQNetwork(lr)
        self.loss_array = []

    def choose_action(self,state):
        possible_states = all_possible_future_states(state)
        future_Q_values = []
        for st in possible_states:
            future_Q_values.append(self.Q_network.forward(onehot_flatten(st)).to(self.Q_network.device).item())
        if player(state) == X:
            new_state = possible_states[np.argmax(future_Q_values)]
        else:
            new_state = possible_states[np.argmin(future_Q_values)]
        return new_state

    def learn(self):
        self.loss_array = []
        for epoch in tqdm(range(self.epochs)):
            for iteration_i in range(self.batch_size):
                episode = self.generate_episode()
                win = winner(episode[-1])
                state_values = []
                if win == X:
                    reward = 1
                    for i in range(len(episode)):
                        state_values.append(reward)
                        reward*=self.gamma
                elif win == O:
                    reward = -1
                    for i in range(len(episode)):
                        state_values.append(reward)
                        reward*=self.gamma
                else:
                    state_values = [0 for i in range(len(episode))]
                
                state_values = state_values[::-1]
                enc = []
                for st in episode:
                    enc.append(onehot_flatten(st))
                for ind in range(len(enc)):
                    state_val = self.Q_network.forward(enc[ind]).to(self.Q_network.device)
                    target_val = T.tensor(state_values[ind],dtype=T.float32).to(self.Q_network.device)
                    target_val = target_val.view(1)
                    loss = self.Q_network.loss(state_val,target_val).to(self.Q_network.device)
                    self.Q_network.optimizer.zero_grad()
                    loss.backward()
                    self.loss_array.append(loss.item())
                    self.Q_network.optimizer.step()
        self.plot_loss()

    def generate_episode(self):
        c_state = initial_state()
        episode = []
        while not terminal(c_state):
            pos_states = all_possible_future_states(c_state)
            '''
            if np.random.random() < self.epsilon:
                new_state = pos_states[np.random.randint(0,len(pos_states))]
                episode.append(new_state)
                c_state = new_state
            else:
                state_values = [self.Q_network.forward(onehot_flatten(state)).to(self.Q_network.device).item() for state in pos_states]
                if player == X:
                    new_state = pos_states[np.argmax(state_values)]
                else:
                    new_state = pos_states[np.argmin(state_values)]
                episode.append(new_state)
                c_state = new_state
            '''
            c_state = pos_states[np.random.randint(0,len(pos_states))]
            episode.append(c_state)
        return episode

    def train_episode(self,episode):
        win = winner(episode[-1])
        state_values = []
        if win == X:
            reward = 1
            for i in range(len(episode)):
                state_values.append(reward)
                reward*=self.gamma
        elif win == O:
            reward = -1
            for i in range(len(episode)):
                state_values.append(reward)
                reward*=self.gamma
        else:
            state_values = [0 for i in range(len(episode))]          
        state_values = state_values[::-1]
        enc = []
        for st in episode:
            enc.append(onehot_flatten(st))
        for ind in range(len(enc)):
            self.Q_network.optimizer.zero_grad()
            state_val = self.Q_network.forward(enc[ind]).to(self.Q_network.device)
            target_val = T.tensor(state_values[ind],dtype=T.float32).to(self.Q_network.device)
            target_val = target_val.view(1)
            loss = self.Q_network.loss(state_val,target_val).to(self.Q_network.device)
            loss.backward()
            self.loss_array.append(loss.item())
            self.Q_network.optimizer.step()
        #Change the way we update (implement incremental update instead of direct update)
        #Try learning rate = 1

    def plot_loss(self):
        plt.plot(self.loss_array)
        plt.title("Loss versus iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()
