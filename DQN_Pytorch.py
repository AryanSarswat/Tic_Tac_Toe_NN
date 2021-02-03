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
        self.fc2 = nn.Linear(32,64)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64,1)
        self.optimizer = optim.SGD(self.parameters(),lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        state = T.Tensor(state).to(self.device)
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent():
    def __init__(self,gamma,epsilon,epsilon_min,epsilon_dec_rate,lr,batch_size,epochs,state):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec_rate = epsilon_dec_rate
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.current_state = state
        self.Q_policy = DeepQNetwork(lr)
        self.Q_target = DeepQNetwork(lr)
        self.loss_array = []

    def choose_action(self,state):
        possible_states = all_possible_future_states(state)
        rand = np.random.random()
        future_Q_values = []
        if rand < self.epsilon:
            new_index = np.random.randint(0,len(possible_states))
            new_state = possible_states[new_index]
        else:
            encoded_states = []
            for st in possible_states:
                encoded_states.append(onehot_flatten(st))
            for st in encoded_states:
                future_Q_values.append(self.Q_target.forward(st).item())
            if player(state) == X:
                new_state = possible_states[np.argmax(future_Q_values)]
            elif player(state) == O:
                new_state = possible_states[np.argmin(future_Q_values)]
        return new_state

    def learn(self):
        for epoch in tqdm(range(self.epochs)):
            for batch in range(self.batch_size):
                self.current_state = initial_state()
                while not terminal(self.current_state):
                    self.Q_policy.optimizer.zero_grad()
                    possible_states = all_possible_future_states(self.current_state)
                    encoded = []
                    for state in possible_states:
                        encoded.append(onehot_flatten(state))
                    q_eval = self.Q_target.forward(onehot_flatten(self.current_state)).to(self.Q_target.device)
                    q_next = []
                    for one_hot_state in encoded:
                        q_next.append(self.Q_target.forward(one_hot_state).to(self.Q_target.device).item())
                    #Implement Epsilon Greedy
                    if np.random.random() <= self.epsilon:
                        index = np.random.randint(0,len(q_next))
                    else:
                        if player(self.current_state) == X:
                            index = np.argmax(q_next)
                        elif player(self.current_state) == O:
                            index = np.argmin(q_next)
                    #index = np.random.randint(0,len(q_next))
                    q_target= q_eval + (self.lr*(utility(possible_states[index]) + self.gamma*q_next[index] - q_eval))
                    loss = self.Q_policy.loss(q_eval,q_target).to(self.Q_policy.device)
                    loss.backward()
                    self.Q_policy.optimizer.step()
                    self.current_state = possible_states[index]
                    self.loss_array.append(loss.item())
            self.epsilon = self.epsilon*self.epsilon_dec_rate if self.epsilon > self.epsilon_min else self.epsilon_min
            self.Q_target.load_state_dict(self.Q_policy.state_dict())
            self.Q_target.eval()
        self.plot_loss()
    
    def train_episode(self,episode):
        l_array = []
        c_state = episode.pop()
        while episode!= []:
            self.Q_target.optimizer.zero_grad()
            q_eval = self.Q_target.forward(onehot_flatten(c_state)).to(self.Q_target.device)
            possible_states = all_possible_future_states(c_state)
            encoded = []
            q_next = []
            for state in possible_states:
                encoded.append(onehot_flatten(c_state))
            for one_hot_state in encoded:
                q_next.append(self.Q_target.forward(one_hot_state).to(self.Q_target.device).item())
            if player(c_state) == X:
                index = np.argmax(q_next)
            elif player(c_state) == O:
                index = np.argmin(q_next)
            q_target= q_eval + (self.lr*(utility(self.current_state) + self.gamma*q_next[index] - q_eval))
            loss = self.Q_target.loss(q_target,q_eval).to(self.Q_target.device)
            l_array.append(loss.item())
            loss.backward()
            self.Q_target.optimizer.step()
            c_state = episode.pop()
        plt.plot(l_array)
        plt.title("Loss versus iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()
        

    def plot_loss(self):
        plt.plot(self.loss_array)
        plt.title("Loss versus iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()
