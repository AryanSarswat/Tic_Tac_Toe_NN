from random import random,randint
import math
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from time import sleep
import torch.optim as optim
import matplotlib.pyplot as plt



"""
Tic Tac Toe Player
"""

"""
HyperParameters
"""
epsilon_val = [0.1,0.2,0.3,0.4,0.5]
gamma_val = [1,0.95,0.9,0.85,0.8]

X = "X"
O = "O"
EMPTY = None
epsilon = 0.2
gamma = 0.90
EPOCHS = 500
BATCH_SIZE = 5
m_x = -epsilon/EPOCHS

def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def player(board):
    """
    Returns player who has the next turn on a board.
    """
    total_pieces = 0
    for i in board:
        total_pieces += i.count(X) + i.count(O)
    if total_pieces%2:
        return O
    else:
        return X

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    pos_action = set()
    for i in range(len(board)):
        for j in range(len((board[0]))):
            if board[i][j] == EMPTY:
                pos_action.add((i,j))
    return pos_action

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if board[action[0]][action[1]] != EMPTY:
        raise Exception('Invalid Move')
    new_board = deepcopy(board)
    new_board[action[0]][action[1]] = player(board)
    return new_board

def transpose(board):
    out = initial_state()
    for i in range(len(board)):
        for j in range(len(board[0])):
            out[j][i] = board[i][j]
    return out

def all_possible_future_states(board):
    """
    Returns a set of all possible states which can follow
    """
    pos_action = actions(board)
    pos_future_states = [result(board,act) for act in pos_action]
    return pos_future_states

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    #Checking for Rows and Columns
    for row in board:
        if row.count(X) == 3:
            return X
        elif row.count(O) == 3:
            return O
    for col in transpose(board):
        if col.count(X) == 3:
            return X
        elif col.count(O) == 3:
            return O
    #Checking the Diagonals
    if board[0][0] != EMPTY:
        if board[0][0] == board[1][1] and board[1][1] == board[2][2]:
            return board[0][0]
    if board[0][2] != EMPTY:
        if board[0][2] == board[1][1] and board[1][1] == board[2][0]:
            return board[0][2]
    else:
        return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) != None:
        return True
    #Check for Draw
    for i in board:
        if EMPTY in i:
            return False
    return True

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def encode_and_flatten(board):
    encoded_board = deepcopy(board)
    for i in range(len(board)):
        for j in range(len(board[i])):
            element = board[i][j]
            if element == EMPTY:
                encoded_board[i][j] = [1,0,0]
            elif element == X:
                encoded_board[i][j] = [0,1,0]
            elif element == O:
                encoded_board[i][j] = [0,0,1]
    encoded_board = np.array(encoded_board)
    encoded_board = torch.Tensor(encoded_board.flatten())
    return encoded_board

def generate_episode(network,iteration):
    current_state = initial_state()
    episode = [current_state]
    Q_values = [network.forward(encode_and_flatten(current_state))]
    while not terminal(current_state):
        pos_states = all_possible_future_states(current_state)
        State_Q_values = list(map(network.forward,list(map(encode_and_flatten,pos_states))))
        
        #Epsilion Greedy Algo
        if random() < ((m_x*iteration) + epsilon):
            index = randint(0,len(pos_states)-1)
            next_state = pos_states[index]
        else:
            index = np.argmax(State_Q_values)
            next_state = pos_states[index]
        
        episode.append(next_state)
        Q_values.append(State_Q_values[index])
        current_state = next_state
    return episode,Q_values
    
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(27, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x           

def main():
    NN = DQN()
    print(NN)
    optimizer = optim.Adam(NN.parameters(), lr=0.01)
    loss_array = []
    for i in tqdm(range(EPOCHS)):
        data = [generate_episode(NN,i) for n in range(BATCH_SIZE)]
        NN.zero_grad()
        for episode,q_values in data:
            win = winner(episode[-1])
            actual_q_values = []
            
            #Calculate the actual q_values
            if win == X:
                for j in range(len(q_values)):
                    actual_q_values.append(1*(gamma**j))
            elif win == O:
                for j in range(len(q_values)):
                    actual_q_values.append(-1*(gamma**j))             
            else:
                actual_q_values = [0 for j in range(len(q_values))]
            
            q_values = torch.tensor(q_values,requires_grad=True)
            actual_q_values = torch.Tensor(actual_q_values[::-1])
            loss = torch.nn.MSELoss(reduction="mean")
            output = loss(q_values,actual_q_values)
            output.backward()
            optimizer.step()
        loss_array.append(output.item())       

    plt.plot(loss_array)
    plt.title(f"$\\epsilon=$ {epsilon} , $\\gamma=$ {gamma} ")
    plt.show()



if __name__ == '__main__':
    for i in epsilon_val:
        for j in gamma_val:
            gamma = j
            epsilon = i
            main()