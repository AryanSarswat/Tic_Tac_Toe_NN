from random import random,randint
import math
from copy import deepcopy
import numpy as np


"""
Tic Tac Toe Player
"""
X = "X"
O = "O"
EMPTY = None

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

def onehot_flatten(state):
    """
    Returns a one hot encoded form of the state which is flattend to make a (1,27) matrix
    """
    output = []
    for row in state:
        for element in row:
            if element == X:
                output+= [1,0,0]
            elif element == O:
                output+= [0,1,0]
            else:
                output+= [0,0,1]
    return output