import os

import numpy as np
from Hex import Hex
from scipy.ndimage.interpolation import shift

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras import models

import pandas as pd

from tqdm import tqdm

"""Create legal moves generator"""


def legal_moves_generator(current_board_state, turn_monitor):
    """Function that returns the set of all possible legal moves and resulting board states,
    for a given input board state and player

    Args:
    current_board_state: The current board state
    turn_monitor: 1 if it's the player who places the mark 1's turn to play, 0 if its his opponent's turn

    Returns:
    legal_moves_dict: A dictionary of a list of possible next coordinate-resulting board state pairs
    The resulting board state is flattened to 1 d array

    """
    legal_moves_dict = {}
    for i in range(current_board_state.shape[0]):
        for j in range(current_board_state.shape[1]):
            if current_board_state[i, j] == -1:
                board_state_copy = current_board_state.copy()
                board_state_copy[i, j] = turn_monitor
                legal_moves_dict[(i, j)] = board_state_copy.flatten()
    return legal_moves_dict


"""Test the moves Generator"""

# TODO Prash


"""Create Move Selector"""


def move_selector(model, current_board_state, turn_monitor, game=None):
    """Function that selects the next move to make from a set of possible legal moves

    Args:
    model: The Evaluator function to use to evaluate each possible next board state
    turn_monitor: 1 if it's the player who places the mark 1's turn to play, 0 if its his opponent's turn

    Returns:
    selected_move: The numpy array coordinates where the player should place their mark
    new_board_state: The flattened new board state resulting from performing above selected move
    score: The score that was assigned to the above selected_move by the Evaluator (model)

    """
    tracker = {}
    legal_moves_dict = legal_moves_generator(current_board_state, turn_monitor)
    for legal_move_coord in legal_moves_dict:
        score = model.predict(legal_moves_dict[legal_move_coord].reshape(1, 121))
        tracker[legal_move_coord] = score
    if len(tracker) == 0:
        for i in current_board_state:
            print(*i)
        print(turn_monitor)
        assert False
    selected_move = max(tracker, key=tracker.get)
    new_board_state = legal_moves_dict[selected_move]
    score = tracker[selected_move]
    return selected_move, new_board_state, score


"""Test the Move Selector"""
# TODO Prash

"""Creating the opponent"""


def opponent_move_selector(current_board_state, turn_monitor):
    """Function that picks a legal move for the opponent

    Args:
    current_board_state: Current board state
    turn_monitor: whose turn it is to move
    mode: whether hard or easy mode

    Returns:
    selected_move: The coordinates of numpy array where placing the 0 will lead to two 0s being there (and no 1s)

    """
    tracker = {}
    legal_moves_dict = legal_moves_generator(current_board_state, turn_monitor)
    for legal_move_coord in legal_moves_dict:
        score = model.predict(legal_moves_dict[legal_move_coord].reshape(1, 121))
        tracker[legal_move_coord] = score
    selected_move = max(tracker, key=tracker.get)
    return selected_move


# TODO Prash

"""Train the model"""


def train(model, print_progress=False):
    """Function trains the Evaluator (model) by playing a game against an opponent
    playing random moves, and updates the weights of the model after the game

    Note that the model weights are updated using SGD with a batch size of 1

    Args:
    model: The Evaluator function being trained

    Returns:
    model: The model updated using SGD
    y: The corrected scores

    """
    # start the game
    if print_progress:
        print("___________________________________________________________________")
        print("Starting a new game")
    game = Hex()
    game.toss()
    scores_list = []
    corrected_scores_list = []
    new_board_states_list = []

    game_status = "In Progress"
    while 1:
        if game.game_status() == "In Progress" and game.turn_monitor == 1:
            # If its the program's turn, use the Move Selector function to select the next move
            selected_move, new_board_state, score = move_selector(model, game.board, game.turn_monitor, game=game)
            scores_list.append(score[0][0])
            new_board_states_list.append(new_board_state)
            # Make the next move
            game_status, board = game.move(game.turn_monitor, selected_move)
            if print_progress:
                print("Program's Move")
                print(board)
                print("\n")
        elif game.game_status() == "In Progress" and game.turn_monitor == 0:
            # selected_move = opponent_move_selector(game.board, game.turn_monitor)
            selected_move, new_board_state, score = move_selector(model, game.board, game.turn_monitor, game=game)
            # Make the next move
            game_status, board = game.move(game.turn_monitor, selected_move)
            if print_progress:
                print("Opponent's Move")
                print(board)
                print("\n")
        else:
            break

    # Correct the scores, assigning 1/0/-1 to the winning/drawn/losing final board state,
    # and assigning the other previous board states the score of their next board state
    new_board_states_list = tuple(new_board_states_list)
    new_board_states_list = np.vstack(new_board_states_list)
    result = ""
    if game_status == "Won" and (1 - game.turn_monitor) == 1:
        corrected_scores_list = shift(scores_list, -1, cval=1.0)
        result = "Won"
    if game_status == "Lost" and (1 - game.turn_monitor) == 0:
        corrected_scores_list = shift(scores_list, -1, cval=-1.0)
        result = "Lost"
    if print_progress:
        print("Program has ", result)
        print("\n Correcting the Scores and Updating the model weights:")
        print("___________________________________________________________________\n")

    x = new_board_states_list
    y = corrected_scores_list

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    # shuffle x and y in unison
    x, y = unison_shuffled_copies(x, y)
    x = x.reshape(-1, 121)

    # update the weights of the model, one record at a time
    model.fit(x, y, epochs=1, batch_size=1, verbose=0)
    return model, y, result


"""A single game."""

# updated_model,y,result=train(model,print_progress=False)


"""Create Model params and initialize"""

# TODO Prash

if os.path.exists("my_model.h5"):
    model = models.load_model('my_model.h5')
else:
    model = Sequential()
    model.add(Dense(242, input_dim=121, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(242, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, kernel_initializer='normal'))

    learning_rate = 0.001
    momentum = 0.8

    sgd = SGD(lr=learning_rate, momentum=momentum, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.summary()

"""Train with multiple games"""

data_for_graph = pd.DataFrame()
print("Training...")

gamec = 0
for game_counter in tqdm(range(1, 200)):
    model, y, result = train(model, print_progress=False)
    data_for_graph = data_for_graph.append({"game_counter": game_counter, "result": result}, ignore_index=True)
    if game_counter % 100 == 0:
        print("Saving new model...")
        model.save('new_model.h5')
    gamec += 1

"""plot and save model"""

bins = np.arange(1, gamec / 10000) * 10000
data_for_graph['game_counter_bins'] = np.digitize(data_for_graph["game_counter"], bins, right=True)
counts = data_for_graph.groupby(['game_counter_bins', 'result']).game_counter.count().unstack()
ax = counts.plot(kind='bar', stacked=True, figsize=(17, 5))
ax.set_xlabel("Count of Games in Bins of 10,000s")
ax.set_ylabel("Counts of Draws/Losses/Wins")
ax.set_title('Distribution of Results Vs Count of Games Played')

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
