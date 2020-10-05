import os
from Hex import Hex
from keras import models

model = models.load_model("my_model.h5")
game = Hex()


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


while game.game_status() == "In Progress":
    if game.turn_monitor:
        selected_move, _, _ = move_selector(model, game.board, game.turn_monitor)
        game.move(game.turn_monitor, selected_move)
        game.display()
    else:
        print("Your turn.")
        s = input()
        assert s[0].isupper()
        assert s[0] <= "K"

        try:
            b = int(s[1:])
            assert b in range(1, 11)
            game.move(game.turn_monitor, (ord(s[0]) - ord("A"), b - 1))
        except Exception as e:
            print("Invalid Input")
            break
        os.system("clear")

print("You" + game.game_status())
