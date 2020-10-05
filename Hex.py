# -*- coding: utf-8 -*-
# Importing required modules
import numpy as np
from disjoint_set import DisjointSet


class Hex(object):
    def __init__(self):
        self.dsu = DisjointSet()
        self.size = 11
        self.turn_monitor = 1
        self.board = np.full((self.size, self.size), -1)

    def toss(self):
        """simulate a toss and decide which player goes first

        Args:

        Returns:
        Returns 1 if player assigned mark 1 has won, or 0 if his opponent won

        """
        turn = np.random.randint(0, 2, size=1)
        if turn.mean() == 0:
            self.turn_monitor = 0
        elif turn.mean() == 1:
            self.turn_monitor = 1
        return self.turn_monitor

    def valid(self, coords):
        """check if coord is valid or not

        Args: coords, an array with two columns and one row

        Returns: None
        """

        return coords[0] in range(self.size) and coords[1] in range(self.size)

    def hash(self, coords):
        """get hash of a cell

        Args: coords, an array with two columns and one row

        Returns: int
        """
        return coords[0] * self.size + coords[1] + 5

    def connect(self, player, coords):
        """connect coords of a move in DSU

        Args: coords, an array with two columns and one row

        Returns: None
        """
        assert isinstance(coords, tuple)
        hsh = self.hash(coords)
        if coords == (0, 0):
            if player:
                self.dsu.union(hsh, 2)
            else:
                self.dsu.union(hsh, 1)
        if coords == (0, self.size - 1):
            if player:
                self.dsu.union(hsh, 2)
            else:
                self.dsu.union(hsh, 4)
        if coords == (self.size - 1, 0):
            if player:
                self.dsu.union(hsh, 3)
            else:
                self.dsu.union(hsh, 1)
        if coords == (self.size - 1, self.size - 1):
            if player:
                self.dsu.union(hsh, 3)
            else:
                self.dsu.union(hsh, 4)
        if coords[1] == 0 and player == 0:
            self.dsu.union(hsh, 1)
        if coords[1] == self.size - 1 and player == 0:
            self.dsu.union(hsh, 4)
        if coords[0] == 0 and player == 1:
            self.dsu.union(hsh, 2)
        if coords[0] == self.size - 1 and player == 1:
            self.dsu.union(hsh, 3)
        for x in list(zip([-1, 0, 1, 0], [0, 1, 0, -1])):
            assert isinstance(x, tuple)
            if self.valid(x) and self.board[x] == player:
                self.dsu.union(hsh, self.hash(x))

        for k in [1, -1]:
            diag = tuple(map(lambda x: x + k, coords))
            if self.valid(diag) and self.board[diag] == player:
                self.dsu.union(hsh, self.hash(diag))

    def move(self, player, coord):
        """perform the action of placing a mark on the tic tac toe board
        After performing the action, this function flips the value of the turn_monitor to
        the next player

        Args:
        player: 1 if player who is assigned the mark 1 is performing the action,
        0 if his opponent is performing the action
        coord: The coordinate where the 1 or 0 is to be placed on the
        tic-tac-toe board (numpy array)

        Returns:
        game_status(): Calls the game status function and returns its value
        board: Returns the new board state after making the move

        """
        assert isinstance(coord, tuple)
        if (
                self.board[coord] != -1 or
                self.game_status() != "In Progress" or
                self.turn_monitor != player
        ):
            raise ValueError("Invalid move")
        self.connect(player, coord)
        self.board[coord] = player
        self.turn_monitor = 1 - player
        return self.game_status(), self.board

    def game_status(self):
        """check the current status of the game,
        whether the game has been won, lost or is in progress

        Args:

        Returns:
        "Won" if the game from player 1 perspective has been won,
        or "In Progress", if the game is still in progress

        """
        if self.dsu.connected(1, 4):
            return "Lost"
        elif self.dsu.connected(2, 3):
            return "Won"
        else:
            return "In Progress"


"""Test the game"""

# TODO Prash
