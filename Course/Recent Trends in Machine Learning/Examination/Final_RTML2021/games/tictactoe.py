import os
import numpy

from .abstract_game import AbstractGame
class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.board = numpy.zeros((3, 3), dtype="int32")
        self.player = 0     ## player O as 0 and player X as 1

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.player

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        self.board = numpy.zeros((3, 3), dtype="int32")
        self.player = 1     ## player O as 1 and player X as -1
        return self.get_observation()

    def get_observation(self):
        """
        Return current state of the game
        Returns:
            The current observation
        """
        board_player1 = numpy.where(self.board == 1, 1, 0)
        board_player2 = numpy.where(self.board == -1, 1, 0)
        board_to_play = numpy.full((3, 3), self.player)
        return numpy.array([board_player1, board_player2, board_to_play], dtype="int32")
        
    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        # action is a number 0-8 for declare row and column of position in the board
        row = action // 3
        col = action % 3

        # Check that the action is illegal action, unless the player should loss from illegal action.
        if not (action in self.legal_actions()):
            return self.get_observation(), -1, True

        # input the action of current player into the board
        self.board[row, col] = self.player

        # Check that the game is finished in 2 condition: have a winner, or no any moves left
        have_win = self.have_winner()
        done = have_win or len(self.legal_actions()) == 0

        # If have the winner, the current player should be a winner.
        reward = 1 if have_win else 0

        # change current player
        self.player *= -1

        return self.get_observation(), reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.
    
        Returns:
            An array of integers, subset of the action space.
        """
        legal = []
        for i in range(9):
            row = i // 3
            col = i % 3
            if self.board[row, col] == 0:
                legal.append(i)
        return legal

    def have_winner(self):
        """
        Check that have winner.
    
        Returns:
            True if there is a winner, otherwise False
        """
        # Horizontal and vertical checks
        for i in range(3):
            if (self.board[i, :] == self.player * numpy.ones(3, dtype="int32")).all():
                return True
            if (self.board[:, i] == self.player * numpy.ones(3, dtype="int32")).all():
                return True

        # Diagonal checks
        if (
            self.board[0, 0] == self.player
            and self.board[1, 1] == self.player
            and self.board[2, 2] == self.player
        ):
            return True
        if (
            self.board[2, 0] == self.player
            and self.board[1, 1] == self.player
            and self.board[0, 2] == self.player
        ):
            return True

        return False

    def render(self):
        """
        Display the game observation.
        """
        for i in range(5):
            row = int(i / 2)
            str = ""
            for j in range(11):
                col = int(j / 4)
                if i == row * 2 + 1 and j == col * 4 + 3:
                    str += "+"
                elif i == row * 2 and j == col * 4 + 1:
                    if self.board[row,col] == 1:
                        str += "O"
                    elif self.board[row,col] == -1:
                        str += "X"
                    else:
                        str += " "
                elif i == row * 2 + 1:
                    str += "-"
                elif j == col * 4 + 3:
                    str += "|"
                
                else:
                    str += " "
            print(str)
            

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        while True:
            try:
                row = int(
                    input(
                        f"Enter the row (1, 2 or 3) to play for the player {self.to_play()}: "
                    )
                )
                col = int(
                    input(
                        f"Enter the column (1, 2 or 3) to play for the player {self.to_play()}: "
                    )
                )
                choice = (row - 1) * 3 + (col - 1)
                if (
                    choice in self.legal_actions()
                    and 1 <= row
                    and 1 <= col
                    and row <= 3
                    and col <= 3
                ):
                    break
            except:
                pass
            print("Wrong input, try again")
        return choice

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        board = self.board
        action = numpy.random.choice(self.legal_actions())
        # Horizontal and vertical checks
        for i in range(3):
            if abs(sum(board[i, :])) == 2:
                ind = numpy.where(board[i, :] == 0)[0][0]
                action = numpy.ravel_multi_index((numpy.array([i]), numpy.array([ind])), (3, 3))[0]
                if self.player * sum(board[i, :]) > 0:
                    return action

            if abs(sum(board[:, i])) == 2:
                ind = numpy.where(board[:, i] == 0)[0][0]
                action = numpy.ravel_multi_index((numpy.array([ind]), numpy.array([i])), (3, 3))[0]
                if self.player * sum(board[:, i]) > 0:
                    return action

        # Diagonal checks
        diag = board.diagonal()
        anti_diag = numpy.fliplr(board).diagonal()
        if abs(sum(diag)) == 2:
            ind = numpy.where(diag == 0)[0][0]
            action = numpy.ravel_multi_index((numpy.array([ind]), numpy.array([ind])), (3, 3))[0]
            if self.player * sum(diag) > 0:
                return action

        if abs(sum(anti_diag)) == 2:
            ind = numpy.where(anti_diag == 0)[0][0]
            action = numpy.ravel_multi_index((numpy.array([ind]), numpy.array([2 - ind])), (3, 3))[0]
            if self.player * sum(anti_diag) > 0:
                return action

        return action

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.
        
        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        row = action_number // 3 + 1
        col = action_number % 3 + 1
        return f"Play row {row}, column {col}"