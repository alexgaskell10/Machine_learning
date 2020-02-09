from typing import List, Tuple
import numpy as np
import random
import operator
from time import time

OFFSET_UPPER_CASE_CHAR_CONVERSION = 64

class Game():
    '''
    Class to construct the board and carry out the game
    '''

    def __init__(self, m=3, n=3, k=3):
        '''
        '''
        self.m = m   # Cols
        self.n = n   # Rows
        self.k = k   # Consecutive pieces for win

        self.player_positions = set()
        self.computer_positions = set()
        self.step = 0
        self.flag_ab = False            # Flag to track if we are using alpha beta pruning

#----------------------------------------------------------------------------------------------------------
# play the game

    def play(self, ab_flag=False):
        '''
        Plays 1 game
        max --> player and therefore X
        min --> computer and therefore O
        '''
        self.ab_flag = ab_flag
        # randomly choose who goes first
        if random.uniform(0,1) < 0.5:
            # PLayer goes first
            # loop until someone has won or there is a draw
            while True:
                self.drawboard()
                last_move = None
                last_move = self.get_player_move_with_recommendation(last_move)
                self.drawboard()
                board = self.construct_full_board(self.player_positions, self.computer_positions)
                if self.has_won(board, last_move, 'X'):
                    print('Player has won')
                    break
                elif self.draw(board):
                    print('Draw')
                    break

                last_move = self.get_computer_move_with_minimax(last_move)
                self.drawboard()
                board = self.construct_full_board(self.player_positions, self.computer_positions)
                if self.has_won(board, last_move, 'O'):
                    print('Computer has won')
                    break
                elif self.draw(board):
                    print('Draw')
                    break
        else:
            # Computer goes first
            while True:
                self.drawboard()
                last_move = None
                last_move = self.get_computer_move_with_minimax(last_move)
                self.drawboard()
                board = self.construct_full_board(self.player_positions, self.computer_positions)
                if self.has_won(board, last_move, 'O'):
                    print('Computer has won')
                    break
                elif self.draw(board):
                    print('Draw')
                    break

                last_move = self.get_player_move_with_recommendation(last_move)
                self.drawboard()
                board = self.construct_full_board(self.player_positions, self.computer_positions)
                if self.has_won(board, last_move, 'X'):
                    print('Player has won')
                    break
                elif self.draw(board):
                    print('Draw')
                    break


    #----------------------------------------------------------------------------------------------------------
    # Get the players move
    def get_player_move(self):
        '''
        Requests a move from the player
        Player inputs a capital letter and a number
        '''
        position_taken = True
        while position_taken:
            coord_str = input('your move (e.g A2) = ')
            coord_x, coord_y = self.get_tuple_coordinates_from_str(coord_str)

            if (coord_x, coord_y) not in self.player_positions and (coord_x, coord_y) not in self.computer_positions:
                position_taken = False
                self.player_positions.add((coord_x, coord_y))
        return (coord_x, coord_y)

    def get_player_move_with_recommendation(self, last_move):
        '''
        Requests a move from the player but suggest the a move from minimax
        Player inputs a capital letter and a number
        '''
        position_taken = True
        board = self.construct_full_board(self.player_positions, self.computer_positions)
        if self.ab_flag:
            alpha, beta = -np.inf, np.inf
            eval, best_move = self.minimax_ab(board, last_move,'O', 'X',self.player_positions, self.computer_positions, alpha, beta)
        else:
            eval, best_move = self.minimax(board, last_move,'O', 'X', self.player_positions, self.computer_positions)

        recommendation = best_move
        # request input from the player, and keep requesting until a valid input is provided
        while position_taken:
            coord_str = input(f"your move (recommended {self.get_str_from_tuple_coordinates(best_move)}) = ")
            coord_x, coord_y = self.get_tuple_coordinates_from_str(coord_str)

            if (coord_x, coord_y) not in self.player_positions and (coord_x, coord_y) not in self.computer_positions:
                position_taken = False
                self.player_positions.add((coord_x, coord_y))
        return (coord_x, coord_y)


    def get_tuple_coordinates_from_str(self, coord_str: str) -> Tuple[int, int]:
        '''
        convert from user input to coordinates
        '''
        coord_str = coord_str.strip()

        if not 2 <= len(coord_str) <= 3:
            raise ValueError(f"The position provided '{coord_str}' is not valid")

        coord_1, coord_2 = coord_str[0], coord_str[1:]

        coord_1 = ord(coord_1) - OFFSET_UPPER_CASE_CHAR_CONVERSION
        coord_2 = int(coord_2)

        if not (0 < coord_1 <= self.m and 0 < coord_2 <= self.n):
            raise ValueError(f"The position provided '{coord_str}' is not valid")

        return coord_1, coord_2

    def get_str_from_tuple_coordinates(self, coordinates: tuple) -> str():
        '''
        coverts the coordinates of the board array back into the user friendly input cooridinates
        '''
        coord_1, coord_2 = coordinates[0], coordinates[1]
        return_string = chr(coord_1 + OFFSET_UPPER_CASE_CHAR_CONVERSION) + str(coord_2)
        return return_string


#----------------------------------------------------------------------------------------------------------
# get the computers move

    def get_computer_move_random(self):
        '''
        Generates a random action from the computer.
        '''

        postion_taken = True
        # loops through until an action is chosen that is not already taken
        while postion_taken:
            coord_x = random.randint(1, self.m)
            coord_y = random.randint(1, self.n)
            if (coord_x, coord_y) not in self.player_positions and (coord_x, coord_y) not in self.computer_positions:
                postion_taken = False
                self.computer_positions.add((coord_x, coord_y))
        return (coord_x, coord_y)

    def get_computer_move_with_minimax(self, last_move):
        '''
        Generates the best move from the minimax funciton with or without pruning.
        '''
        board = self.construct_full_board(self.player_positions, self.computer_positions)

        if self.ab_flag:
            alpha, beta = -np.inf, np.inf
            eval, best_move = self.minimax_ab(board, last_move,'X', 'O', self.player_positions, self.computer_positions, alpha, beta)
        else:
            eval, best_move = self.minimax(board, last_move,'X', 'O', self.player_positions, self.computer_positions)

        self.computer_positions.add(best_move)
        return best_move



    def minimax_ab(self, board, last_move, last_symbol, current_symbol,  X_positions, O_positions, alpha, beta):
        '''
        recursive function which performs minimax with alpha beta pruning
        '''
        if self.step%100000 == 0:
            print('The number of steps:',self.step)
        self.step += 1

        # if someonehas won or there is a draw, a leaf has been reached therefore return the evaulation function:
        # if 'X' won --> 1
        # if 'O' won --> -1
        # if draw --> 0

        if self.has_won(board, last_move, last_symbol):
            if last_symbol == 'X':
                return 1, (0,0)
            elif last_symbol == 'O':
                return -1, (0,0)
        elif self.draw(board):
            return 0, (0,0)


        # now if its not a leaf, recurisively apply the minimax function to search further down the tree.
        # to keep log of which move is best I will create a dictionary of all the moves. The Key will be the value
        # of the move and the value will be the move itself (a tuple)

        move_scores = {}

        if current_symbol == 'X':
            # now we need to iterate through all the possible actions
            for move in self.get_possible_moves(board):
                # need to create a duplicate of the board to apply these test moves too.
                duplicate_X_positions = X_positions.copy()
                duplicate_O_positions = O_positions.copy()
                duplicate_X_positions.add(move)
                duplicate_board = self.construct_full_board(duplicate_X_positions, duplicate_O_positions)

                # now recursively search down the tree.
                evaluation, discard = self.minimax_ab(duplicate_board, move, 'X', 'O', duplicate_X_positions, duplicate_O_positions, alpha, beta)
                if evaluation >= beta:
                    return evaluation, None
                alpha = max(alpha, evaluation)
                move_scores[move] = alpha
            return alpha , self.choose_best_move(move_scores, 'X')

        if current_symbol == 'O':
            # now we need to iterate through all the possible actions
            for move in self.get_possible_moves(board):
                duplicate_X_positions = X_positions.copy()
                duplicate_O_positions = O_positions.copy()
                duplicate_O_positions.add(move)
                duplicate_board = self.construct_full_board(duplicate_X_positions, duplicate_O_positions)

                # now recursively search down the tree.
                evaluation, discard = self.minimax_ab(duplicate_board, move,'O', 'X', duplicate_X_positions, duplicate_O_positions, alpha, beta)
                if evaluation <= alpha:
                    return evaluation, None
                beta = min(beta, evaluation)
                move_scores[move] = beta
            return beta, self.choose_best_move(move_scores, 'O')
        # now we return the best move
        return self.choose_best_move(move_scores, current_symbol)


    def minimax(self, board, last_move, last_symbol, current_symbol,  X_positions, O_positions):
        '''
        recursive funciton which performs minimax
        '''

        # if someonehas won or there is a draw, a leaf has been reached therefore return the evaulation function:
        # if 'X' won --> 1
        # if 'O' won --> -1
        # if draw --> 0



        if self.has_won(board, last_move, last_symbol):
            if last_symbol == 'X':
                return 1, (0,0)
            elif last_symbol == 'O':
                return -1, (0,0)
        elif self.draw(board):
            return 0, (0,0)


        # now if its not a leaf, recurisively apply the minimax function to search further down the tree.
        # to keep log of which move is best I will create a dictionary of all the moves. The Key will be the value
        # of the move and the value will be the move itself (a tuple)

        move_scores = {}


        if current_symbol == 'X':
            max_evaluation = -np.inf
            # now we need to iterate through all the possible actions
            for move in self.get_possible_moves(board):
                # need to create a duplicate of the board to apply these test moves too.
                duplicate_X_positions = X_positions.copy()
                duplicate_O_positions = O_positions.copy()
                duplicate_X_positions.add(move)
                duplicate_board = self.construct_full_board(duplicate_X_positions, duplicate_O_positions)

                # nowrecursively search down the tree.
                evaluation, discard = self.minimax(duplicate_board, move, 'X', 'O',duplicate_X_positions, duplicate_O_positions)
                max_evaluation = max(max_evaluation, evaluation)
                move_scores[move] = max_evaluation

            return max_evaluation , self.choose_best_move(move_scores, 'X')

        if current_symbol == 'O':
            min_evaluation = np.inf
            # now we need to iterate through all the possible actions
            for move in self.get_possible_moves(board):
                duplicate_X_positions = X_positions.copy()
                duplicate_O_positions = O_positions.copy()
                duplicate_O_positions.add(move)
                duplicate_board = self.construct_full_board(duplicate_X_positions, duplicate_O_positions)
                evaluation, discard = self.minimax(duplicate_board, move,'O', 'X', duplicate_X_positions, duplicate_O_positions)
                min_evaluation = min(min_evaluation, evaluation)
                move_scores[move] = min_evaluation

            return min_evaluation, self.choose_best_move(move_scores, 'O')
        # now we return the best move
        return self.choose_best_move(move_scores, current_symbol)

    def get_possible_moves(self, board):
        '''
        input: board
        output: list of possible moves, coordinates in tuples (x,y)
        '''
        possible_moves = []
        for i in range(self.m):
            for j in range(self.n):
                if board[j][i] == ' ':
                    # changing the indexing back to starting at 1 and not 0
                    # also board[Y][X]
                    X = i + 1
                    Y = j + 1
                    possible_moves.append((X,Y))
        return possible_moves

    def choose_best_move(self, move_scores, symbol):
        '''
        Input:
        move_score --> dictionary with keys = score and values = move coordinate (x,y)
        symbol --> X or Y strings
        Output:
        The best move --> tuple (x,y)
        '''
        if symbol == 'X':
            best_move = max(move_scores, key=lambda key: move_scores[key])
        elif symbol == 'O':
            best_move = min(move_scores, key=lambda key: move_scores[key])

        return best_move



#----------------------------------------------------------------------------------------------------------
# Check to see if someone has won or there is a tie
    def has_won(self, board, last_move, symbol):
        '''
        1. Only check the last coordinate.
        2. Create 4 lists which consist of the X, 0 or space, +- k either side of the last coordinate for each trajectory,
        horizontal, vertical and both diagonals.
        3. Iterate through these lists and see if there is a string longer than k of the required symbol.
        '''
        if last_move == None:
            return False

        last_X = last_move[0]
        last_Y = last_move[1]


        vertical_list = []
        horizontal_list = []
        rdiag_list = []
        ldiag_list = []


        # only need to check k-1 either side of the last move
        for i in range(self.k + (self.k-1)):
            vertical_list.append(self.return_full_board(board, last_Y-1, (last_X-1)-(self.k-1)+i))
            horizontal_list.append(self.return_full_board(board, (last_Y-1)-(self.k-1)+i, last_X-1))
            rdiag_list.append(self.return_full_board(board, (last_Y-1)-(self.k-1)+i, (last_X-1)-(self.k-1)+i))
            ldiag_list.append(self.return_full_board(board, (last_Y-1)+(self.k-1)-i, (last_X-1)-(self.k-1)+i))

        vertical_count = 0
        horizontal_count = 0
        rdiag_count = 0
        ldiag_count = 0

        # now we check each of the lists to see if we have k in a row. Here we rest the count to 0 if we encounter a
        # symbol which is not the players
        for i in range(self.k + (self.k-1)):

            if vertical_list[i] == symbol:
                vertical_count += 1
            if vertical_list[i] != symbol:
                vertical_count = 0

            if horizontal_list[i] == symbol:
                horizontal_count += 1
            if horizontal_list[i] != symbol:
                horizontal_count = 0

            if rdiag_list[i] == symbol:
                rdiag_count += 1
            if rdiag_list[i] != symbol:
                rdiag_count = 0

            if ldiag_list[i] == symbol:
                ldiag_count += 1
            if ldiag_list[i] != symbol:
                ldiag_count = 0

            if vertical_count == self.k or horizontal_count == self.k or rdiag_count == self.k or ldiag_count == self.k:
                return True

        return False

    def draw(self, board):
        '''
        if all spaces have an X or O then the game is a draw. (function only ran after the has_won function)
        '''
        count = 0
        for i in range(self.m):
            for j in range(self.n):
                if board[j][i] == 'X' or board[j][i] == 'O':
                    count += 1
        if count == self.m * self.n:
            return True
        else:
            return False

    def construct_full_board(self, player_positions, computer_positions):

        '''
        combine the players and the computers move coordinates into one board which is easier to check
        '''
        full_board = [[' ' for _ in range(self.m)] for _ in range(self.n)]

        for x, y in player_positions:
            full_board[y - 1][x - 1] = 'X'
        for x, y in computer_positions:
            full_board[y - 1][x - 1] = 'O'
        return full_board

    def return_full_board(self, board,  index1, index2):
        '''
        function that deals with indexing off the board. If we do index off the board simply return None so
        that nothing is added to the list.
        '''
        if 0 <= index2 < (self.m) and 0 <= index1 < (self.n):
            return board[index1][index2]
        else:
            return None
#----------------------------------------------------------------------------------------------------------
# printing the board
    def drawboard(self):
        array_board = [[' ' for _ in range(self.m)] for _ in range(self.n)]

        for x, y in self.player_positions:
            array_board[y - 1][x - 1] = 'X'
        for x, y in self.computer_positions:
            array_board[y - 1][x - 1] = 'O'


        board_str = self._get_board_string_from_array_chars(array_board)
        print(board_str)

    def drawboard_2(self, player_positions, computer_positions):
        '''
        Used for checking the progress of the minimax tree search.
        '''
        array_board = [[' ' for _ in range(self.m)] for _ in range(self.n)]

        for x, y in player_positions:
            array_board[y - 1][x - 1] = 'X'
        for x, y in computer_positions:
            array_board[y - 1][x - 1] = 'O'


        board_str = self._get_board_string_from_array_chars(array_board)
        print(board_str)

    def _get_board_string_from_array_chars(self, array_board: List[List[str]]) -> str:
        list_lines = []

        array_first_line = [chr(code + OFFSET_UPPER_CASE_CHAR_CONVERSION) for code in range(1, self.m + 1)]
        first_line = ' ' * 6 + (' ' * 5).join(array_first_line) + ' \n'

        for index_line, array_line in enumerate(array_board, 1):
            number_spaces_before_line = 2 - len(str(index_line))
            space_before_line = number_spaces_before_line * ' '
            list_lines.append(f'{space_before_line}{index_line} |  ' + '  |  '.join(array_line) + '  |\n')

        line_dashes = '   ' + '-' * 6 * self.m + '-\n'

        board_str = first_line + line_dashes + line_dashes.join(list_lines) + line_dashes

        return board_str


if __name__ == '__main__':

    game = Game(3, 3, 3)
    game.play(ab_flag=True)
