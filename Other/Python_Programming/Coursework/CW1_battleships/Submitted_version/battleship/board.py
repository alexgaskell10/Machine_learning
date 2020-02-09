from typing import List, Tuple

from battleship.ship import Ship
# from ship import Ship
from itertools import combinations
from random import randint, seed, choice

OFFSET_UPPER_CASE_CHAR_CONVERSION = 64

class Board(object):
    """
    Class representing the board of the player. Interface between the player and its ships.
    """
    SIZE_X = 10  # length of the rectangular board, along the x axis
    SIZE_Y = 10  # length of the rectangular board, along the y axis

    # dict: length -> number of ships of that length
    DICT_NUMBER_SHIPS_PER_LENGTH = {1: 1,
                                    2: 1,
                                    3: 1,
                                    4: 1,
                                    5: 1}

    def __init__(self,
                 list_ships: List[Ship]):
        """
        :param list_ships: list of ships for the board.
        :raise ValueError if the list of ships is in contradiction with Board.DICT_NUMBER_SHIPS_PER_LENGTH.
        :raise ValueError if there are some ships that are too close from each other
        """

        self.list_ships = list_ships
        self.set_coordinates_previous_shots = set()

        if not self.lengths_of_ships_correct():
            total_number_of_ships = sum(self.DICT_NUMBER_SHIPS_PER_LENGTH.values())

            error_message = f"There should be {total_number_of_ships} ships in total:\n"

            for length_ship, number_ships in self.DICT_NUMBER_SHIPS_PER_LENGTH.items():
                error_message += f" - {number_ships} of length {length_ship}\n"

            raise ValueError(error_message)

        if self.are_some_ships_too_close_from_each_other():
            raise ValueError("There are some ships that are too close from each other.")

    def has_no_ships_left(self) -> bool:
        """
        :return: True if and only if all the ships on the board have sunk.
        """
        ships_sunk = [True for s in self.list_ships if s.has_sunk()].count(True)
        return ships_sunk == len(self.DICT_NUMBER_SHIPS_PER_LENGTH)

    def is_attacked_at(self, coord_x: int, coord_y: int) -> Tuple[bool, bool]:
        """
        The board receives an attack at the position (coord_x, coord_y).
        - if there is no ship at that position -> nothing happens
        - if there is a ship at that position -> it is damaged at that coordinate

        :param coord_x: integer representing the projection of a coordinate on the x-axis
        :param coord_y: integer representing the projection of a coordinate on the y-axis
        :return: a tuple of bool variables (is_ship_hit, has_ship_sunk) where:
                    - is_ship_hit is True if and only if the attack was performed at a set of coordinates where an
                    opponent's ship is.
                    - has_ship_sunk is True if and only if that attack made the ship sink.
        """
        # Save shot
        self.set_coordinates_previous_shots.add((coord_x, coord_y))

        # Check each ship to see if it has been hit
        ship_damages = []
        for s in self.list_ships:
            s.gets_damage_at(coord_x, coord_y)
            ship_hit = (coord_x, coord_y) in s.set_coordinates_damages
            ship_damages.append((ship_hit, s.has_sunk()))

        return ship_damages

    def print_board_with_ships_positions(self) -> None:
        array_board = [[' ' for _ in range(self.SIZE_X)] for _ in range(self.SIZE_Y)]

        for x_shot, y_shot in self.set_coordinates_previous_shots:
            array_board[y_shot - 1][x_shot - 1] = 'O'

        for ship in self.list_ships:
            if ship.has_sunk():
                for x_ship, y_ship in ship.set_all_coordinates:
                    array_board[y_ship - 1][x_ship - 1] = '$'
                continue

            for x_ship, y_ship in ship.set_all_coordinates:
                array_board[y_ship - 1][x_ship - 1] = 'S'

            for x_ship, y_ship in ship.set_coordinates_damages:
                array_board[y_ship - 1][x_ship - 1] = 'X'

        board_str = self._get_board_string_from_array_chars(array_board)

        print(board_str)

    def print_board_without_ships_positions(self) -> None:
        array_board = [[' ' for _ in range(self.SIZE_X)] for _ in range(self.SIZE_Y)]

        for x_shot, y_shot in self.set_coordinates_previous_shots:
            array_board[y_shot - 1][x_shot - 1] = 'O'

        for ship in self.list_ships:
            if ship.has_sunk():
                for x_ship, y_ship in ship.set_all_coordinates:
                    array_board[y_ship - 1][x_ship - 1] = '$'
                continue

            for x_ship, y_ship in ship.set_coordinates_damages:
                array_board[y_ship - 1][x_ship - 1] = 'X'

        board_str = self._get_board_string_from_array_chars(array_board)

        print(board_str)

    def _get_board_string_from_array_chars(self, array_board: List[List[str]]) -> str:
        list_lines = []

        array_first_line = [chr(code + OFFSET_UPPER_CASE_CHAR_CONVERSION) for code in range(1, self.SIZE_X + 1)]
        first_line = ' ' * 6 + (' ' * 5).join(array_first_line) + ' \n'

        for index_line, array_line in enumerate(array_board, 1):
            number_spaces_before_line = 2 - len(str(index_line))
            space_before_line = number_spaces_before_line * ' '
            list_lines.append(f'{space_before_line}{index_line} |  ' + '  |  '.join(array_line) + '  |\n')

        line_dashes = '   ' + '-' * 6 * self.SIZE_X + '-\n'

        board_str = first_line + line_dashes + line_dashes.join(list_lines) + line_dashes

        return board_str

    def lengths_of_ships_correct(self) -> bool:
        """
        :return: True if and only if there is the right number of ships of each length, according to
        Board.DICT_NUMBER_SHIPS_PER_LENGTH
        """
        ship_lengths = sorted([s.length() for s in self.list_ships])
        required_ship_lengths = sorted(self.DICT_NUMBER_SHIPS_PER_LENGTH)

        ship_lengths_correct = ship_lengths == required_ship_lengths

        return ship_lengths_correct

    def are_some_ships_too_close_from_each_other(self) -> bool:
        """
        :return: True if and only if there are at least 2 ships on the board that are near each other.
        """
        close_ships = [True for s1,s2 in combinations(self.list_ships, 2) if s1.is_near_ship(s2)].count(True)
        return close_ships > 0


class BoardAutomatic(Board):
    def __init__(self):
        super().__init__(self.generate_ships_automatically())

    def generate_ships_automatically(self) -> List[Ship]:
        """
        :return: A list of automatically (randomly) generated ships for the board
        """
        # Algo:
        # 1. Randomly choose a ship of length x from the interval [1,5] with random orientation
        # 2. Randomly choose start coordinate
        # 3. Ensure this ship positing is valid (i.e. it isn't close to other ships)
        # 4. Repeat steps 1-3 until all ships have been generated on the board

        list_ships = []

        i = 0
        num_ships_flag = False
        while not num_ships_flag:

            valid_ship_flag = False
            while not valid_ship_flag:

                ship_id, ship_alignment = self.define_ship_params(list_ships)       # Generate parameters for the ship

                coord_start, coord_end = self.generate_ship_coords(ship_id, ship_alignment)     # Find start and end locations for the ship

                ship = self.create_ship(coord_start, coord_end)

                valid_ship_flag = self.validate_ship_closeness(ship, list_ships)        # Ensure the ship isn't close to others

                i += 1

            list_ships.append(ship)     # Save ship

            num_ships_flag = (len(list_ships) == 5)

            i += 1

        return list_ships



    def define_ship_params(self, list_ships):
        '''
        Randomly generates size and alignment of a ship

        :param list_ships: list of ships which have already been created
        :return ship_id: integer denoting size
        :return ship_alignment: either 'vertical' or 'horizonatal'; specifies alignment of ship
        '''
        ship_lengths = [s.length() for s in list_ships]
        ship_id = choice([i for i in range(1,6) if i not in ship_lengths])
        if randint(0,1):
            ship_alignment = 'vertical'
        else:
            ship_alignment = 'horizontal'
        return ship_id, ship_alignment

    def generate_ship_coords(self, ship_id, ship_alignment):
        '''
        Randomly generates a start coordinate for the ship (and therefore an end coord given
        length and alignment of ship).

        :param ship_id: size of ship2
        :param ship_alignment: either 'vertical' or 'horizonatal'; specifies alignment of ship
        :return coord_start: tuple containing start coords (x,y)
        :return coord_end: tuple containing end coords (x,y)
        '''
        # Limit board so only valid cells remain (i.e. a vertical ship of length
        # 5 cannot begin in rows 7-10 so these can be discarded immediately)
        x_coord_max, y_coord_max = self.SIZE_X, self.SIZE_X

        if ship_alignment == 'horizontal':
            x_coord_max -= ship_id + 1
        else:
            y_coord_max -= ship_id + 1

        # Define ship start and end coordinates
        coord_start = (randint(1,x_coord_max), randint(1,y_coord_max))
        if ship_alignment == 'horizontal':
            coord_end = (coord_start[0] + ship_id - 1, coord_start[1])
        else:
            coord_end = (coord_start[0], coord_start[1] + ship_id - 1)

        return coord_start, coord_end

    def create_ship(self, coord_start, coord_end):
        '''
        :return coord_start: tuple containing start coords (x,y)
        :return coord_end: tuple containing end coords (x,y)

        '''
        return Ship(coord_start, coord_end)

    def validate_ship_closeness(self, ship, list_ships):
        '''
        Validate whether ship is near to other ships.

        :param ship: object of class ship
        :return: True if and only if ship is not close to other ships
        '''
        ships_close = [s1.is_near_ship(s2) for s1,s2 in combinations(list_ships + [ship], 2)].count(True)
        return ships_close == 0

if __name__ == '__main__':
    # SANDBOX for you to play and test your functions
    # list_ships = [
    #     Ship(coord_start=(1, 1), coord_end=(1, 1)),
    #     Ship(coord_start=(2, 3), coord_end=(2, 4)),
    #     Ship(coord_start=(5, 3), coord_end=(5, 5)),
    #     Ship(coord_start=(7, 1), coord_end=(7, 4)),
    #     Ship(coord_start=(9, 3), coord_end=(9, 7)),
    # ]

    # board = Board(list_ships)
    # # print(board.has_no_ships_left())
    board = BoardAutomatic()
    board.generate_ships_automatically()
    board.print_board_with_ships_positions()

    # board.print_board_without_ships_positions()
    # print(board.lengths_of_ships_correct())
    # ship_coords = []
    # for s in list_ships:
    #     for x,y in s.get_all_coordinates():
    #         ship_coords.append((x,y))
    #         board.is_attacked_at(x,y)
    #     print(s.has_sunk())
    # print(board.has_no_ships_left())
    # print(board.is_attacked_at(1, 1),'\n',
    #       board.is_attacked_at(5, 5),'\n',
    #       board.is_attacked_at(5, 3))
    # print(list_ships[2].set_coordinates_damages)
    # print(board.set_coordinates_previous_shots)
    # print(board.lengths_of_ships_correct())
    # print(board.are_some_ships_too_close_from_each_other())
