import random
from typing import Tuple
from itertools import product
from operator import add

from battleship.board import Board, BoardAutomatic
from battleship.ship import Ship
from battleship.convert import get_tuple_coordinates_from_str, get_str_coordinates_from_tuple

# from board import Board, BoardAutomatic
# from ship import Ship
# from convert import get_tuple_coordinates_from_str, get_str_coordinates_from_tuple

class Player(object):
    """
    Class representing the player
    - chooses where to perform an attack
    """
    index_player = 0

    def __init__(self,
                 board: Board,
                 name_player: str = None,
                 ):
        Player.index_player += 1

        self.board = board

        if name_player is None:
            self.name_player = "player_" + str(self.index_player)
        else:
            self.name_player = name_player

    def __str__(self):
        return self.name_player

    def attacks(self,
                opponent) -> Tuple[bool, bool]:
        """
        :param opponent: object of class Player representing the person to attack
        :return: a tuple of bool variables (is_ship_hit, has_ship_sunk) where:
                    - is_ship_hit is True if and only if the attack was performed at a set of coordinates where an
                    opponent's ship is.
                    - has_ship_sunk is True if and only if that attack made the ship sink.
        """
        assert isinstance(opponent, Player)

        print(f"Here is the current state of {opponent}'s board before {self}'s attack:\n")
        opponent.print_board_without_ships()

        if isinstance(self, PlayerAutomatic):
            coord_x, coord_y = self.auto_select_coordinates_to_attack()

            print(f"{self} attacks {opponent} "
              f"at position {get_str_coordinates_from_tuple(coord_x, coord_y)}")

            is_ship_hit, has_ship_sunk = self.auto_attack(coord_x, coord_y)

        else:
            coord_x, coord_y = self.select_coordinates_to_attack(opponent)

            print(f"{self} attacks {opponent} "
              f"at position {get_str_coordinates_from_tuple(coord_x, coord_y)}")

            is_ship_hit, has_ship_sunk = opponent.is_attacked_at(coord_x, coord_y)

        if has_ship_sunk:
            print(f"\nA ship of {opponent} HAS SUNK. {self} can play another time.")
        elif is_ship_hit:
            print(f"\nA ship of {opponent} HAS BEEN HIT. {self} can play another time.")
        else:
            print("\nMissed".upper())

        return is_ship_hit, has_ship_sunk

    def is_attacked_at(self,
                       coord_x: int,
                       coord_y: int
                       ) -> Tuple[bool, bool]:
        """
        :param coord_x: integer representing the projection of a coordinate on the x-axis
        :param coord_y: integer representing the projection of a coordinate on the y-axis
        :return: a tuple of bool variables (is_ship_hit, has_ship_sunk) where:
                    - is_ship_hit is True if and only if the attack was performed at a set of coordinates where a
                    ship is (on the board owned by the player).
                    - has_ship_sunk is True if and only if that attack made the ship sink.
        """
        # Sort list so that, if attack was a hit, it will be the final element of the list
        attack_results = sorted(self.board.is_attacked_at(coord_x, coord_y))
        return attack_results[-1]


    def select_coordinates_to_attack(self, opponent) -> Tuple[int, int]:
        """
        Abstract method, for choosing where to perform the attack
        :param opponent: object of class Player representing the player under attack
        :return: a tuple of coordinates (coord_x, coord_y) at which the next attack will be performed
        """
        raise NotImplementedError

    def has_lost(self) -> bool:
        """
        :return: True if and only if all the ships of the player have sunk
        """
        return self.board.has_no_ships_left()

    def print_board_with_ships(self):
        self.board.print_board_with_ships_positions()

    def print_board_without_ships(self):
        self.board.print_board_without_ships_positions()


class PlayerUser(Player):
    """
    Player representing a user playing manually
    """

    def select_coordinates_to_attack(self, opponent: Player) -> Tuple[int, int]:
        """
        Overrides the abstract method of the parent class.
        :param opponent: object of class Player representing the player under attack
        :return: a tuple of coordinates (coord_x, coord_y) at which the next attack will be performed
        """
        print(f"It is now {self}'s turn.")

        while True:
            try:
                coord_str = input('coordinates target = ')
                coord_x, coord_y = get_tuple_coordinates_from_str(coord_str)
                return coord_x, coord_y
            except ValueError as value_error:
                print(value_error)



class PlayerRandom(Player):
    def __init__(self, name_player: str = None):
        board = BoardAutomatic()
        self.set_positions_previously_attacked = set()
        self.last_attack_coord = None
        self.list_ships_opponent_previously_sunk = []
        self.playertype = 'random'

        super().__init__(board, name_player)

    def select_coordinates_to_attack(self, opponent: Player) -> tuple:
        position_to_attack = self.select_random_coordinates_to_attack()

        self.set_positions_previously_attacked.add(position_to_attack)
        self.last_attack_coord = position_to_attack
        return position_to_attack

    def select_random_coordinates_to_attack(self) -> tuple:
        has_position_been_previously_attacked = True
        is_position_near_previously_sunk_ship = True
        coord_random = None

        while has_position_been_previously_attacked or is_position_near_previously_sunk_ship:
            coord_random = self._get_random_coordinates()

            has_position_been_previously_attacked = coord_random in self.set_positions_previously_attacked
            is_position_near_previously_sunk_ship = self._is_position_near_previously_sunk_ship(coord_random)

        return coord_random

    def _get_random_coordinates(self) -> tuple:
        coord_random_x = random.randint(1, self.board.SIZE_X)
        coord_random_y = random.randint(1, self.board.SIZE_Y)

        coord_random = (coord_random_x, coord_random_y)

        return coord_random

    def _is_position_near_previously_sunk_ship(self, coord: tuple) -> bool:
        for ship_opponent in self.list_ships_opponent_previously_sunk:  # type: Ship
            if ship_opponent.has_sunk() and ship_opponent.is_near_coordinate(*coord):
                return True
        return False



class PlayerAutomatic(PlayerRandom):
    """
    Player playing automatically using a strategy.
    """

    def __init__(self, opponent=None, name_player: str = None):
        super().__init__(name_player)
        self.opponent = opponent
        self.list_successful_attack_coords = []
        self.list_target_ship_coords = []

    def auto_select_coordinates_to_attack(self) -> (int, int):
        """
        Overrides the abstract method of the parent class. Chooses coordinates strategically if and only if
        the last attack hit (and did not sink) opponent's ship; otherwise, attack coordinates are
        chosen randomly.
        :param opponent: object of class Player representing the player under attack
        :return: a tuple of coordinates (coord_x, coord_y) at which the next attack will be performed
        """

        if len(self.list_target_ship_coords) == 0:      # Check if we currently have a target ship
            attack_coord_x, attack_coord_y = self.select_coordinates_to_attack(self.opponent)
        else:
            attack_coord_x, attack_coord_y = self.auto_select_smart_attack_coords()

        return attack_coord_x, attack_coord_y

    def auto_attack(self, attack_coord_x, attack_coord_y) -> (bool, bool):
        '''
        Method to attack opponent and save attack coords if attack successful. Given a set of attack
        coords, attack the opponent there and take the following actions:
        - If attack hit, append coord to list of successful attack coords
        - If attack hit an opponent ship but did not sink it, this opponent ship is now the "target"
        (i.e. we will continue aiming for this ship until it is sunk). Therefore also append last attack
        coord to the list of attack coords for this target ship
        - If attack sunk the ship, we need to find a new target ship, so clear the list of attack coords
        for the target ship

        :return last_attack_hit: True if last attack hit opponent ship
        :return last_attack_sunk_ship: True if last attack sunk opponent ship
        '''
        self.last_attack_coord = attack_coord_x, attack_coord_y

        last_attack_hit, last_attack_sunk_ship = self.opponent.is_attacked_at(self.last_attack_coord[0], self.last_attack_coord[1])

        # Save coord if attack hit enemy ship
        if last_attack_hit:
            self.list_successful_attack_coords.append(self.last_attack_coord)
            self.list_target_ship_coords.append(self.last_attack_coord)

            # If last attack sunk target ship, find a new target
            if last_attack_sunk_ship:

                sunk_ship_start_coord = min(self.list_target_ship_coords, key=sum)
                sunk_ship_end_coord = max(self.list_target_ship_coords, key=sum)

                sunk_ship = Ship(sunk_ship_start_coord, sunk_ship_end_coord)
                for x,y in self.list_target_ship_coords:
                    sunk_ship.gets_damage_at(x,y)

                self.list_ships_opponent_previously_sunk.append(sunk_ship)      # Save coordinates of sunk ships so future attacks will not be near sunk ships

                self.list_target_ship_coords.clear()        # Clear target list so new target can be found

        return last_attack_hit, last_attack_sunk_ship


    def auto_select_smart_attack_coords(self) -> tuple:
        '''
        Method to strategically select next attack coords. This method finds a (valid) adjacent
        coord to the successful attack cords on the target ship. If there have been two successful attacks
        on the current target ship, the alignment of the target ship is also found and used.
        i.e. if opponent has ship at coords (1,1) --> (1,3) and we have already hit coords (1,1) & (1,2),
        the alignment will be determined as vertical so the next attack coords will be either (1,0) or (1,3).
        This method would reject (1,0) as an invalid coord, therefore the next attack coord would be (1,3).

        :return position_to_attack: tuple coordinate for next attack
        '''

        coord_adjustments = [(-1,0),(1,0),(0,-1),(0,1)] # Move one coord left, right, up, down

        alignment = self.auto_opponent_ship_alignment()      # Determine alignment of target ship

        i, valid_coord = 0, False
        while not valid_coord:     # Using above adjustments, loop until it finds a valid new attack cell

            self.list_target_ship_coords.sort(key=sum)      # Sort the previous successful strikes so the adjustment can be made to the end points

            if alignment == 'Vertical':
                if i == 0:
                    i += 2      # Limit coord_adjustments to vertical adjustments only
                    reference_target_attack_coord = self.list_target_ship_coords[0]
                else:
                    reference_target_attack_coord = self.list_target_ship_coords[-1]

            else:
                if i==0:
                    reference_target_attack_coord = self.list_target_ship_coords[0]
                else:
                    reference_target_attack_coord = self.list_target_ship_coords[-1]

            position_to_attack = (reference_target_attack_coord[0] + coord_adjustments[i][0],      # Get new attack coord by applying adjustment to previous successful target attack coord
                                    reference_target_attack_coord[1] + coord_adjustments[i][1])

            valid_coord = self.verify_coord(position_to_attack)      # Verify if poposed attack coordinate is valid before proceeding
            i += 1

        self.set_positions_previously_attacked.add(position_to_attack)
        self.last_attack_coord = position_to_attack
        return position_to_attack

    def auto_opponent_ship_alignment(self) -> str:
        '''
        Determine alignment for target opposition ship. If the target ship has
        been hit more then once, we can determine its alignment. If not, reuturn 'No alignment''.
        :return target_ship_alignment: string describing alignment of target ship
        '''
        if len(self.list_target_ship_coords) < 2:       # Verify the ship has been hit more than once
            target_ship_alignment = 'No alignmnent'

        else:
            distance_between_last_two_hits = [abs(a1 - a2) for a1, a2 in zip(self.list_target_ship_coords[0],
                                                                            self.list_target_ship_coords[1])]

            # Determine target ship alignment
            if distance_between_last_two_hits[0] == 1:
                target_ship_alignment = 'Horizontal'
            else:
                target_ship_alignment = 'Vertical'

        return target_ship_alignment

    def verify_coord(self, proposed_attack_coord) -> bool:
        '''
        Check if a coordinate is valid to attack.
        :returns valid_coord: reutnfalse if and only if either of the following (returns true otherwise):
            1. coordinate is off the board
            2. coordinate has been attacked already
            3. coordinate is close to a previously sunk ship

        '''
        valid_coord = True
        if max(proposed_attack_coord) > self.board.SIZE_X or min(proposed_attack_coord) < 1 or \
                proposed_attack_coord in self.set_positions_previously_attacked or \
                self._is_position_near_previously_sunk_ship(proposed_attack_coord):

            valid_coord = False

        return valid_coord


if __name__ == '__main__':
    # SANDBOX for you to play and test your functions
    list_ships = [
        Ship(coord_start=(1, 1), coord_end=(1, 1)),
        Ship(coord_start=(3, 3), coord_end=(3, 4)),
        Ship(coord_start=(5, 3), coord_end=(5, 5)),
        Ship(coord_start=(7, 1), coord_end=(7, 4)),
        Ship(coord_start=(9, 3), coord_end=(9, 7)),
    ]

    # for i in [31]:#range(30,33):
    #     random.seed(i)
    #
    #     board_opponent = BoardAutomatic()
    #     player_opponent = PlayerRandom()
    #
    #     player_auto = PlayerAutomatic(player_opponent)
    #
    #     # player_auto.auto_attack(5,4)
    #     # player_auto.auto_attack(5,3)
    #
    #     big_ship = sorted(player_opponent.board.list_ships, key=lambda x: x.length())[-1]
    #     big_ship_coordinates = big_ship.get_all_coordinates()
    #     first_attack_coord = random.choice(list(big_ship_coordinates))
    #     print(big_ship_coordinates, first_attack_coord)
    #
    #     player_auto.auto_attack(first_attack_coord[0],first_attack_coord[1])
    #     # player_opponent.board.print_board_with_ships_positions()
    #
    #     for i in range(7):
    #         player_auto.auto_attack()
    #         player_opponent.board.print_board_with_ships_positions()
    #
    #     player_opponent.board.print_board_with_ships_positions()
    #
    #     print(player_auto.list_ships_opponent_previously_sunk)
    #     print(player_auto._is_position_near_previously_sunk_ship((3,2)))

    # player_auto.auto_select_coordinates_to_attack()

    # print(player_auto.opponent._is_position_near_previously_sunk_ship((1,1)))

    # for s in player_opponent.board.list_ships:
        # print(s.set_coordinates_damages)
    # player = PlayerUser(board)
    # board.print_board_with_ships_positions()
    # print(board.list_ships)
    # print(board.list_ships[1].is_horizontal())
    # for s in board_opponent.list_ships:
    #     for x,y in s.get_all_coordinates():
    #         # ship_coords.append((x,y))
    #         player_auto.auto_attack(x,y)
    #         board_opponent.print_board_with_ships_positions()
    #     # print(s.has_sunk())
    # print(player_opponent.has_lost())
