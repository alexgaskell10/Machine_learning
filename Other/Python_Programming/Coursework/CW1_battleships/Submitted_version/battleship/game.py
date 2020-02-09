import random

from battleship.player import Player


class Game(object):
    """
    Perform game simulations.
    It is also in this class that the general rules of the game are defined, such as:
    - if a ship is hit, the player who performed the attack has the right to play another time.
    - if all the opponent's ships have sunk, the game stops, and the results are printed
    """

    def __init__(self,
                 player_1: Player,
                 player_2: Player):
        """
        :param player_1: First competitor (Player object)
        :param player_2: Second competitor (Player object)
        """
        self.player_1 = player_1
        self.player_2 = player_2

    def play(self) -> None:
        """
        Simulates an entire game. Prints necessary information (boards without ships, positions under attack... )
        """

        # Chooses position first turn
        if random.choice([True, False]):
            print(f"{self.player_1} starts the game.")
            player_turn = self.player_1
            player_opponent = self.player_2
        else:
            print(f"{self.player_2} starts the game.")
            player_turn = self.player_2
            player_opponent = self.player_1

        # Simulates the game, until a player has lost
        while not self.player_1.has_lost() and not self.player_2.has_lost():
            print("-" * 75 + "\n"* 5 + "-" * 75 + "\n")
            is_ship_hit = None

            # if an opponent's ship is hit, the player is allowed to play another time.
            while is_ship_hit is None or is_ship_hit:

                is_ship_hit, _ = player_turn.attacks(player_opponent)

                if self.player_1.has_lost() or self.player_2.has_lost():
                    break

                if is_ship_hit:
                    print("-" * 75)

            player_turn, player_opponent = player_opponent, player_turn  # Now it's the opponent's turn

        winner = self._print_results()
        return winner

    def _print_results(self):
        print("-" * 75 + "\n" * 5 + "-" * 75 + "\n")
        print(f"Here is the final state of {self.player_1}'s board:\n ")
        self.player_1.print_board_with_ships()

        print("-" * 75 + "\n")
        print(f"Here is the final state of {self.player_2}'s board:\n")
        self.player_2.print_board_with_ships()

        print("-" * 75 + "\n" * 3)
        if self.player_1.has_lost():
            print(f"--- {self.player_2} WINS THE GAME ---")
            return str(self.player_2)  # So we can track who is the winner
        else:
            print(f"--- {self.player_1} WINS THE GAME ---")
            return str(self.player_1)  # So we can track who is the winner
