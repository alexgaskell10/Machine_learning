from battleship.board import Board, BoardAutomatic
from battleship.game import Game
from battleship.player import PlayerUser, PlayerRandom, PlayerAutomatic
from battleship.ship import Ship


def example_two_players_users():
    # Creating the ships MANUALLY for the 2 players Alice and Bob

    list_ships_player_alice = [
        Ship(coord_start=(3, 1), coord_end=(3, 5)),  # length = 5
        Ship(coord_start=(9, 7), coord_end=(9, 10)),  # length = 4
        Ship(coord_start=(1, 9), coord_end=(3, 9)),  # length = 3
        Ship(coord_start=(5, 2), coord_end=(6, 2)),  # length = 2
        Ship(coord_start=(8, 3), coord_end=(8, 3)),  # length = 1
    ]

    list_ships_player_bob = [
        Ship(coord_start=(5, 8), coord_end=(9, 8)),  # length = 5
        Ship(coord_start=(5, 4), coord_end=(8, 4)),  # length = 4
        Ship(coord_start=(3, 1), coord_end=(5, 1)),  # length = 3

        Ship.get_ship_from_str_coordinates(coord_str_start='F10',
                                           coord_str_end='G10'),  # Another way of creating a Ship

        Ship.get_ship_from_str_coordinates(coord_str_start='A4',
                                           coord_str_end='A4'),  # Another way of creating a Ship
    ]

    # Creating their boards
    board_player_alice = Board(list_ships_player_alice)
    board_player_bob = Board(list_ships_player_bob)

    # Creating the players
    player_alice = PlayerUser(board_player_alice, name_player="Alice")
    player_bob = PlayerUser(board_player_bob, name_player="Bob")

    # Creating and launching the game
    game = Game(player_1=player_alice,
                player_2=player_bob)

    game.play()


def example_user_manual_board_vs_full_automatic():
    # Creating the ships MANUALLY for the User (Alice)

    list_ships_player_alice = [
        Ship(coord_start=(3, 1), coord_end=(3, 5)),  # length = 5
        Ship(coord_start=(9, 7), coord_end=(9, 10)),  # length = 4
        Ship(coord_start=(1, 9), coord_end=(3, 9)),  # length = 3
        Ship(coord_start=(5, 2), coord_end=(6, 2)),  # length = 2
        Ship(coord_start=(8, 3), coord_end=(8, 3)),  # length = 1
    ]

    # Creating her boards
    board_player_alice = Board(list_ships_player_alice)

    # Creating her player
    player_alice = PlayerUser(board_player_alice, name_player="Alice")

    # Creating a Random Player Bob, its board is automatically created randomly
    player_bob = PlayerRandom(name_player="Bob")

    # Creating and launching the game
    game = Game(player_1=player_alice,
                player_2=player_bob)

    game.play()


def example_user_automatic_board_vs_full_automatic():
    # Creating the Board Automatically for the User (Alice)
    board_player_alice = BoardAutomatic()

    # Creating her player
    player_alice = PlayerUser(board_player_alice, name_player="Alice")

    # Creating a Random Player Bob, its board is automatically created randomly
    player_bob = PlayerRandom(name_player="Bob")

    # Creating and launching the game
    game = Game(player_1=player_alice,
                player_2=player_bob)

    game.play()



def player_automatic_vs_player_random():
    '''
    Plays a player of PlayerAutomatic class against a player of PlayerRandom class
    :return winner: string representing the name of the winner (so that multiple
    trials can be run and we can track the number of wins for each player)
    '''

    # Creating a Random Player Bob, its board is automatically created randomly
    player_bob = PlayerRandom(name_player="Bob")

    # Creating an AI player to play automatically, its board is created automatically
    player_auto = PlayerAutomatic(player_bob, name_player='Alex')

    # Creating and launching the game
    game = Game(player_1=player_auto,
                player_2=player_bob)

    winner = game.play()

    return winner


def player_automatic_vs_player_automatic():
    '''
    Plays a player of PlayerAutomatic class against a player of PlayerRandom class
    :return winner: string representing the name of the winner (so that multiple
    trials can be run and we can track the number of wins for each player)
    '''

    # Creating an AI player to play automatically, its board is created automatically
    player_alice = PlayerAutomatic(name_player='Alice')

    # Creating a Random Player Bob, its board is automatically created randomly
    player_bob = PlayerAutomatic(player_alice, name_player="Bob")

    player_alice.opponent = player_bob

    # Creating and launching the game
    game = Game(player_1=player_alice,
                player_2=player_bob)

    winner = game.play()

    return winner
