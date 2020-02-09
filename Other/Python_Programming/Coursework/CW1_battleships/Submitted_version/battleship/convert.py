from typing import Tuple

from battleship.board import Board, OFFSET_UPPER_CASE_CHAR_CONVERSION
# from board import Board, OFFSET_UPPER_CASE_CHAR_CONVERSION


def get_str_coordinates_from_tuple(coord_x: int,
                                   coord_y: int):
    return chr(coord_x + OFFSET_UPPER_CASE_CHAR_CONVERSION) + str(coord_y)


def get_tuple_coordinates_from_str(coord_str: str) -> Tuple[int, int]:
    coord_str = coord_str.strip()

    if not 2 <= len(coord_str) <= 3:
        raise ValueError(f"The position provided '{coord_str}' is not valid")

    coord_1, coord_2 = coord_str[0], coord_str[1:]

    coord_1 = ord(coord_1) - OFFSET_UPPER_CASE_CHAR_CONVERSION
    coord_2 = int(coord_2)

    if not (0 < coord_1 <= Board.SIZE_X and 0 < coord_2 <= Board.SIZE_Y):
        raise ValueError(f"The position provided '{coord_str}' is not valid")

    return coord_1, coord_2


if __name__ == '__main__':
    print(get_tuple_coordinates_from_str('J9'))
