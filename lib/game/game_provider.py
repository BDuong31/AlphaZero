from lib.game.caro_19x19.caro_19x19 import Caro19x19
from lib.game.caro_17x17.caro_17x17 import Caro17x17
from lib.game.caro_15x15.caro_15x15 import Caro15x15
from lib.game.caro_13x13.caro_13x13 import Caro13x13
from lib.game.caro_9x9.caro_9x9 import Caro9x9
from lib.game.caro_7x7.caro_7x7 import Caro7x7
from lib.game.caro_5x5.caro_5x5 import Caro5x5
from lib.game.tictactoe.tictactoe import TicTacToe

def add_game_argument(parser):
    """
    Thêm đối số --game vào trình phân tích đối số với các trò chơi có sẵn    
    
    Đối số:
        parser (argparse.ArgumentParser): Trình phân tích đối số để thêm đối số trò chơi
    """
    parser.add_argument("-g", "--game", required=True, choices=['0', '1', '2', '3', '4', '5', '6', '7'],
                        help="Loại trò chơi: 0 - TicTacToe, 1 - Caro 5x5, 2 - Caro 7x7, 3 - Caro 9x9, 4 - Caro 13x13, 5 - Caro 15x15, 6 - Caro 17x17, 7 - Caro 19x19")
def get_game(args):
    """
    Trả về trò chơi dựa trên đối số đã phân tích
    
    Đối số:
        args (argparse.Namespace): Đối tượng chứa các đối số đã phân tích
    
    Trả về:
        game: Trò chơi tương ứng với đối số đã phân tích
    """
    game_type = args.game
    if game_type == '0':
        return TicTacToe()
    elif game_type == '1':
        return Caro5x5()
    elif game_type == '2':
        return Caro7x7()
    elif game_type == '3':
        return Caro9x9()
    elif game_type == '4':
        return Caro13x13()
    elif game_type == '5':
        return Caro15x15()
    elif game_type == '6':
        return Caro17x17()
    elif game_type == '7':
        return Caro19x19()
    else:
        raise ValueError("Trò chơi không hợp lệ")