from lib.game.tictactoe.tictactoe import TicTacToe
from lib.game.caro.caro_19x19 import Caro19x19

def add_game_argument(parser):
    """
    Thêm đối số --game vào trình phân tích đối số với các trò chơi có sẵn    
    
    Đối số:
        parser (argparse.ArgumentParser): Trình phân tích đối số để thêm đối số trò chơi
    """
    parser.add_argument("-g", "--game", required=True, choices=['0', '1'],
                        help="Loại trò chơi: 0 - Caro 19x19, 1 - Tic Tac Toe")
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
        return Caro19x19()
    elif game_type == '1':
        return TicTacToe()
    else:
        raise ValueError("Trò chơi không hợp lệ")