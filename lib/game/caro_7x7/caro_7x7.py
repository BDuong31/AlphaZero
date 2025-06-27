import re
import numpy as np
from lib.game.game import BaseGame
from lib.game.caro_7x7 import caro_7x7_helpers
from typing import List, Tuple

Matrix = List[List[int]]


class Caro7x7(BaseGame):
    """
    Biểu diễn một tập hợp con của trò chơi m,n,k tổng quát
    (https://en.wikipedia.org/wiki/M,n,k-game)
    với n=m và chiến thắng chỉ bằng cách đặt k quân cờ liền kề
    với nhau mà không cần bất kỳ điều kiện nào khác.
    Trạng thái trò chơi có thể được biểu diễn theo 2 cách:
    - Là danh sách nxn các danh sách số nguyên
    trong đó 1 & 0 biểu diễn vị trí quân cờ của một trong hai người chơi và 2 biểu diễn
    một ô vuông trống, còn gọi là dạng "Ma trận"
    - Là số nguyên nxn chữ số, có cùng ý nghĩa với 0, 1 và 2. Vị trí
    của mỗi chữ số tương ứng với chỉ số của mỗi ô vuông trên bàn cờ từ
    trên xuống dưới và từ trái sang phải, ví dụ trên bàn cờ 3x3:
    |0|1|2|
    |3|4|5|
    |6|7|8|
    """

    def __init__(self, n: int = 7, k_to_win: int = 4):
        """
        Tạo một phiên bản của trò chơi.
        Với các đối số mặc định, trò chơi là trò chơi Tic Tac Toe.

        Đối số:
            n (int, tùy chọn): Số ô vuông cho mỗi bên của bàn cờ.
            Mặc định là 5.
            k_to_win (int, tùy chọn): Số lượng quân cờ liên tiếp để thắng.
            Mặc định là 4.
        """
        super().__init__()
        self.board_len = n
        self.k_to_win = k_to_win
        self.player_black = 1
        self.player_white = 0
        self.empty = 2

    @staticmethod
    def flatten_nested_list(nested_list: List[List]) -> List:
        """
        Giải mã danh sách lồng nhau kép thành danh sách phẳng

        Đối số:
            nested_list (List[List])

        Trả về:
            List
        """
        return [item for sublist in nested_list for item in sublist]

    @property
    def initial_state(self) -> int:
        """
        Trạng thái ban đầu của trò chơi ở dạng MCTS. Trạng thái này được sử dụng trong
        utils.play_game để bắt đầu vòng lặp MCTS.
        """
        empty_board = np.full(
            (self.board_len, self.board_len), self.empty).tolist()
        return self.encode_game_state(empty_board)

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """
        Hình dạng của mạng nơ-ron dạng trạng thái trò chơi.
        Đây phải là một bộ các số nguyên. Ví dụ: Đối với trò chơi Tic-Tac-Toe, có thể là
        (2, 3, 3) tức là trạng thái trò chơi được đưa vào mạng nơ-ron
        là một tenxơ 2x3x3: 2 người chơi x 3x3 bàn cờ (bàn cờ nhìn từ phía của mỗi người chơi)
        """
        return (2, self.board_len, self.board_len)

    @property
    def action_space(self) -> int:
        """
        Tổng số tất cả các hành động có thể thực hiện được.
        Đây là hằng số cho mỗi trò chơi biểu diễn tất cả các hành động,
        bất kể chúng có hợp lệ ở mỗi trạng thái trò chơi hay không.

        Được sử dụng để khởi tạo các nút MCTS (xác định các giá trị, avg_values ​​&
        các vectơ số lần truy cập lớn đến mức nào)

        Trả về:
        (int): tổng số các hành động có thể thực hiện được
        """
        return self.board_len ** 2

    def _pad_mcts_state(self, mcts_state_str: str) -> str:
        """
        Vì trạng thái trò chơi ở dạng int có thể có số 0 đứng đầu,
        Chúng ta phải thêm vào nó độ dài thích hợp trước khi chuyển đổi sang dạng Ma trận

        Đối số:
            mcts_state (int): Trạng thái trò chơi ở dạng int

        Trả về:
            str: Chuỗi trạng thái trò chơi ở dạng thân thiện với MCTS, được thêm vào
            độ dài thích hợp với số 0 đứng đầu
        """
        return mcts_state_str.rjust(self.board_len ** 2, "0")

    def encode_game_state(self, state_list: Matrix) -> int:
        """
        Chuyển đổi trạng thái trò chơi từ dạng Ma trận sang dạng thân thiện với MCTS (int)
        nhỏ hơn để lưu trữ và có thể băm để tìm kiếm nhanh trong tìm kiếm MCTS

        Đối số:
            state_list (Ma trận): Trạng thái trò chơi dưới dạng danh sách các danh sách mã thông báo

        Trả về:
            int: Trạng thái trò chơi dưới dạng int
        """
        flattened = self.flatten_nested_list(state_list)
        stringified = [str(i) for i in flattened]
        return int(''.join(stringified))

    def convert_mcts_state_to_list_state(self, mcts_state: int) -> Matrix:
        """
        Chuyển đổi trạng thái trò chơi từ dạng băm nhỏ gọn hơn, tức là thân thiện với MCTS
        sang dạng dễ nghĩ hơn (ma trận mã thông báo)

        Đối số:
            mcts_state (int): Trạng thái trò chơi ở dạng thân thiện với MCTS

        Trả về:
            (Ma trận): Trạng thái trò chơi ở dạng thân thiện với con người hơn (danh sách danh sách mã thông báo)
        """
        # Thêm số ô vuông trên bảng (trong trường hợp có số 0 ở đầu)
        padded = self._pad_mcts_state(str(mcts_state))
        state = []
        for i, c in enumerate(padded):
            if i % self.board_len == 0:
                # new row only every board_len items
                state.append([int(c)])
            else:
                state[i // self.board_len].append(int(c))
        return state

    def possible_moves(self, mcts_state: int) -> List:
        """Trả về chỉ số của các ô trống, từ trái sang phải, từ trên xuống dưới
            |0|1|2|
            |3|4|5|
            |6|7|8|

            Đối số:
                mcts_state (int): Trạng thái trò chơi ở dạng thân thiện với MCTS

            Trả về:
                Iterable: [mô tả]
        """
        padded = self._pad_mcts_state(str(mcts_state))
        return [i for i, c in enumerate(padded) if c == str(self.empty)]

    def invalid_moves(self, mcts_state: int) -> List:
        """
        Trả về các ô không trống

        Đối số:
            mcts_state (int): Trạng thái trò chơi ở dạng thân thiện với MCTS

        Trả về:
            Danh sách: Danh sách các nước đi không hợp lệ (ô đã chiếm)
        """
        padded = self._pad_mcts_state(str(mcts_state))
        return [i for i, c in enumerate(padded) if c != str(self.empty)]

    def _encode_list_state(self, dest_np: np.ndarray, state: Matrix, who_move: int) -> None:
        """
        Mã hóa tại chỗ trạng thái danh sách thành mảng numpy số không
        :param dest_np: mảng dest, dự kiến ​​là số không
        :param who_move: chỉ số người chơi (game.player_white hoặc game.player_black)
        """
        assert dest_np.shape == self.obs_shape

        for row_idx, row in enumerate(state):
            for col_idx, cell in enumerate(row):
                if cell == who_move:
                    dest_np[0, row_idx, col_idx] = 1.0
                elif cell != self.empty:
                    dest_np[1, row_idx, col_idx] = 1.0

    def states_to_training_batch(self, state_ints: List[int],
                                 who_moves_lists: List[int]) -> np.ndarray:
        """
        Chuyển đổi trạng thái trò chơi thành các mảng có thể đưa vào mạng nơ-ron

        Đối số:
            state_ints (List[int]): Danh sách các trạng thái trò chơi ở dạng MCTS
            who_moves_lists (List[int]): Danh sách tương ứng của người chơi có nước đi
            dẫn đến trạng thái trò chơi

        Trả về:
            np.array: mỗi trạng thái trò chơi sẽ được biểu diễn dưới dạng một
            2 x board_len x board_len mảng. Mỗi mảng sẽ phản ánh
            các quân cờ/nước đi của một người chơi có giá trị 1 và có giá trị bằng không ở
            tất cả các vị trí khác (quân cờ của người chơi đối diện và các vị trí trống).
            Đây là cách dữ liệu được đưa vào mạng nơ-ron trong bài báo AlphaZero.
            ví dụ:
            |0|1|
            | |0| với người chơi 0 trở thành:

            [[[1, 0],
            [0, 1]],
            [[0, 1],
            [0, 0]]]
        """
        batch_size = len(state_ints)
        batch = np.zeros((batch_size,) + self.obs_shape, dtype=np.float32)
        for idx, (state, who_move) in enumerate(zip(state_ints, who_moves_lists)):
            converted_state = self.convert_mcts_state_to_list_state(state)
            self._encode_list_state(batch[idx], converted_state, who_move)
        return batch

    def move(self, mcts_state: int, move: int, player: int) -> Tuple[int, bool]:
        """
        Ở trạng thái trò chơi nhất định, thực hiện một nước đi (hợp lệ) của một người chơi được chỉ định

        Đối số:
            mcts_state (int): Trạng thái trò chơi ở dạng MCTS
            move (int): Vị trí nước đi, có thể được tính là chỉ số của ô vuông
            trên bàn cờ từ trên xuống dưới, từ phải sang trái, ví dụ:
            |0|1|2|
            |3|4|5|
            |6|7|8|
            player (int): 0 hoặc 1, người chơi nào đang thực hiện nước đi

        Trả về:
            Tuple[int, bool]: Trạng thái trò chơi mới & nếu trò chơi đã được thắng bởi
            người chơi vừa thực hiện nước đi
        """
        assert player == self.player_white or player == self.player_black
        assert move >= 0 and move <= self.action_space

        board = self.convert_mcts_state_to_list_state(mcts_state)
        row_idx, col_idx = divmod(move, self.board_len)
        board[row_idx][col_idx] = player
        won = caro_7x7_helpers.check_win(
            board, (row_idx, col_idx), self.k_to_win, player)
        new_mcts_state = self.encode_game_state(board)
        return new_mcts_state, won

    def render(self, mcts_state: int) -> str:
        """
        Biểu diễn chuỗi của bảng, để tương tác với người chơi

        Đối số:
            mcts_state (int): Trạng thái trò chơi ở dạng MCTS

        Trả về:
            str: Biểu diễn chuỗi của trạng thái trò chơi
        """
        list_state = self.convert_mcts_state_to_list_state(mcts_state)
        for row_idx, row in enumerate(list_state):
            for col_idx, cell in enumerate(row):
                if cell == self.empty:
                    list_state[row_idx][col_idx] = str(
                        row_idx * self.board_len + col_idx)
                elif cell == self.player_white:
                    list_state[row_idx][col_idx] = "❌"
                elif cell == self.player_black:
                    list_state[row_idx][col_idx] = "⭕"
        # substitute semi-colons with pipe |
        list_str = [f'|{"|".join(row)}|' for row in list_state]
        board = '\n'.join(list_str).replace(',', '')
        return board
