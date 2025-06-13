import numpy as np
from lib.game.game import BaseGame
from lib.game.caro import caro_19x19_helpers # Import helpers mới
from typing import List, Tuple

Matrix = List[List[int]]


class Caro19x19(BaseGame):
    """
    Một đại diện của trò chơi cờ Caro (Gomoku đơn giản) 19x19,
    trong đó chiến thắng được xác định bằng cách đặt 5 quân liên tiếp.
    Trạng thái trò chơi có thể được biểu diễn theo 2 cách:
    - Dạng ma trận nxn List[List[int]]:
      1 & 0 đại diện cho quân cờ của người chơi (1 cho đen, 0 cho trắng),
      và 2 đại diện cho ô trống.
    - Dạng số nguyên nxn chữ số: với cùng ý nghĩa cho 0, 1 và 2. Vị trí
      của mỗi chữ số tương ứng với chỉ số của mỗi ô trên bảng từ
      trên xuống dưới và từ trái sang phải.
      Ví dụ, trên bảng 3x3:
      |0|1|2|
      |3|4|5|
      |6|7|8|
    """

    def __init__(self):
        """Tạo một phiên bản của trò chơi cờ Caro 19x19."""
        super().__init__()
        self.board_len = 19  # Kích thước bảng là 19x19
        self.k_to_win = 5    # Số ô liên tiếp để thắng là 5
        self.player_black = 1
        self.player_white = 0
        self.empty = 2 # Sử dụng 2 cho ô trống, như TicTacToe

    @staticmethod
    def flatten_nested_list(nested_list: List[List]) -> List:
        """San phẳng một danh sách lồng nhau hai cấp thành một danh sách phẳng."""
        return [item for sublist in nested_list for item in sublist]

    @property
    def initial_state(self) -> int:
        """Trạng thái ban đầu của trò chơi ở dạng MCTS."""
        empty_board = np.full(
            (self.board_len, self.board_len), self.empty, dtype=np.int8).tolist()
        return self.encode_game_state(empty_board)

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Hình dạng của trạng thái trò chơi ở dạng mạng nơ-ron."""
        return (2, self.board_len, self.board_len)

    @property
    def action_space(self) -> int:
        """Tổng số hành động có thể thực hiện."""
        return self.board_len ** 2

    def _pad_mcts_state(self, mcts_state_str: str) -> str:
        """Đệm trạng thái trò chơi ở dạng chuỗi số nguyên với các số 0 ở đầu
        để đảm bảo đúng độ dài."""
        return mcts_state_str.rjust(self.board_len ** 2, "0")

    def encode_game_state(self, state_list: Matrix) -> int:
        """Chuyển đổi trạng thái trò chơi từ dạng ma trận sang dạng số nguyên
        (thân thiện với MCTS)."""
        flattened = self.flatten_nested_list(state_list)
        stringified = [str(i) for i in flattened]
        return int(''.join(stringified))

    def convert_mcts_state_to_list_state(self, mcts_state: int) -> Matrix:
        """Chuyển đổi trạng thái trò chơi từ dạng số nguyên (MCTS-friendly)
        sang dạng ma trận (dễ đọc hơn)."""
        # Đệm đến số ô trên bảng (trong trường hợp có các số 0 ở đầu)
        padded = self._pad_mcts_state(str(mcts_state))
        state = []
        for i, c in enumerate(padded):
            if i % self.board_len == 0:
                # Hàng mới chỉ xuất hiện sau mỗi board_len mục
                state.append([int(c)])
            else:
                state[i // self.board_len].append(int(c))
        return state

    def possible_moves(self, mcts_state: int) -> List:
        """Trả về các chỉ số của các ô trống, từ trái sang phải, từ trên xuống dưới."""
        padded = self._pad_mcts_state(str(mcts_state))
        return [i for i, c in enumerate(padded) if c == str(self.empty)]

    def invalid_moves(self, mcts_state: int) -> List:
        """Trả về các ô không trống."""
        padded = self._pad_mcts_state(str(mcts_state))
        return [i for i, c in enumerate(padded) if c != str(self.empty)]

    def _encode_list_state(self, dest_np: np.ndarray, state: Matrix, who_move: int) -> None:
        """
        Mã hóa trạng thái danh sách tại chỗ vào mảng numpy không.
        :param dest_np: mảng đích, dự kiến là không.
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
        """Chuyển đổi các trạng thái trò chơi thành các mảng có thể đưa vào mạng nơ-ron."""
        batch_size = len(state_ints)
        batch = np.zeros((batch_size,) + self.obs_shape, dtype=np.float32)
        for idx, (state, who_move) in enumerate(zip(state_ints, who_moves_lists)):
            converted_state = self.convert_mcts_state_to_list_state(state)
            self._encode_list_state(batch[idx], converted_state, who_move)
        return batch

    def move(self, mcts_state: int, move: int, player: int) -> Tuple[int, bool]:
        """Tại một trạng thái trò chơi nhất định, thực hiện một nước đi (hợp lệ)
        bởi một người chơi được chỉ định."""
        assert player == self.player_white or player == self.player_black
        assert 0 <= move < self.action_space # Sửa điều kiện move <= self.action_space thành <

        board = self.convert_mcts_state_to_list_state(mcts_state)
        row_idx, col_idx = divmod(move, self.board_len)
        
        # Kiểm tra xem ô đã bị chiếm đóng chưa
        if board[row_idx][col_idx] != self.empty:
            raise ValueError(f"Ô {move} đã bị chiếm đóng.")

        board[row_idx][col_idx] = player
        won = caro_19x19_helpers.check_win( # Gọi helper mới
            board, (row_idx, col_idx), self.k_to_win, player)
        new_mcts_state = self.encode_game_state(board)
        return new_mcts_state, won

    def render(self, mcts_state: int) -> str:
        """Biểu diễn trạng thái bảng dưới dạng chuỗi, để tương tác với người chơi."""
        list_state = self.convert_mcts_state_to_list_state(mcts_state)
        
        # Thêm chỉ số cột và hàng để dễ nhìn hơn
        header = "   " + " ".join([f"{i:2d}" for i in range(self.board_len)]) # Chỉ số cột 0-18
        separator = "  +" + "---" * self.board_len + "-"
        
        board_lines = []
        for i, row in enumerate(list_state):
            row_str = []
            for cell in row:
                if cell == self.empty:
                    row_str.append(" .") # Dấu chấm cho ô trống
                elif cell == self.player_white:
                    row_str.append(" O") # O cho người chơi trắng
                elif cell == self.player_black:
                    row_str.append(" X") # X cho người chơi đen
            board_lines.append(f"{i:2d}|{'|'.join(row_str)}|") # Chỉ số hàng 0-18
        
        return f"{header}\n{separator}\n" + "\n".join(board_lines) + f"\n{separator}\n{header}"