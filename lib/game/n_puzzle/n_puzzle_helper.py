# lib/game/n_puzzle/n_puzzle_helper.py

import numpy as np
import random

# Định nghĩa các hướng di chuyển
DIRECTIONS = {
    0: np.array([-1, 0]),  # Lên
    1: np.array([1, 0]),   # Xuống
    2: np.array([0, -1]),  # Trái
    3: np.array([0, 1]),   # Phải
}

class NPuzzleHelper:
    """Lớp trợ giúp quản lý trạng thái và logic của trò chơi N-Puzzle."""

    def __init__(self, size=3):
        """
        Khởi tạo câu đố.
        :param size: Kích thước của bàn cờ (ví dụ: size=3 cho 8-puzzle).
        """
        self.size = size
        self.board = self._get_solved_state()
        self.solved_state = self.board.copy()
        self.empty_pos = self._find_empty_pos()
        self.reset()

    def _get_solved_state(self):
        """Tạo ra trạng thái đã giải của câu đố."""
        board = np.arange(1, self.size * self.size + 1)
        board[-1] = 0  # Số 0 đại diện cho ô trống
        return board.reshape((self.size, self.size))

    def reset(self):
        """Xáo trộn bàn cờ để tạo một câu đố mới, có thể giải được."""
        self.board = self.solved_state.copy()
        
        # Thực hiện một số lượng lớn các nước đi ngẫu nhiên để xáo trộn
        for _ in range(self.size * self.size * 10):
            possible_moves = self.get_possible_moves()
            if possible_moves:
                random_move = random.choice(list(possible_moves.keys()))
                self.move(random_move)
        
        # Đảm bảo bàn cờ sau khi xáo trộn không phải là trạng thái đã giải
        if self.is_solved():
            self.reset() # Thử lại nếu nó đã được giải

    def is_solved(self):
        """Kiểm tra xem bàn cờ hiện tại đã ở trạng thái được giải hay chưa."""
        return np.array_equal(self.board, self.solved_state)

    def _find_empty_pos(self):
        """Tìm tọa độ của ô trống (số 0)."""
        pos = np.where(self.board == 0)
        return np.array([pos[0][0], pos[1][0]])

    def get_possible_moves(self):
        """
        Trả về một từ điển các hành động hợp lệ (lên, xuống, trái, phải).
        """
        possible = {}
        for action, dr_dc in DIRECTIONS.items():
            new_pos = self.empty_pos + dr_dc
            if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
                possible[action] = new_pos
        return possible

    def move(self, action):
        """
        Thực hiện một hành động (di chuyển ô trống).
        :param action: Một trong các phím từ DIRECTIONS (0, 1, 2, 3).
        """
        possible_moves = self.get_possible_moves()
        if action not in possible_moves:
            raise ValueError(f"Hành động không hợp lệ: {action}")

        # Vị trí của ô cần tráo đổi với ô trống
        tile_to_swap_pos = possible_moves[action]
        
        # Tráo đổi giá trị
        val_at_tile = self.board[tile_to_swap_pos[0], tile_to_swap_pos[1]]
        self.board[self.empty_pos[0], self.empty_pos[1]] = val_at_tile
        self.board[tile_to_swap_pos[0], tile_to_swap_pos[1]] = 0
        
        # Cập nhật vị trí ô trống
        self.empty_pos = tile_to_swap_pos

    def render(self):
        """Tạo một chuỗi biểu diễn cho bàn cờ."""
        res = []
        for r in range(self.size):
            res.append(" | ".join(f"{x:2}" for x in self.board[r]))
        return "\n".join(res).replace(" 0", "  ")