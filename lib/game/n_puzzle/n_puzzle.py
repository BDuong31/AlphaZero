# lib/game/n_puzzle/n_puzzle.py

import numpy as np
from typing import Tuple, List, Dict
from lib.game.game import BaseGame
from lib.game.n_puzzle.n_puzzle_helper import NPuzzleHelper, DIRECTIONS

# Hằng số cho các hành động
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

class NPuzzle(BaseGame):
    """Lớp bao bọc trò chơi N-Puzzle để tuân thủ giao diện BaseGame."""

    def __init__(self, size=3):
        self.size = size
        self.helper = NPuzzleHelper(size=self.size)
        # Bộ đệm để lưu trữ các trạng thái board, vì MCTS sử dụng hash làm khóa
        self.state_cache: Dict[int, np.ndarray] = {}
        # Thêm trạng thái ban đầu vào bộ đệm
        initial_board = self.helper.board.copy()
        self._initial_state = hash(initial_board.tobytes())
        self.state_cache[self._initial_state] = initial_board

    @property
    def initial_state(self) -> int:
        return self._initial_state

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        # 1 kênh, kích thước x kích thước bàn cờ
        return 1, self.size, self.size

    @property
    def action_space(self) -> int:
        # 4 hướng: Lên, Xuống, Trái, Phải
        return 4

    def _get_board_from_state(self, mcts_state: int) -> np.ndarray:
        """Lấy board NumPy từ hash của trạng thái MCTS."""
        board = self.state_cache.get(mcts_state)
        if board is None:
            raise ValueError("Trạng thái không được tìm thấy trong bộ đệm. Điều này không nên xảy ra.")
        return board

    def possible_moves(self, mcts_state: int) -> List[int]:
        board = self._get_board_from_state(mcts_state)
        temp_helper = NPuzzleHelper(self.size)
        temp_helper.board = board
        temp_helper.empty_pos = temp_helper._find_empty_pos()
        return list(temp_helper.get_possible_moves().keys())

    def invalid_moves(self, mcts_state: int) -> List[int]:
        possible = self.possible_moves(mcts_state)
        return [move for move in range(self.action_space) if move not in possible]

    def move(self, mcts_state: int, move: int, player: int) -> Tuple[int, bool]:
        """
        Thực hiện một nước đi. `player` bị bỏ qua vì đây là trò chơi một người chơi.
        """
        if move not in self.possible_moves(mcts_state):
             raise ValueError(f"Nước đi không hợp lệ {move} cho trạng thái.")

        board = self._get_board_from_state(mcts_state).copy()
        temp_helper = NPuzzleHelper(self.size)
        temp_helper.board = board
        temp_helper.empty_pos = temp_helper._find_empty_pos()

        temp_helper.move(move)
        
        new_board = temp_helper.board
        won = temp_helper.is_solved()
        
        new_mcts_state = hash(new_board.tobytes())
        if new_mcts_state not in self.state_cache:
            self.state_cache[new_mcts_state] = new_board
            
        return new_mcts_state, won

    def states_to_training_batch(self, mcts_states: List[int], who_moves_lists: List[int]) -> np.ndarray:
        """
        Chuyển đổi danh sách các trạng thái MCTS thành một lô để huấn luyện.
        `who_moves_lists` bị bỏ qua.
        """
        batch_size = len(mcts_states)
        batch = np.zeros((batch_size, *self.obs_shape), dtype=np.float32)
        for i, state_hash in enumerate(mcts_states):
            board = self._get_board_from_state(state_hash)
            # Bình thường hóa các giá trị ô để cải thiện việc huấn luyện
            normalized_board = board / (self.size * self.size - 1)
            batch[i, 0, :, :] = normalized_board
        return batch

    def render(self, mcts_state: int) -> str:
        board = self._get_board_from_state(mcts_state)
        temp_helper = NPuzzleHelper(self.size)
        temp_helper.board = board
        return temp_helper.render()