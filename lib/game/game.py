"""
    Định nghĩa giao diện cho lớp Game
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Union
import numpy as np

class BaseGame(ABC):
    """
    Một trò chơi cờ bàn dành cho hai người chơi, trong đó người chơi thay phiên nhau thay đổi trạng thái trò chơi.
    Trạng thái trò chơi phải được biểu diễn theo kiểu bất biến đối với quy trình MCTS
    (Monte Carlo Tree Search).
    Trạng thái trò chơi cũng cần được chuyển đổi thành dạng có thể đưa vào
    Mạng nơ-ron.
    """

    @property
    @abstractmethod
    def initial_state(self) -> int:
        """
        Trạng thái ban đầu của trò chơi ở dạng MCTS. Trạng thái này được sử dụng trong
        utils.play_game để bắt đầu vòng lặp MCTS.
        """
        pass

    @property
    @abstractmethod
    def obs_shape(self) -> Tuple[int, ...]:
        """
        Hình dạng của mạng nơ-ron dạng trạng thái trò chơi.
        Đây phải là một bộ các số nguyên. Ví dụ: Đối với trò chơi Tic-Tac-Toe, nó có thể là
        (2, 3, 3) tức là trạng thái trò chơi được đưa vào mạng nơ-ron
        là một tenxơ 2x3x3: 2 người chơi x 3x3 bàn cờ (bàn cờ nhìn từ phía của mỗi người chơi)        
        """
        pass

    @property
    @abstractmethod
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
        pass

    @abstractmethod
    def possible_moves(self, mcts_state: int) -> List:
        """
        Trả về danh sách tất cả các nước đi có thể có dựa trên trạng thái trò chơi

        Đối số:
            mcts_state (int): Trạng thái trò chơi ở dạng MCTS

        Trả về:
            Danh sách: Danh sách các hành động có thể có
        """
        pass

    @abstractmethod
    def invalid_moves(self, mcts_state: int) -> List:
        """
        Trả về danh sách tất cả các nước đi không hợp lệ cho trạng thái trò chơi.
        Được sử dụng để phạt các hành động này.

        Đối số:
            mcts_state (int): Trạng thái trò chơi trong biểu mẫu MCTS

        Trả về:
            Danh sách: Danh sách các hành động không hợp lệ
        """
        pass

    @abstractmethod
    def states_to_training_batch(self, mcts_lists: List, who_moves_lists: List[int]) -> np.ndarray:
        """
        Chuyển đổi trạng thái trò chơi thành dạng có thể được sử dụng làm dữ liệu đào tạo
        cho mạng nơ-ron.

        Đối số:
            state_lists (Danh sách): Danh sách các trạng thái trò chơi ở dạng MCTS
            who_moves_lists (Danh sách[int]): Danh sách tương ứng của người chơi đã thực hiện
            nước đi ở trạng thái trò chơi đó.

        Trả về:
            np.ndarray: Đối với trò chơi 2 người chơi với bàn cờ m x n, mỗi trạng thái trò chơi
            sẽ được biểu diễn dưới dạng mảng 2 x m x n. Mỗi mảng sẽ phản ánh
            các quân cờ/nước đi của một người chơi có giá trị 1 và có giá trị bằng không ở
            tất cả các vị trí khác (quân cờ của người chơi đối diện và các vị trí trống).
            Đây là cách dữ liệu được đưa vào mạng nơ-ron trong bài báo AlphaZero.
        """
        pass

    @abstractmethod
    def move(self, mcts_state: int, move: int, player: int) -> Tuple[int, bool]:
        """
        Với trạng thái trò chơi hiện tại và một nước đi (hợp lệ) của một người chơi cụ thể,
        hãy lấy trạng thái trò chơi mới và liệu nước đi đó có dẫn đến chiến thắng hay không.

        Đối số:
            mcts_state (int): Trạng thái trò chơi ở dạng MCTS
            move (int): Hành động đã thực hiện
            player (int): Người chơi nào trong số 2 người chơi (0 hoặc 1) đã thực hiện hành động

        Trả về:
            Tuple[int, bool]: Trạng thái trò chơi mới, liệu trò chơi đã được thắng hay chưa
            (bởi người chơi thực hiện nước đi)
        """
        pass

    @abstractmethod
    def render(self, mcts_state: int) -> str:
        """
        Hiển thị chuỗi biểu diễn trạng thái trò chơi hiện tại

        Đối số:
            mcts_state (int): Trạng thái trò chơi ở dạng MCTS

        Trả về:
            Union[str, List[str]]: Chuỗi biểu diễn trạng thái trò chơi
        """
        pass