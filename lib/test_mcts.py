import pytest
from unittest.mock import MagicMock, patch
from lib.mcts import MCTS

@pytest.fixture
def tree():
    """
    Hãy mô phỏng một trò chơi với 2 hành động có thể xảy ra: 0 & 1
    Hãy xây dựng một cây bắt đầu từ trạng thái 1,
    thực hiện hành động 1 một lần và dẫn đến trạng thái 2,
    sau đó từ trạng thái 2 thực hiện hành động 0 dẫn đến trạng thái 3
    """
    mock_game = MagicMock()
    tree = MCTS(mock_game)
    tree.visit_count = {1: [0, 1], 2: [1, 0], 3: [0, 0]}
    tree.value = {1: [0.0, 0.5], 2: [0.6, 0.0], 3: [0.0, 0.0]}
    tree.value_avg = {1: [0.0, 0.5], 2: [0.6, 0.0], 3: [0.0, 0.0]}
    # Nhớ rằng tổng xác suất trước của các hành động tại mỗi trạng thái là 1
    # Trạng thái 3 chưa được truy cập nên mọi thứ đều bằng 0 ngoại trừ các xác suất trước
    # được truy vấn từ mạng nơ-ron
    tree.probs = {1: [0.1, 0.9], 2: [0.8, 0.2], 3: [0.7, 0.3]}
    return tree

class TestBackup:
    def test_back_up(self, tree):
        value = 0.2
        states = [1, 2, 3]
        # Giả sử chúng ta thực hiện lại các hành động tương tự (1 --1--> 2 --0--> 3)
        # sau đó là 0
        actions = [1, 0, 0]
        tree._backup(value, states, actions)
        assert tree.visit_count == {1: [0, 2], 2: [2, 0], 3: [1, 0]}
        
        # Nhớ lật dấu giá trị ở mỗi lượt
        assert tree.value == {1: [0.0, 0.3], 2: [0.8, 0.0], 3: [-0.2, 0.0]}
        
        # Giá trị trung bình trên visit_count
        assert tree.value_avg == {
            1: [0.0, 0.15], 2: [0.4, 0.0], 3: [-0.2, 0.0]}
