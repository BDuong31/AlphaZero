import collections
import numpy as np
from typing import Union, Tuple, Dict
import torch
from lib import mcts, model
from lib.game.game import BaseGame


def update_counts(counts_dict: Dict, key: Union[str, Tuple[str, str]], counts: Tuple[int, int, int]) -> None:
    """
    Cập nhật counts_dict với win, lose, draw từ counts nếu key tồn tại.
    Nếu không, hãy khởi tạo mục nhập mới với 0, 0, 0
    Key có thể là một chuỗi biểu diễn tên mô hình hoặc một bộ đại diện cho
    2 mô hình đấu tay đôi

    Đối số:
        counts_dict (Dict): Từ điển theo dõi các trận thắng, thua và hòa
        key (Union[str, Tuple[str, str]])
        counts (Tuple[int, int, int]): Thắng, Thua và Hòa
    """
    v = counts_dict.get(key, (0, 0, 0))
    res = (v[0] + counts[0], v[1] + counts[1], v[2] + counts[2])
    counts_dict[key] = res


def play_game(game: BaseGame, mcts_stores, replay_buffer: Union[collections.deque, None],
              net1: model.Net, net2: model.Net,
              steps_before_tau_0: int, mcts_searches: int, mcts_batch_size: int,
              net1_plays_first: bool = None, device: str = "cpu"):
    """
    Chơi một trò chơi duy nhất, ghi nhớ các chuyển tiếp vào bộ đệm phát lại
    :param net1: player1
    :param net2: player2

    Đối số:
        game ([type]): [description]
        mcts_stores ([type]): có thể là None hoặc một MCTS hoặc hai MCTS cho từng net
        replay_buffer (deque): xếp hàng với (trạng thái, xác suất, giá trị), nếu None, không có gì được lưu trữ
        net1 (model.Net): [description]
        net2 (model.Net): [description]
        steps_before_tau_0 ([type]): [description]
        mcts_searches (int): [description]
        mcts_batch_size (int): [description]
        net1_plays_first (bool, tùy chọn): [description]. Mặc định là None.
        device (str, tùy chọn): [description]. Mặc định là "cpu".

    Trả về:
        [int]: giá trị cho trò chơi liên quan đến net_1 (+1 nếu p1 thắng, -1 nếu thua, 0 nếu hòa)
        [int]:
    """
    assert isinstance(replay_buffer, (collections.deque, type(None)))
    assert isinstance(mcts_stores, (mcts.MCTS, type(None), list))
    assert isinstance(net1, model.Net)
    assert isinstance(net2, model.Net)
    assert isinstance(steps_before_tau_0, int) and steps_before_tau_0 >= 0
    assert isinstance(mcts_searches, int) and mcts_searches > 0
    assert isinstance(mcts_batch_size, int) and mcts_batch_size > 0

    if mcts_stores is None:
        mcts_stores = [mcts.MCTS(game), mcts.MCTS(game)]
    elif isinstance(mcts_stores, mcts.MCTS):
        mcts_stores = [mcts_stores, mcts_stores]

    state = game.initial_state
    nets = [net1, net2]
    if net1_plays_first is None:
        cur_player = np.random.choice(2)
    else:
        cur_player = 0 if net1_plays_first else 1
    step = 0
    tau = 1 if steps_before_tau_0 > 0 else 0
    game_history = []

    result = None
    net1_result = None

    while result is None:
        mcts_stores[cur_player].search_batch(
            mcts_searches, mcts_batch_size, state,
            cur_player, nets[cur_player], device=device)
        probs, _ = mcts_stores[cur_player].get_policy_value(
            state, tau=tau)
        game_history.append((state, cur_player, probs))
        action = np.random.choice(game.action_space, p=probs)
        if action not in game.possible_moves(state):
            print("Đã chọn hành động không thể thực hiện được")
        state, won = game.move(state, action, cur_player)
        if won:
            result = 1
            net1_result = 1 if cur_player == 0 else -1
            break
        cur_player = 1-cur_player
        # check the draw case
        if len(game.possible_moves(state)) == 0:
            result = 0
            net1_result = 0
            break
        step += 1
        if step >= steps_before_tau_0:
            tau = 0

    if replay_buffer is not None:
        for state, cur_player, probs in reversed(game_history):
            replay_buffer.append(
                (state, cur_player, probs, result)
            )
            result = -result

    return net1_result, step


class TBMeanTracker:
    """
    Trình theo dõi giá trị TensorBoard: cho phép nhóm một lượng cố định các giá trị lịch sử và ghi giá trị trung bình của chúng vào TB
    Được thiết kế và thử nghiệm với pytorch-tensorboard
    """

    def __init__(self, writer, batch_size):
        """
        :param writer: writer với các phương thức close() và add_scalar()
        :param batch_size: kích thước số nguyên của lô để theo dõi
        """
        assert isinstance(batch_size, int)
        assert writer is not None
        self.writer = writer
        self.batch_size = batch_size

    def __enter__(self):
        self._batches = collections.defaultdict(list)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    @staticmethod
    def _as_float(value):
        assert isinstance(value, (float, int, np.ndarray, np.generic,
                          torch.autograd.Variable)) or torch.is_tensor(value)
        tensor_val = None
        if isinstance(value, torch.autograd.Variable):
            tensor_val = value.data
        elif torch.is_tensor(value):
            tensor_val = value

        if tensor_val is not None:
            return tensor_val.float().mean().item()
        elif isinstance(value, np.ndarray):
            return float(np.mean(value))
        else:
            return float(value)

    def track(self, param_name, value, iter_index):
        assert isinstance(param_name, str)
        assert isinstance(iter_index, int)

        data = self._batches[param_name]
        data.append(self._as_float(value))

        if len(data) >= self.batch_size:
            self.writer.add_scalar(param_name, np.mean(data), iter_index)
            data.clear()
