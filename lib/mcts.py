"""
Triển khai MCTS (Monte Carlo Tree Search) cho môi trường trò chơi.
"""
import math as m
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional

import config as cfg
from lib.model import Net
from lib.game.game import BaseGame

StateInt = int
VisitCount = Dict[StateInt, List[int]]
Value = Dict[StateInt, List[float]]
ValueAverage = Dict[StateInt, List[float]]
Probs = Dict[StateInt, List[float]]

# Triển khai MCTS
class MCTS:
    """
    Lớp lưu giữ số liệu thống kê cho mọi trạng thái gặp phải trong quá trình tìm kiếm
    """

    def __init__(self, game: BaseGame):
        self.c_puct = cfg.C_PUCT

        # Số lần truy cập vào mỗi trạng thái, state_int -> [N(s, a)]
        self.visit_count: VisitCount = {}

        # Tổng giá trị của hành động của trạng thái, state_int -> [W(s, a)]
        self.value: Value = {}

        # Giá trị trung bình của các hành động, state_int -> [Q(s, a)]
        self.value_avg: ValueAverage = {}

        # xác suất trước của các hành động, state_int -> [P(s,a)]
        self.probs: Probs = {}
        self.game = game

    # Hàm xoá dữ liệu thống kê cho mọi trạng thái
    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.value_avg.clear()
        self.probs.clear()

    def __len__(self):
        return len(self.value)

    def _add_noise(self, probs: List[float]) -> List[float]:
        """
        Thêm nhiễu vào xác suất hành động để khuyến khích khám phá

        Đối số:
            probs (list): Danh sách xác suất hành động
        """
        alpha = cfg.ALPHA
        explore = cfg.EXPLORE
        noises = np.random.dirichlet(
            [alpha] * self.game.action_space)
        probs_with_noise = [
            (1 - explore) * prob + explore * noise
            for prob, noise in zip(probs, noises)
        ]
        return probs_with_noise

    def _calculate_upper_bound(self, values_avg: List[float], probs: List[float],
                               counts: List[int]) -> List[float]:
        """
        Tính điểm cho mỗi hành động tại trạng thái trò chơi hiện tại
        từ các giá trị trung bình, xác suất & số lần đếm.

        Đối số:
            values_avg (List[float]): Q(s, a) trong bài báo — Giá trị hành động trung bình.
            Đây là kết quả trò chơi trung bình trên các mô phỏng hiện tại đã thực hiện hành động a.
            probs (List[float]): P(s,a) — Xác suất trước đó được lấy từ mạng.
            counts (List[int]): N(s,a) — Số lần truy cập hoặc số lần chúng ta đã thực hiện
            hành động này với trạng thái này trong các mô phỏng hiện tại
        Trả về:
            (List[float])
        """
        total_sqrt = m.sqrt(sum(counts))
        return [
            value + self.c_puct * prob * total_sqrt/(1+count)
            for value, prob, count in
            zip(values_avg, probs, counts)
        ]
    
    def _mask_invalid_actions(self, scores: List[float], cur_state: int) -> None:
        """
        Thay thế các hành động không hợp lệ bằng mặt nạ để ngăn chúng được chọn

        Đối số:
            điểm (danh sách float): Danh sách điểm không được che cho mỗi hành động
            cur_state (int): Trạng thái của trò chơi
        """
        invalid_actions = self.game.invalid_moves(cur_state)
        for invalid in invalid_actions:
            scores[invalid] = -np.inf

    def find_leaf(self, state_int: StateInt,
                  player: int) -> Tuple[Optional[float], StateInt, int, List[StateInt], List[int]]:
        """
        Duyệt cây trò chơi từ trạng thái trò chơi cho đến khi kết thúc trò chơi hoặc nút lá
        (trạng thái mà chúng ta chưa từng thấy trước đây), theo dõi tất cả các trạng thái đã truy cập
        và hành động đã thực hiện
        :return: bộ (giá trị, trạng thái lá, người chơi, trạng thái, hành động)

        Đối số:
            state_int (int): trạng thái nút gốc
            người chơi (int): người chơi di chuyển tại nút gốc

        Trả về:
            value (float): Không có nếu nút lá, nếu không (kết thúc trò chơi) bằng kết quả trò chơi cho người chơi tại lá
            leaf_state (int): trạng thái trò chơi cuối cùng tại nút lá
            player (int): người chơi tại nút lá
            states (List[int]): danh sách các trạng thái đã duyệt
            actions List[int]]: danh sách các hành động đã thực hiện
        """
        states = []
        actions = []
        cur_state = state_int
        cur_player = player
        value = None

        while not self.is_leaf(cur_state):
            states.append(cur_state)
            
            counts = self.visit_count[cur_state]
            probs = self.probs[cur_state]
            values_avg = self.value_avg[cur_state]

            # Trong nút gốc(lần di chuyển đầu tiên), thêm nhiễu vào xác suất
            if cur_state == state_int:
                probs = self._add_noise(probs)

            scores = self._calculate_upper_bound(values_avg, probs, counts)
            self._mask_invalid_actions(scores, cur_state)

            # chọn và ghi lại hành động với điểm cao nhất
            action = int(np.argmax(scores))
            actions.append(action)
            cur_state, won = self.game.move(
                cur_state, action, cur_player)
            if won:
                # Nếu ai đó thắng trò chơi, giá trị của trạng thái cuối cùng là -1 (giống như trong lượt của đối thủ)
                value = -1.0
            cur_player = 1-cur_player
            # Kiểm tra xem có phải là hòa không
            if value is None and len(self.game.possible_moves(cur_state)) == 0:
                value = 0.0
        
        return value, cur_state, cur_player, states, actions

    def is_leaf(self, state_int: StateInt) -> bool:
        """
        Xác định xem trạng thái trò chơi gặp phải có phải là một lá trong cây tìm kiếm hay không,
        tức là nếu nó chưa được gặp.

        Đối số:
            state_int (int): [description]

        Trả về:
            [type]: [description]
        """
        return state_int not in self.probs


    def search_batch(self, count: int, batch_size: int, state_int: StateInt,
                     player: int, net: Net, device: str = "cpu"):
        """
        Thực hiện một số tìm kiếm MCTS từ trạng thái trò chơi đã cho

        Đối số:
            count (int): [description]
            batch_size (int): [description]
            state_int (int): [description]
            player (int): [description]
            net (Net): [description]
            device (str, tùy chọn): [description]. Mặc định là "cpu".
        """
        for _ in range(count):
            self.search_minibatch(batch_size, state_int,
                                  player, net, device)
    
    def _create_node(self, leaf_state: int, prob: List[float]):
        """
        Tạo một nút mới trong cây trạng thái trò chơi

        Đối số:
            leaf_state (int): Trạng thái trò chơi của nút lá mới
            prob (List[float]): Xác suất trước của mỗi hành động ở trạng thái đó
            được truy vấn từ mạng nơ-ron
        """
        action_space = self.game.action_space
        self.visit_count[leaf_state] = [0]*action_space
        self.value[leaf_state] = [0.0]*action_space
        self.value_avg[leaf_state] = [0.0]*action_space
        self.probs[leaf_state] = prob

    def _expand_tree(self, expand_states: List[StateInt], expand_players: List[int],
                     expand_queue: List[Tuple[int, List[int], List[int]]],
                     backup_queue: List,
                     net: Net, device: str = "cpu") -> None:
        """
        Với hàng đợi các trạng thái trò chơi chưa gặp, hãy truy vấn hàng loạt mạng
        để có được xác suất dự đoán cho từng hành động và giá trị dự đoán

        Đối số:
            expand_states (List[int]): Danh sách các trạng thái trò chơi của nút lá để mở rộng từ
            expand_players (List[int]): Danh sách "lượt của ai" tương ứng
            với từng trạng thái nút lá
            expand_queue (List[(int, List(int), List(int))]): Danh sách các trạng thái lá
            để thực hiện quy trình sao lưu. Ngoài ra, hãy liệt kê các trạng thái đã truy cập và
            danh sách các hành động đã thực hiện để có được trạng thái lá, cả hai đều cần thiết cho
            quy trình sao lưu
            backup_queue (List): Hàng đợi giá trị, trạng thái và hành động để thực hiện
            quy trình sao lưu sau.
            net (Net): Mạng nơ-ron để truy vấn xác suất hành động và giá trị trạng thái trò chơi từ
            device (str): cpu hoặc gpu cho PyTorch
        """
        batch_v = self.game.states_to_training_batch(
            expand_states, expand_players)
        batch_tensor = torch.tensor(batch_v).to(device)
        logits_v, values_v = net(batch_tensor)
        probs_v = F.softmax(logits_v, dim=1)
        values = values_v.data.cpu().numpy()[:, 0]
        probs = probs_v.data.cpu().numpy()

        # tạo các nút
        for (leaf_state, states, actions), value, prob in zip(expand_queue, values, probs):
            self._create_node(leaf_state, prob)
            backup_queue.append((value, states, actions))

    def _backup(self, value: float, states: List[StateInt], actions: List[int]):
        """
        Cập nhật các trạng thái hiện có bằng các giá trị mới, giá trị trung bình, số lần truy cập
        khi quy trình MCTS mở rộng cây trạng thái.

        Đối số:
            value (float): Giá trị của trạng thái trò chơi được dự đoán bởi mạng nơ-ron
            hoặc giá trị cuối cùng của trò chơi (thắng/thua/hòa)
            state (Danh sách[int]): Danh sách các trạng thái trò chơi cần sao lưu
            actions (Danh sách[int]): Danh sách các hành động được thực hiện dẫn đến
            trạng thái trò chơi tương ứng
        """
        # trạng thái lá không được lưu trữ trong các trạng thái và hành động,
        # vì vậy giá trị của lá sẽ là giá trị của đối thủ
        cur_value = -value
        for state_int, action in zip(states[::-1],
                                     actions[::-1]):
            self.visit_count[state_int][action] += 1
            self.value[state_int][action] += cur_value
            # update the average value with new value
            self.value_avg[state_int][action] = (self.value[state_int][action] /
                                                 self.visit_count[state_int][action])
            cur_value = -cur_value # đảo ngược giá trị cho người chơi tiếp theo

    def search_minibatch(self, batch_size: int, state_int: StateInt, player: int,
                         net: Net, device: str = "cpu") -> None:
        """
        Thực hiện một số tìm kiếm MCTS. Mạng nơ-ron PyTorch được truy vấn theo từng đợt,
        do đó, thực hiện MCTS theo từng đợt cũng thuận tiện hơn.

        Đối số:
            batch_size (int): Số lượng tìm kiếm trong đợt này
            state_int (int): Trạng thái trò chơi ở dạng MCTS
            player (int): Người chơi đến lượt thực hiện nước đi
            net (Net): Mạng nơ-ron để lấy giá trị chính sách cho các trạng thái chưa gặp
            device (str, tùy chọn): Thiết bị PyTorch. Mặc định là "cpu".
        """
        backup_queue = []
        expand_states = []
        expand_players = []
        expand_queue = []
        planned = set()
        for _ in range(batch_size):
            value, leaf_state, leaf_player, states, actions = \
                self.find_leaf(state_int, player)
            if value is not None:
                # reached terminal game state, can backup with actual reward
                backup_queue.append((value, states, actions))
            else:
                if leaf_state not in planned:
                    planned.add(leaf_state)
                    expand_states.append(leaf_state)
                    expand_players.append(leaf_player)
                    expand_queue.append((leaf_state, states,
                                         actions))

        # mở rộng các nút
        if expand_queue:
            self._expand_tree(expand_states, expand_players,
                              expand_queue, backup_queue, net, device)
        
        # thực hiện sao lưu các tìm kiếm
        for value, states, actions in backup_queue:
            self._backup(value, states, actions)

    def get_policy_value(self, state_int: StateInt, tau: int = 1) -> Tuple[List[float], List[float]]:
        """
        Trích xuất chính sách và giá trị hành động theo trạng thái
        :return: (probs, values)

        Đối số:
            state_int (int): trạng thái của bảng
            tau (int, tùy chọn): Một siêu tham số khiến mô hình
            khám phá nhiều hơn. Mặc định là 1.

        Trả về:
            List[float]: Danh sách các xác suất cho mỗi hành động trong tổng số
            không gian hành động
            List[float]: Danh sách các giá trị cho mỗi hành động trong tổng số
            không gian hành động
        """
        counts = self.visit_count[state_int]
        if tau == 0:
            probs = [0.0] * self.game.action_space
            probs[np.argmax(counts)] = 1.0
        else:
            counts_adjusted = [count ** (1.0 / tau) for count in counts]
            total = sum(counts_adjusted)
            probs = [count / total for count in counts_adjusted]
        values = self.value_avg[state_int]
        return probs, values