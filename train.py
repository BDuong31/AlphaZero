#!/usr/bin/env python3
import os
import sys
import time
import random
import argparse
import collections
from typing import Union

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import config as cfg
from lib.model import Net, NetWrapper
from lib.mcts import MCTS
from lib.utils import play_game, TBMeanTracker
from lib.game.game import BaseGame
from lib.game import game_provider

Optimizer = torch.optim.Optimizer


def self_play(game: BaseGame, mcts_store: MCTS, replay_buffer: Union[collections.deque, None],
              model: Net, tb_tracker, device: str) -> None:
    """
    Để mô hình (tốt nhất hiện tại) chơi với chính nó để tạo dữ liệu đào tạo.
    Lưu trữ các nước đi vào bộ đệm phát lại.

    Đối số:
        game (Game): Trò chơi mà mạng đang được đào tạo để chơi
        mcts_store (MCTS): Cây Monte Carlo về trạng thái và kết quả của các trò chơi đã chơi đầy đủ
        replay_buffer (deque): Hàng đợi các nước đi và giá trị dự đoán được thực hiện bởi
        current best net
        model (Net): Mạng nơ-ron được đào tạo để chơi trò chơi
        tb_tracker (Bộ theo dõi bảng Tensorflow) để thu thập số liệu thống kê
        device (str): cpu hoặc gpu cho PyTorch
    """
    t = time.time()
    prev_nodes = len(mcts_store)
    game_steps = 0
    for _ in range(cfg.PLAY_EPISODES):
        _, steps = play_game(game, mcts_store, replay_buffer,
                             best_net.target_model, best_net.target_model,
                             steps_before_tau_0=cfg.STEPS_BEFORE_TAU_0,
                             mcts_searches=cfg.MCTS_SEARCHES,
                             mcts_batch_size=cfg.MCTS_BATCH_SIZE, device=device)
        game_steps += steps
    game_nodes = len(mcts_store) - prev_nodes
    dt = time.time() - t
    speed_steps = game_steps / dt
    speed_nodes = game_nodes / dt
    tb_tracker.track("speed step", speed_steps, step_idx)
    tb_tracker.track("speed node", speed_nodes, step_idx)
    sys.stdout.flush()
    buffer_len = len(replay_buffer) if replay_buffer else 0
    print("Step %d, game steps %3d, node %4d, step/s %5.2f, node/s %6.2f, best idx %d, replay %d" % (
        step_idx, game_steps, game_nodes, speed_steps, speed_nodes, best_idx, buffer_len),
        end='\r')


def train_neural_net(game: BaseGame, replay_buffer: collections.deque, optimizer: Optimizer,
                     tb_tracker, device: str) -> None:
    """
    Cung cấp một bộ đệm phát lại đủ lớn, huấn luyện mạng nơ-ron
    sử dụng dữ liệu từ bộ đệm phát lại theo từng đợt.

    Đối số:
        game (Game): Loại trò chơi mà mô hình đang được huấn luyện
        replay_buffer (deque): Hàng đợi các bước được thu thập trong quá trình tự chơi. Đây
        là dữ liệu được sử dụng để huấn luyện mạng nơ-ron
        optimizer (PyTorch optimizer): optimizer thực hiện gradient descent
        & cập nhật các giá trị tensor của mạng nơ-ron
        tb_tracker (TensorflowBoard): trình theo dõi ghi số liệu thống kê của mô hình vào Tensorflow
        Board
        device (str): cpu hoặc gpu (đối với PyTorch)
    """
    TRAIN_ROUNDS = cfg.TRAIN_ROUNDS
    sum_loss = 0.0
    sum_value_loss = 0.0
    sum_policy_loss = 0.0

    for _ in range(TRAIN_ROUNDS):
        # PyTorch được đào tạo theo từng đợt. Chúng tôi xử lý các đợt dữ liệu ở đây thành tenxơ
        # có thể đưa vào mạng nơ-ron để đào tạo
        batch = random.sample(replay_buffer, cfg.BATCH_SIZE)
        batch_states, batch_who_moves, batch_probs, batch_values = zip(
            *batch)
        states_v = game.states_to_training_batch(
            batch_states, batch_who_moves)
        states_tensor = torch.tensor(states_v).to(device)

        optimizer.zero_grad()
        probs_v = torch.FloatTensor(batch_probs).to(device)
        values_v = torch.FloatTensor(batch_values).to(device)
        out_logits_v, out_values_v = net(states_tensor)

        # tính toán MSE giữa giá trị dự đoán của mô hình so với kết quả thực tế
        loss_value_v = F.mse_loss(out_values_v.squeeze(-1), values_v)

        # tính toán entropy chéo giữa các xác suất chính sách của mô hình và xác suất
        # lấy mẫu từ MCTS
        loss_policy_v = -F.log_softmax(out_logits_v, dim=1) * probs_v
        loss_policy_v = loss_policy_v.sum(dim=1).mean()

        loss_v = loss_policy_v + loss_value_v
        # lan truyền ngược & giảm dần độ dốc
        loss_v.backward()
        optimizer.step()
        sum_loss += loss_v.item()
        sum_value_loss += loss_value_v.item()
        sum_policy_loss += loss_policy_v.item()

    tb_tracker.track("total loss", sum_loss / TRAIN_ROUNDS, step_idx)
    tb_tracker.track("value loss", sum_value_loss /
                     TRAIN_ROUNDS, step_idx)
    tb_tracker.track("policy loss", sum_policy_loss /
                     TRAIN_ROUNDS, step_idx)


def evaluate(game: BaseGame, challenger: Net, champion: Net,
             rounds: int, device: str = "cpu") -> float:
    """
    Đánh giá hiệu suất của 2 mạng nơ-ron được huấn luyện để chơi trò chơi bằng cách cho chúng
    chơi các vòng đấu với nhau.

    Đối số:
        game (Game): Trò chơi mà mạng đang được huấn luyện để chơi
        challenger, champion (Net): 2 trường hợp của mạng nơ-ron PyTorch
        được huấn luyện để chơi trò chơi để so sánh hiệu suất
        round (int): Số vòng mà các mạng sẽ chơi
        device (str, tùy chọn): [description]. Mặc định là "cpu".

    Trả về:
        [float]: Tỷ lệ chiến thắng của challenger
    """
    challenger_win, champion_win, draw = 0, 0, 0
    mcts_stores = [MCTS(game), MCTS(game)]

    for r_idx in range(rounds):
        r, _ = play_game(game=game, mcts_stores=mcts_stores, replay_buffer=None,
                         net1=challenger, net2=champion,
                         steps_before_tau_0=0, mcts_searches=20, mcts_batch_size=16,
                         device=device)
        if r < -0.5:
            champion_win += 1
        elif r > 0.5:
            challenger_win += 1
        elif r == 0:
            draw += 1
    return challenger_win / (challenger_win + champion_win + draw)


def parse_args():
    """
    Thêm và phân tích các đối số.
    Trả về:
        [args]: đối tượng args với các đối số đã phân tích trong các trường của nó
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable CUDA")
    game_provider.add_game_argument(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if args.cuda else "cpu"

    saves_path = os.path.join("saves", args.name)
    os.makedirs(saves_path, exist_ok=True)
    writer = SummaryWriter(comment="-" + args.name)

    game = game_provider.get_game(args)
    model_shape = game.obs_shape

    net = Net(input_shape=model_shape,
              actions_n=game.action_space).to(device)
    best_net = NetWrapper(net)
    print(net)

    optimizer = optim.SGD(net.parameters(), lr=cfg.LEARNING_RATE, momentum=0.9)

    replay_buffer = collections.deque(maxlen=cfg.REPLAY_BUFFER)
    mcts_store = MCTS(game)
    step_idx = 0
    best_idx = 0

    with TBMeanTracker(writer, batch_size=10) as tb_tracker:
        # Về mặt lý thuyết, vòng lặp đào tạo có thể tiếp tục mãi mãi
        # để tạo ra các tác nhân tốt hơn và tốt hơn 
        while True:
            self_play(game, mcts_store, replay_buffer,
                      best_net.target_model, tb_tracker, device)
            step_idx += 1

            if len(replay_buffer) < cfg.MIN_REPLAY_TO_TRAIN:
                continue

            # Bộ đệm phát lại có đủ dữ liệu. Đào tạo mạng
            train_neural_net(game, replay_buffer,
                             optimizer, tb_tracker, device)

            # đánh giá mạng, sau đó thay thế mạng tốt nhất nếu hiệu suất đạt yêu cầu
            if step_idx % cfg.EVALUATE_EVERY_STEP == 0:
                win_ratio = evaluate(game,
                                     net, best_net.target_model, rounds=cfg.EVALUATION_ROUNDS, device=device)
                print("Mạng đã được đánh giá, tỷ lệ thắng = %.2f" % win_ratio)
                writer.add_scalar("Win ratio", win_ratio, step_idx)
                if win_ratio > cfg.BEST_NET_WIN_RATIO:
                    print("Mạng tốt hơn cur best, đồng bộ")
                    best_net.sync()
                    best_idx += 1
                    file_name = os.path.join(
                        saves_path, "best_%03d_%05d.dat" % (best_idx, step_idx))
                    torch.save(net.state_dict(), file_name)
                    mcts_store.clear()
