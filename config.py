# train.py
PLAY_EPISODES = 5              # Giảm mạnh để tạo dữ liệu nhanh
MCTS_SEARCHES = 50             # AI "suy nghĩ" ít hơn cho mỗi nước đi
MCTS_BATCH_SIZE = 64
REPLAY_BUFFER = 10000            # Giảm kích thước bộ nhớ đệm
LEARNING_RATE = 0.1
BATCH_SIZE = 128
TRAIN_ROUNDS = 50               # Train ít vòng hơn trên mỗi bộ dữ liệu
MIN_REPLAY_TO_TRAIN = 500      # Bắt đầu train sớm hơn rất nhiều

BEST_NET_WIN_RATIO = 0.55      # Hạ thấp ngưỡng để model mới dễ được chấp nhận

EVALUATE_EVERY_STEP = 50       # Đánh giá model mới thường xuyên hơn
EVALUATION_ROUNDS = 10         # Đánh giá với ít ván hơn
STEPS_BEFORE_TAU_0 = 12         # Chơi ngẫu nhiên trong ít bước hơn
# play.py
PLAY_MCTS_SEARCHES = 200
PLAY_MCTS_BATCH_SIZE = 64

# telegram-bot.py
BOT_MCTS_SEARCHES = 200
BOT_MCTS_BATCH_SIZE = 64

# lib/mcts.py
C_PUCT = 1.5
ALPHA = 0.03
EXPLORE = 0.25

# lib/model.py
NUM_FILTERS = 64

PUZZLE_MAX_STEPS = 100
DISCOUNT_FACTOR = 0.99
PUZZLE_SOLVE_RATE_TARGET = 0.6

REWARD_SOLVED = 1.0
REWARD_STEP = -0.01 
REWARD_INVALID_MOVE = -0.1