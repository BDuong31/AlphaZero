#!/usr/bin/env python3
"""
Giao diện đồ họa (GUI) để người chơi có thể đấu với AI.

Cách chạy:
1. Đảm bảo bạn đã cài đặt Pygame: pip install pygame
2. Chạy từ thư mục gốc của dự án.
3. Lệnh cho TicTacToe:
   python gui_play.py -g 1 -m "path/to/your/tictactoe_model.h5"
4. Lệnh cho Caro:
   python gui_play.py -g 0 -m "path/to/your/caro_model.h5"
"""
import pygame
import argparse
import sys
import time

from lib.play_session import Session
from lib.game import game_provider
from lib.game.tictactoe.tictactoe import TicTacToe
from lib.game.caro.caro_19x19 import Caro19x19

# --- Cấu hình Pygame và Giao diện ---
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 500
BOARD_SIZE = 400

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_LINE = (200, 200, 200)
COLOR_PLAYER_X = (84, 84, 84)
COLOR_PLAYER_O = (242, 235, 211)

# --- Các hàm tiện ích và vẽ ---

def get_board_dimensions(game):
    if hasattr(game, 'board_len'):
        return game.board_len, game.board_len
    elif hasattr(game, 'rows') and hasattr(game, 'cols'):
        return game.rows, game.cols
    else:
        raise ValueError("Không thể xác định kích thước bàn cờ cho game này.")

def draw_grid(screen, rows, cols):
    cell_width = BOARD_SIZE // cols
    cell_height = BOARD_SIZE // rows
    for x in range(0, BOARD_SIZE + 1, cell_width):
        pygame.draw.line(screen, COLOR_LINE, (x, 0), (x, BOARD_SIZE), 1)
    for y in range(0, BOARD_SIZE + 1, cell_height):
        pygame.draw.line(screen, COLOR_LINE, (0, y), (BOARD_SIZE, y), 1)

def draw_pieces(screen, board_state_list, game, rows, cols):
    cell_width = BOARD_SIZE // cols
    cell_height = BOARD_SIZE // rows
    for r in range(rows):
        for c in range(cols):
            cell_content = board_state_list[r][c]
            center_x = c * cell_width + cell_width // 2
            center_y = r * cell_height + cell_height // 2

            if cell_content == game.player_white:
                pygame.draw.circle(screen, COLOR_PLAYER_O, (center_x, center_y), cell_width // 2 - 3, 2)
            elif cell_content == game.player_black:
                offset = cell_width // 3
                pygame.draw.line(screen, COLOR_PLAYER_X, (center_x - offset, center_y - offset), (center_x + offset, center_y + offset), 3)
                pygame.draw.line(screen, COLOR_PLAYER_X, (center_x + offset, center_y - offset), (center_x - offset, center_y + offset), 3)

def draw_status_text(screen, text):
    font = pygame.font.Font(None, 40)
    text_surface = font.render(text, True, COLOR_BLACK)
    text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, BOARD_SIZE + (SCREEN_HEIGHT - BOARD_SIZE) // 2))
    screen.blit(text_surface, text_rect)

def convert_coords_to_move(pos, rows, cols):
    x, y = pos
    if y > BOARD_SIZE or x > BOARD_SIZE:
        return None
    c = x // (BOARD_SIZE // cols)
    r = y // (BOARD_SIZE // rows)
    return r * cols + c

def get_list_state(game, state):
    if isinstance(game, (TicTacToe, Caro19x19)):
        return game.convert_mcts_state_to_list_state(state)
    raise TypeError(f"Loại game không xác định: {type(game).__name__}")

# === HÀM MỚI ĐỂ CHƠI LẠI ===
def reset_game(game, model_path):
    """Khởi tạo lại session và các biến trạng thái để bắt đầu ván mới."""
    print("---------------------------------")
    print("Bắt đầu ván mới!")
    new_session = Session(game, model_path, player_moves_first=True)
    game_over = False
    player_turn = True
    status_message = "Lượt của bạn!"
    return new_session, game_over, player_turn, status_message
# ===============================

# --- Hàm chính ---
def main():
    global SCREEN_WIDTH, SCREEN_HEIGHT, BOARD_SIZE
    
    parser = argparse.ArgumentParser()
    game_provider.add_game_argument(parser)
    parser.add_argument("-m", "--model", required=True, help="Tệp model để sử dụng")
    args = parser.parse_args()

    game = game_provider.get_game(args)
    
    print("========================================")
    print(f"Đang tải model từ file: {args.model}")
    session, game_over, player_turn, status_message = reset_game(game, args.model)
    print("Tải model thành công! Bắt đầu game.")
    print("========================================")

    rows, cols = get_board_dimensions(game)
    
    if isinstance(game, Caro19x19):
        BOARD_SIZE = 760
        SCREEN_WIDTH = 760
        SCREEN_HEIGHT = 820

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"AlphaZero - Chơi với AI ({type(game).__name__})")
    
    running = True
    while running:
        # === VÒNG LẶP SỰ KIỆN ĐÃ CẬP NHẬT ===
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Xử lý input khi game chưa kết thúc
            if not game_over and player_turn and event.type == pygame.MOUSEBUTTONDOWN:
                move = convert_coords_to_move(event.pos, rows, cols)
                if move is not None and session.is_valid_move(move):
                    player_won = session.move_player(move)
                    player_turn = False
                    if player_won:
                        status_message = "Bạn đã thắng! Nhấn SPACE để chơi lại."
                        game_over = True
                    elif session.is_draw():
                        status_message = "Hòa! Nhấn SPACE để chơi lại."
                        game_over = True
                    else:
                        status_message = "Máy đang suy nghĩ..."
            
            # Xử lý input khi game đã kết thúc (để chơi lại)
            if game_over and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    session, game_over, player_turn, status_message = reset_game(game, args.model)
        # ========================================

        # Xử lý logic và vẽ
        screen.fill(COLOR_WHITE)
        list_state = get_list_state(game, session.state)
        draw_grid(screen, rows, cols)
        draw_pieces(screen, list_state, game, rows, cols)
        draw_status_text(screen, status_message)
        pygame.display.flip()

        # Máy thực hiện nước đi
        if not game_over and not player_turn:
            time.sleep(0.5)
            bot_won = session.move_bot()
            player_turn = True
            if bot_won:
                status_message = "Máy đã thắng! Nhấn SPACE để chơi lại."
                game_over = True
            elif session.is_draw():
                status_message = "Hòa! Nhấn SPACE để chơi lại."
                game_over = True
            else:
                status_message = "Lượt của bạn!"

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()