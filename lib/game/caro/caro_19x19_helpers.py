from typing import List, Tuple

Matrix = List[List[int]]
Coord = Tuple[int, int]


def check_win(matrix: Matrix, move: Coord, k: int, token: int) -> bool:
    """Kiểm tra xem nước đi cuối cùng có tạo thành chiến thắng k-ô liên tiếp hay không.

    Args:
        matrix (Matrix): Ma trận biểu diễn trạng thái bảng.
        move (Coord): Tọa độ (hàng, cột) của nước đi cuối cùng.
        k (int): Số lượng quân liên tiếp cần để thắng.
        token (int): Quân cờ của người chơi vừa di chuyển.

    Returns:
        bool: True nếu người chơi thắng, False nếu ngược lại.
    """
    row_idx, col_idx = move
    
    # Lấy hàng, cột, đường chéo chính, đường chéo phụ tại vị trí nước đi
    # và kiểm tra thắng trên từng đường.
    for func in (get_row, get_col, get_diag, get_antidiag):
        arr = func(matrix, (row_idx, col_idx))
        if k_in_a_row(arr, k, token):
            return True
    return False


def k_in_a_row(arr: List[int], k: int, token: int) -> bool:
    """Kiểm tra xem có k quân liên tiếp của một loại token trong một mảng hay không.

    Args:
        arr (List[int]): Một danh sách các quân cờ (ví dụ: một hàng, một cột, hoặc một đường chéo).
        k (int): Số lượng quân liên tiếp cần tìm.
        token (int): Quân cờ cần kiểm tra.

    Returns:
        bool: True nếu tìm thấy k quân liên tiếp, False nếu ngược lại.
    """
    assert k > 1, "Chúng tôi không xử lý các trường hợp tầm thường khi k <= 1"
    if len(arr) < k:
        # Không thể có k quân liên tiếp nếu độ dài mảng nhỏ hơn k
        return False

    current_consecutive = 0
    for i in range(len(arr)):
        if arr[i] == token:
            current_consecutive += 1
            if current_consecutive >= k:
                # Đã tìm thấy đủ k quân liên tiếp
                return True
        else:
            # Gặp một quân không khớp, đặt lại số lượng quân liên tiếp
            current_consecutive = 0
    return False


def get_row(matrix: Matrix, coord: Coord) -> List[int]:
    """Lấy hàng tại tọa độ đã cho từ ma trận."""
    row_idx = coord[0]
    return matrix[row_idx]


def get_col(matrix: Matrix, coord: Coord) -> List[int]:
    """Lấy cột tại tọa độ đã cho từ ma trận."""
    col_idx = coord[1]
    return [row[col_idx] for row in matrix]


def get_diag(matrix: Matrix, coord: Coord) -> List[int]:
    """Lấy đường chéo chính (từ trên-trái xuống dưới-phải) đi qua tọa độ đã cho.
    Args:
        matrix (Matrix): Ma trận.
        coord (Coord): Tọa độ (hàng, cột) cần lấy đường chéo.

    Returns:
        List[int]: Danh sách các phần tử trên đường chéo chính.
    """
    row_idx = coord[0]
    col_idx = coord[1]
    board_size = len(matrix)

    diag = []
    # Tìm điểm bắt đầu của đường chéo (ở biên trên hoặc biên trái)
    start_row = row_idx - min(row_idx, col_idx)
    start_col = col_idx - min(row_idx, col_idx)

    x, y = start_row, start_col
    while x < board_size and y < board_size:
        diag.append(matrix[x][y])
        x += 1
        y += 1
    return diag


def get_antidiag(matrix: Matrix, coord: Coord) -> List[int]:
    """Lấy đường chéo phụ (từ dưới-trái lên trên-phải) đi qua tọa độ đã cho.

    Args:
        matrix (Matrix): Ma trận.
        coord (Coord): Tọa độ (hàng, cột) cần lấy đường chéo phụ.

    Returns:
        List[int]: Danh sách các phần tử trên đường chéo phụ.
    """
    assert len(matrix) == len(matrix[0]), "Chúng tôi chỉ xử lý các bảng vuông"
    row_idx = coord[0]
    col_idx = coord[1]
    board_size = len(matrix)

    anti = []
    # Tìm điểm bắt đầu của đường chéo phụ (ở biên dưới hoặc biên trái)
    start_row = row_idx + min(board_size - 1 - row_idx, col_idx)
    start_col = col_idx - min(board_size - 1 - row_idx, col_idx)

    x, y = start_row, start_col
    while x >= 0 and y < board_size:
        anti.append(matrix[x][y])
        x -= 1
        y += 1
    return anti