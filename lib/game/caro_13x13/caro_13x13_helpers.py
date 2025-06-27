from typing import List, Tuple

Matrix = List[List[int]]
Coord = Tuple[int, int]


def check_win(matrix: Matrix, move: Coord, k: int, token: int) -> bool:
    """
        [tóm tắt]

        Đối số:
        matrix (Ma trận): [mô tả]
        move (Coord): [mô tả]
        k (int): [mô tả]
        token (int): [mô tả]

        Trả về:
        bool: [mô tả]
    """
    for func in (get_row, get_col, get_diag, get_antidiag):
        arr = func(matrix, move)
        if k_in_a_row(arr, k, token):
            return True
    return False


def k_in_a_row(arr: List[int], k: int, token: int) -> bool:
    """
        [tóm tắt]

        Đối số:
        arr (Danh sách[int]): [mô tả]
        k (int): [mô tả]
        token (int): [mô tả]

        Trả về:
        bool: [mô tả]
    """
    assert k > 1, "Chúng tôi không xử lý các trường hợp tầm thường khi k <= 1"
    if len(arr) < k:
        # Không thể nào
        return False

    matchStartIndex = None
    for i in range(len(arr)):
        if arr[i] == token:
            if matchStartIndex is None:
                matchStartIndex = i
            elif (i - matchStartIndex + 1) >= k:
                # Có >= k mã token, chúng ta có thể trả về
                return True
        else:
            # Đã tìm thấy mã token không khớp. Đặt lại
            matchStartIndex = None
            if i >= (len(arr) - k):
                # Còn lại ít hơn k token, chúng tôi không thể khớp
                return False
    return False


def get_row(matrix: Matrix, coord: Coord) -> List[int]:
    """
    [tóm tắt]

    Đối số:
        matrix (Ma trận): [mô tả]
        coord (Coord): [mô tả]

    Trả về:
        [kiểu]: [mô tả]
    """
    return matrix[coord[0]]


def get_col(matrix: Matrix, coord: Coord) -> List[int]:
    """
    [tóm tắt]

    Đối số:
        matrix (Ma trận): [mô tả]
        coord (Coord): [mô tả]
    """
    col_idx = coord[1]
    return [row[col_idx] for row in matrix]


def get_diag(matrix: Matrix, coord: Coord) -> List[int]:
    """
    [tóm tắt]

    _|0|1|2|3|

    0|_|_|_|_|
    1|_|_|_|_|
    2|_|_|_|_|
    3|_|_|_|_|

    Đối số:
        matrix (Ma trận): [mô tả]
        coord (Coord): [mô tả]

    Trả về:
        List[int]: [mô tả]
    """
    row_idx = coord[0]
    col_idx = coord[1]

    if row_idx >= col_idx:
        # Các giá trị này là đúng đối với đường chéo chính
        # (từ trên cùng bên trái đến dưới cùng bên phải qua tâm bảng)
        # và các đường chéo nhỏ hơn khác bên dưới nó
        # Các đường chéo này luôn chứa cột đầu tiên
        col_start = 0
        row_start = row_idx - col_idx
        # Các đường chéo này luôn chứa hàng cuối cùng
        row_end = len(matrix) - 1
        # Do tính đối xứng.
        # Vẽ và quan sát hàng mà đường chéo kết thúc
        # là chỉ số đảo ngược của cột nơi nó bắt đầu
        col_end = row_end - col_start
    else:
        # Các đường chéo phía trên đường chéo chính luôn chứa hàng đầu tiên        
        row_start = 0
        col_start = col_idx - row_idx
        # và cột cuối cùng
        col_end = len(matrix) - 1
        row_end = col_end - col_start

    diag = []
    x = row_start
    y = col_start
    while x <= row_end and y <= col_end:
        diag.append(matrix[x][y])
        x += 1
        y += 1
    return diag


def get_antidiag(matrix: Matrix, coord: Coord) -> List[int]:
    """
    Chúng ta có các đường chéo ngược (từ dưới cùng bên trái đến trên cùng bên phải)

    Đối số:
        matrix (Ma trận): [description]
        coord (Coord): [description]

    Trả về:
        List[int]: [description]
    """
    assert len(matrix) == len(matrix[0]), "chúng tôi chỉ xử lý hình vuông"
    row_idx = coord[0]
    col_idx = coord[1]

    if row_idx + col_idx < len(matrix):
        # Các giá trị này là đúng đối với đường chéo chính
        # (từ dưới cùng bên trái đến trên cùng bên phải qua tâm bảng)
        # và các đường chéo nhỏ hơn khác ở trên nó
        # Các đường chéo này luôn chứa cột đầu tiên
        col_start = 0
        # hàng dưới cùng. Chúng ta bắt đầu giảm dần từ đây
        row_start = row_idx + col_idx
        # Các đường chéo đối này luôn chứa hàng cuối cùng
        row_end = 0
        # Do tính đối xứng.
        # Vẽ và quan sát cột mà đường chéo ngược kết thúc
        # có cùng chỉ số với hàng mà nó bắt đầu
        col_end = row_start
    else:
        # Các đường chéo đối diện bên dưới đường chéo chính luôn chứa cột cuối cùng
        col_end = len(matrix) - 1
        col_start = col_idx + row_idx - col_end
        # Và hàng cuối cùng. Chúng ta cũng bắt đầu giảm dần từ đây
        row_start = len(matrix) - 1
        row_end = col_start

    anti = []
    x = row_start
    y = col_start
    while x >= row_end and y <= col_end:
        anti.append(matrix[x][y])
        # Thu thập từ dưới cùng của ma trận lên trên cùng
        x -= 1
        # Chúng ta vẫn đang thu thập từ trái sang phải
        # vì vậy chúng ta vẫn tăng y
        y += 1
    return anti
