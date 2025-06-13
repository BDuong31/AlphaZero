Triển khai [Kiến trúc AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go) có cấu trúc lại cơ sở mã để có thể dễ dàng mở rộng với các trò chơi mới.Thiết kế sẵn lớp [m,n,k-game](https://en.wikipedia.org/wiki/M,n,k-game) và cấu trúc sẵn 2 trò chơi caro (19x19) và Tic Tac Toe (3x3).
# Nội dung thực tế
## Cài đặt
Môi trường được quản lý bằng [Pipenv](https://pipenv.pypa.io/en/latest/install/). Từ thư mục dự án của bạn, hãy chạy `pipenv install` để cài đặt tất cả các phụ thuộc từ Pipfile.
Sau đó, mỗi lần bạn muốn tải môi trường ảo vào shell của mình, chỉ cần chạy `pipenv shell` từ thư mục dự án.
Ngoài ra, bạn có thể sử dụng `pipenv run [command]` để chạy lệnh từ bên trong môi trường Pipenv.
Nếu bạn cần xóa môi trường ảo, bạn có thể thực hiện bằng `pipenv --rm`.
## Chọn trò chơi
Mỗi lệnh dưới đây có thể được chạy với cờ `-g` để chỉ định trò chơi bạn đang luyện tập hoặc chơi. `-g 0` cho caro (19 x 19) và `-g 1` cho TicTacToe (3 x 3).
## Đào tạo
Huấn luyện mô hình bằng `python train.py -g [game] -n [bất kỳ tên nào bạn muốn cho lần chạy này]`. Mô hình đã huấn luyện sẽ được lưu vào `saves/[run name]/[auto-generated-model-name].dat`.
Có thể quan sát số liệu thống kê huấn luyện bằng TensorBoard. Bắt đầu phiên bằng `tensorboard --logdir .` và TensorBoard sẽ mở trong trình duyệt (theo mặc định tại
[http://localhost:6006/](http://localhost:6006/)). Từ TensorBoard, bạn có thể quan sát được giá trị mất mát ở mỗi bước đào tạo và tỷ lệ chiến thắng của đối thủ so với mô hình tốt nhất hiện tại. Biểu đồ sau có thể sẽ tăng lên đến một điểm trước khi giảm mạnh khi mô hình tốt nhất được thay thế bằng một đối thủ đủ thành công, trước khi tăng trở lại khi một đối thủ mới trở nên tốt hơn. 
## Cho các mô hình chơi với nhau
Các mô hình cũng có thể chơi với nhau bằng cách sử dụng
`python play.py -g [game] model1_filename model2_filename ...` 
Bạn cũng có thể chỉ định số vòng mà các mô hình sẽ chơi bằng đối số `-r`
Tập lệnh sẽ in bảng xếp hạng các mô hình với chiến thắng, thua và hòa
# Thêm chi tiết
## Giới thiệu về AlphaZero
AlphaZero là một kiến ​​trúc học tăng cường có thể học cách chơi [thông tin hoàn hảo](https://en.wikipedia.org/wiki/Perfect_information) trò chơi cờ bàn 2 người chơi mà không cần kiến ​​thức về lĩnh vực của con người hoặc dữ liệu đào tạo. Đây là [một giải thích kỹ thuật rất hay](https://web.stanford.edu/~surag/posts/alphazero.html) về AlphaZero, nhưng tóm lại:
1. Một mạng nơ-ron với trọng số ngẫu nhiên được khởi tạo, nó thực hiện các nước đi ngẫu nhiên.
2. Mạng tự chơi với chính nó một số lần ghép cho đến khi hoàn thành. Bây giờ chúng ta có một số dữ liệu về giá trị của mỗi hành động ở mỗi trạng thái trò chơi. Điều này tạo ra dữ liệu đào tạo.
3. Mạng được đào tạo dựa trên dữ liệu được tạo ở bước 2. Nó (có thể) trở nên giỏi hơn trong việc thực hiện các nước đi. Điều này được đánh giá bằng cách để mạng mới được đào tạo chơi với mạng cũ.
4. Nếu mạng mới đánh bại mạng cũ (theo một tỷ lệ nhất định), thì nó sẽ trở thành mạng "tốt nhất" mới sẽ tự chơi để tạo dữ liệu đào tạo nhằm tạo ra một mạng tốt hơn, v.v.

Sẽ có thêm thông tin chi tiết trong từng thành phần.
## Thành phần
### MCTS
`lib/mcts.py`
Có thể nói đây là cốt lõi của AlphaZero. Bạn có thể nghĩ về MCTS như AlphaZero "suy nghĩ trước" từ một trạng thái trò chơi nhất định. MCTS về cơ bản là tiến hành một cây trò chơi, tức là một biểu đồ các trạng thái trò chơi với các cạnh là các hành động dẫn từ trạng thái này sang trạng thái khác. Đối với mọi trạng thái trò chơi `s` đã được khám phá và đối với mỗi hành động `a` tại mỗi trạng thái trò chơi `s`, dữ liệu sau đây được theo dõi:
1. Số lần hành động `a` đã được thực hiện tại trạng thái trò chơi `s`: `N(s,a)`
2. [Kết quả kỳ vọng](https://en.wikipedia.org/wiki/Q-learning) của hành động `a`: `Q(s,a)`
3. Xác suất trước đó để thực hiện hành động `a`, được dự đoán bởi mạng nơ-ron (tốt nhất hiện tại), cho trạng thái trò chơi `s`: `p(s,a)`
Từ những điều trên (và siêu tham số `c_puct`), chúng ta có thể tính toán giới hạn độ tin cậy trên đã điều chỉnh cho các giá trị Q của mỗi hành động tại mỗi trạng thái trò chơi `U(s,a)`.
Đối với hầu hết các trò chơi, rõ ràng là có nhiều trạng thái trò chơi hơn mức có thể khám phá, nhưng giá trị trên có thể đóng vai trò là phương pháp tìm kiếm để chọn trạng thái trò chơi nào cần kiểm tra.
Trong `lib/mcts.py`, các giá trị trên được lưu trữ, mỗi giá trị trong một `dict` với các khóa là dạng số nguyên của trạng thái trò chơi, mỗi trạng thái truy cập vào danh sách các giá trị tương ứng với các hành động của trò chơi. Về mặt lý thuyết, trạng thái trò chơi có thể là bất kỳ loại có thể lập chỉ mục nào, nhưng đối với triển khai này, loại `int` được chọn vì mạng nơ-ron chấp nhận các mảng số làm đầu vào và trong hầu hết các trường hợp, việc chuyển đổi trạng thái trò chơi từ một số nguyên duy nhất sang dạng có thể được mạng nơ-ron chấp nhận khá đơn giản.
#### Tìm kiếm trạng thái
Đối với mỗi lượt mô phỏng: cho một trạng thái, hãy chọn hành động có U cao nhất. Truyền nó cho logic trò chơi, trả về trạng thái trò chơi mới. Nếu tìm thấy trạng thái mới, hãy tra cứu U trong dict. Nếu không, hãy thêm trạng thái vào hàng đợi để mở rộng nút sau (việc mở rộng nút này được thực hiện theo từng đợt để hiệu quả hơn khi truy vấn các giá trị từ mạng nơ-ron Pytorch)
#### Mở rộng và sao lưu nút
Với hàng đợi các trạng thái mới, chưa gặp, nếu trạng thái không phải là trạng thái cuối cùng (thắng, thua hoặc hòa), hãy truy vấn mạng nơ-ron để biết dự đoán của nó về xác suất của từng hành động tại mỗi trạng thái trò chơi và giá trị trò chơi dự đoán chung tại trạng thái đó. Tạo các nút mới trong cây MCTS, tức là thêm trạng thái mới vào mỗi từ điển với xác suất dự đoán, 0 cho số lượng hành động và giá trị.
Nếu trạng thái là cuối cùng, chúng ta sẽ nhận được giá trị thực: -1 cho thua, 0 cho hòa, +1 cho thắng.
Chúng ta cũng thực hiện sao lưu: cập nhật giá trị trò chơi và số lần truy cập dọc theo đường dẫn đã thực hiện cho đến nay.
#### Tìm kiếm theo lô và tìm kiếm theo lô nhỏ
Điểm nghẽn của quy trình MCTS là truy vấn mạng nơ-ron để mở rộng các nút cây mới. Để hiệu quả hơn với việc này, chúng tôi truy vấn mạng nơ-ron theo lô của một số trạng thái lá (`search_minibatch()`). Tuy nhiên, điều này không tối ưu trong giai đoạn đầu của MCTS khi cây trò chơi chưa có nhiều người. Vì chúng tôi chỉ sao lưu các giá trị và số lượng nút sau một lô truy vấn, nên MCTS sẽ tự lặp lại nhiều lần trong một lô nhỏ. Do đó, để mở rộng cây nhiều hơn với mỗi bước MCTS, chúng tôi thực hiện một số tìm kiếm theo lô nhỏ này (`search_batch()`).
#### Nhận giá trị chính sách
Đối với quy trình tìm kiếm cây MCTS, chúng tôi chọn hành động có giá trị cao nhất một cách xác định tại mỗi trạng thái trò chơi. Nhưng đối với việc chơi thực tế (bao gồm cả tự chơi để tạo dữ liệu đào tạo), chúng tôi chọn ngẫu nhiên một hành động từ cây trạng thái dựa trên tần suất hành động đó được chọn, vì quy trình MCTS khiến các hành động tốt được chọn thường xuyên hơn. Mức độ khám phá được kiểm soát bởi siêu tham số Tau. Trong bài báo AlphaZero, đối với 30 lần di chuyển đầu tiên, Tau được đặt thành 1 (khám phá tối đa), do đó, nước đi thực tế là một lựa chọn ngẫu nhiên có trọng số với xác suất là số lần truy cập được chuẩn hóa của mỗi hành động. Sau 30 lần di chuyển, Tau = 0, tức là mô hình luôn chọn nước đi được truy cập nhiều nhất. Số bước trước khi đặt Tau = 0 là siêu tham số có thể điều chỉnh (`config.py`). Nó nên được đặt thành giá trị nhỏ hơn đối với các trò chơi đơn giản hơn
#### Các trường hợp ngoại lệ & Gotchas
Đối với nút gốc (trạng thái trò chơi ban đầu) khi chưa có dữ liệu để tính toán, chúng tôi tạo ra các xác suất bằng cách sử dụng [phân phối Dirichlet](https://stats.stackexchange.com/questions/322831/purpose-of-dirichlet-noise-in-the-alphazero-paper).
Giá trị chúng tôi nhận được phải luôn được đảo ngược trước khi được thêm vào các nút cây. Sau khi thực hiện một nước đi, "góc nhìn" của trò chơi sẽ bị đảo ngược (đến lượt người chơi khác, nếu họ thắng, bạn sẽ thua).
### Mô hình
`lib/model.py`
Đây là mạng nơ-ron với 4 lớp tích chập với chuẩn hóa theo lô, sử dụng Leaky ReLU để kích hoạt, như được đề xuất trong sách. Mạng chấp nhận các mảng 2D có kích thước tùy ý với 2 kênh (một cho mỗi người chơi). Tôi chưa thử nghiệm mạng sâu hơn. Mạng đưa ra một mảng head: chính sách có độ dài bằng không gian hành động, sau đó chúng ta softmax để có được xác suất và một giá trị head: ước tính duy nhất của phần thưởng tại trạng thái trò chơi đó.
### Trò chơi
Logic của mỗi trò chơi. Kiểm tra `lib/game/game.py` để biết giao diện mong đợi. Trò chơi cần có khả năng biểu diễn trạng thái trò chơi dưới dạng số nguyên cho MCTS. Cách đơn giản nhất để thực hiện điều này có lẽ là triển khai TicTacToe: sử dụng một chữ số cho mỗi quân cờ của người chơi và một chữ số cho các ô trống. Tuần tự hóa bảng trò chơi thành một chuỗi các chữ số đơn.
Trò chơi cũng cần cập nhật trạng thái trò chơi sau mỗi lần di chuyển (cũng được mong đợi là một số nguyên đơn), xác định xem nước đi có dẫn đến kết quả cuối cùng hay không, lấy danh sách các nước đi hợp lệ và bất hợp pháp dựa trên trạng thái trò chơi.
Cuối cùng, trò chơi có trách nhiệm chuyển đổi trạng thái trò chơi của mình thành danh sách các đầu vào để đào tạo mạng nơ-ron. Theo bài báo AlphaZero, đầu vào là một mảng 2 chiều 2 kênh, với mỗi kênh là vị trí của các quân cờ của một người chơi trên bảng trò chơi. MCTS sẽ nhóm các trạng thái trò chơi lại với nhau trong một danh sách để đào tạo mạng theo từng đợt, do đó trò chơi sẽ có thể chuyển đổi danh sách các trạng thái trò chơi thành danh sách các mảng có thể nhập vào mạng.
Để thêm trò chơi mới, chỉ cần thêm một mô-đun khác vào thư mục `lib/game` và triển khai giao diện `BaseGame` được định nghĩa trong `lib/game/game.py`. Danh mục các trò chơi khả dụng được lưu trong `lib/game/game_provider.py`. Sửa đổi mục này để cung cấp trò chơi của bạn cho các tập lệnh train, play.
## tham số
Tất cả các siêu tham số có thể được tìm thấy trong `config.py`. Các giá trị được lấy từ bài báo AlphaZero cho Go, trừ khi có ghi chú khác.