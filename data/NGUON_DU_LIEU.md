# Nguồn dữ liệu thực nghiệm

Tất cả ảnh đều là ảnh mẫu chuẩn công khai, đã kiểm tra đọc được bằng `cv2.imread` và đúng màu
(ảnh scikit-image được lưu bằng chuyển RGB → BGR chuẩn). Mỗi ảnh có một preset trong `config/<tên>.json`
(màu mục tiêu + ngưỡng H, S, V + tham số hậu xử lý đã tinh chỉnh và kiểm tra bằng mắt).

## 1. Ảnh thật – `data/images/` (16 ảnh)

| Ảnh | Nguồn | Kích thước (W×H) | Màu mục tiêu | Mục đích |
|---|---|---|---|---|
| 01_smarties.png | OpenCV samples | 413×356 | Đỏ (Hue vòng) | **Ảnh minh họa chính** cho pipeline: kẹo nhiều màu trên nền trắng, có viền tối và vệt phản sáng |
| 02_peppers.png | Ảnh bài Lab (`Lab/Images/peppers.png`) | 512×512 | Đỏ (Hue vòng) | Ớt đỏ cạnh ớt xanh |
| 03_fruits.jpg | OpenCV samples | 512×480 | Cam | Lát cam giữa nhiều loại quả |
| 04_baboon.jpg | OpenCV samples | 512×512 | Xanh dương | Hai má xanh của khỉ mandrill |
| 05_happyfish.jpg | OpenCV samples | 259×194 | Vàng | Ảnh hoạt hình màu thuần |
| 06_apple.jpg | OpenCV samples | 512×512 | Đỏ | Quả táo trên nền vàng lục |
| 07_orange.jpg | OpenCV samples | 512×512 | Cam | Quả cam có phần bị bóng |
| 08_opencv_logo.png | OpenCV samples | 600×794 | Đỏ / Xanh lá / Xanh dương | Màu thuần → **có mask chuẩn chính xác** (`data/masks/`) để tính IoU trên ảnh thật |
| 09_board.jpg | OpenCV samples | 640×480 | Xanh lá | Ví dụ **hạn chế**: nền trùng màu |
| 10_butterfly.jpg | OpenCV samples | 493×356 | Xanh lá | Lá cây quanh con bướm |
| 11_astronaut.png | scikit-image `astronaut()` | 512×512 | Cam | Bộ đồ phi hành gia màu cam |
| 12_coffee.png | scikit-image `coffee()` | 600×400 | Đỏ | Đĩa, tách đỏ (S cao để tránh mặt bàn gỗ) |
| 13_chelsea.png | scikit-image `chelsea()` | 451×300 | Cam | Ví dụ **hạn chế**: lông mèo màu không đồng nhất |
| 14_ihc.png | scikit-image `immunohistochemistry()` | 512×512 | Cam/nâu (DAB) | Ứng dụng **y sinh**: tách vùng nhuộm nâu |
| 15_colorwheel.png | scikit-image `colorwheel()` | 371×370 | Xanh lá | Thấy rõ dải Hue được chọn |
| 16_home.jpg | OpenCV samples | 512×384 | Xanh dương | Bầu trời |

URL gốc OpenCV: `https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/<tên file>`
(smarties.png, fruits.jpg, baboon.jpg, HappyFish.jpg, apple.jpg, orange.jpg, opencv-logo.png, board.jpg, butterfly.jpg, home.jpg).
`opencv-logo.png` gốc có kênh alpha; `evaluate.py --tao-du-lieu` tạo mask chuẩn từ alpha + màu tô rồi ghép logo lên nền trắng.

## 2. Ảnh tổng hợp có mask chuẩn – `data/synthetic/` (6 ảnh, 640×480)

Sinh bằng `python evaluate.py --tao-du-lieu` (seed cố định 2026). Hàng `images/` là ảnh, `masks/` là đáp án (0/255).

| Ảnh | Tình huống kiểm thử | Màu mục tiêu |
|---|---|---|
| syn_01_do_hue_vong | Đỏ nằm ở cả hai đầu Hue (H≈1–5 và 173–178), có cam và tím gây nhiễu | Đỏ |
| syn_02_xanh_nhat | Xanh dương nhạt, S chỉ 45–77 (thí nghiệm S_min 80 vs 30) | Xanh dương |
| syn_03_nhieu_muoi_tieu | 15% nhiễu xung màu (thí nghiệm Median) | Xanh lá |
| syn_04_bong_toi | Ánh sáng giảm dần + vùng bóng (thí nghiệm V_min) | Vàng |
| syn_05_nhieu_gauss | Nhiễu Gauss σ = 40 (thí nghiệm hình thái học) | Xanh dương |
| syn_06_mau_gan_nhau | Cam đặt cạnh đỏ và vàng (Hue gần nhau) | Cam |

## 3. Video – `data/videos/smarties_chuyen_dong.avi`

6 giây, 25 fps, 640×480, tạo từ `01_smarties.png` (xoay, dịch chuyển, độ sáng thay đổi 50–100%) bằng
`evaluate.py` để demo chức năng mở video và kiểm tra độ ổn định của ngưỡng HSV khi ánh sáng đổi.
