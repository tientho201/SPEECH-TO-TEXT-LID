# 🎙️ Speech-to-Text & Language Identification Model

Một giải pháp tích hợp Deep Learning để nhận diện ngôn ngữ (LID) và chuyển đổi giọng nói thành văn bản (STT) với độ chính xác cao. Mô hình phân loại gồm 6 nhãn **(en, vi, jp, ko, th, zh)** .Được finetuning trên model **facebook/wav2vec2-large-xlsr-53**, dữ liệu được lấy từ voxlingua107 và huấn luyện trên GPU **L40s**.

## 📖 Giới thiệu
Dự án này cung cấp một quy trình (pipeline) hoàn chỉnh từ khâu xử lý dữ liệu âm thanh thô đến việc huấn luyện và đánh giá mô hình. Hệ thống được thiết kế để:

1. Phân loại ngôn ngữ: Xác định ngôn ngữ của đoạn audio đầu vào.

2. Nhận dạng giọng nói: Chuyển đổi tín hiệu âm thanh thành văn bản tương ứng.

Dự án được tối ưu hóa để chạy trên môi trường Notebook (.ipynb) giúp dễ dàng theo dõi, trực quan hóa dữ liệu và gỡ lỗi (debug).

## 📂 Cấu trúc thư mục

```
.
├── datasets/                   # Quản lý dữ liệu
│   ├── raw/                    # Dữ liệu gốc
│   ├── processed/              # Dữ liệu đã xử lý
│   ├── train/                  # Dữ liệu huấn luyện
│   ├── val/                    # Dữ liệu validation
│   └── test/                   # Dữ liệu test
├── models/                     # Quản lý mô hình
│   ├── checkpoints/    # Checkpoints trong quá trình training
│   ├── model/                  # Mô hình đã huấn luyện xong
├── requirements.txt            # Chứa các thư viện cần thiết 
└── train_model.ipynb           # File chứa tất cả
```

## ⚙️ Yêu cầu hệ thống
Để đảm bảo quá trình huấn luyện diễn ra suôn sẻ, hệ thống cần đáp ứng:

| Thành phần | Yêu cầu tối thiểu | Khuyên dùng |
| :--------- | :-----------------| :-----------|
|OS          |Linux / Windows / MacOS | Linux (Ubuntu 20.04+) |
|Python|3.8|3.10+|
|Framework|PyTorch / TensorFlow|PyTorch (CUDA Support)|
|GPU| NVIDIA RTX 3060 (12GB VRAM) hoặc T4/P100 (Cloud) | L40s (46GB VRAM)|
|RAM|16 GB|32 GB+|
|Storage|SSD 100GB|SSD 500GB (NVMe)|

## 🚀 Cài đặt & Thiết lập
* Bước 1: Clone dự án

```Bash

git clone https://github.com/tientho201/speech-to-text-lid.git
cd speech-to-text-lid
```

* Bước 2: Tạo môi trường ảo (Khuyến khích)

```Bash

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

* Bước 3: Cài đặt thư viện

```Bash

pip install --upgrade pip
pip install -r requirements.txt
```
## 🛠️ Hướng dẫn Sử dụng

Toàn bộ quy trình được tích hợp trong file train_model.ipynb. Hãy mở Jupyter Notebook và thực hiện tuần tự:

**1. Chuẩn bị dữ liệu**

* Đặt file âm thanh gốc vào datasets/raw/.

* Chạy các cell ở phần **Preprocessing** trong notebook để làm sạch, trích xuất đặc trưng (Spectrogram/MFCC) và chia tập train/val/test vào các thư mục tương ứng.

**2. Huấn luyện (Training)**

* Cấu hình Hyperparameters (Learning rate, Batch size, Epochs).

* Lưu ý quan trọng: Code đã được cấu hình để lưu checkpoint sau mỗi epoch vào thư mục **models/checkpoints/**.

⚠️ Cảnh báo: Nếu chạy trên Google Colab hoặc máy thuê, hãy mount Google Drive hoặc tải checkpoints về máy cá nhân thường xuyên để tránh mất tiền và công sức nếu session bị ngắt kết nối.

**3. Đánh giá (Evaluation)**

* Sử dụng tập datasets/test/ để tính toán độ chính xác.

* Các chỉ số quan trọng:

    * LID: Accuracy, F1-Score.

    * STT: WER (Word Error Rate), CER (Character Error Rate).

## 🤝 Đóng góp
Mọi đóng góp đều được hoan nghênh! Vui lòng mở Issue để thảo luận về những thay đổi lớn trước khi gửi Pull Request.

# 📝 Lưu ý:
1. **Sửa Code:** Vì tôi đã đổi tên thư mục từ checkpoints + model thành checkpoints trong README cho chuẩn, bạn hãy vào file train_model.ipynb và sửa lại đường dẫn lưu file tương ứng nhé (xóa đoạn + model đi).

2. **Quá trình huấn luyện**: Tôi huấn luyện trên L40s với mô hình Wav2Vec2 (khá nặng) mất khoảng 4 tiếng nên khuyến khích cân nhắc nếu như muốn finetuning Model như tôi.

3. Có thể tải **dataset** thông qua link bên dưới:

* **Dataset** đã qua xử lí và nén thành các file .arrow
    * **[Link train](https://www.kaggle.com/datasets/tinthnguyn/lid-all-train)**
    * **[Link val và test](https://www.kaggle.com/datasets/tinthnguyn/lid-val-test)**

* **Dataset** chưa qua xử lí là các file wav
    * **[Link raw](https://drive.google.com/drive/folders/1imzGYI9ihO7RCOQP9PMglXp-ovnk4GOo?usp=drive_link)**