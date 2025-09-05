# =============================================================================
# DỰ ĐOÁN BỆNH TIỂU ĐƯỜNG SỬ DỤNG MẠNG NƠ-RON NHÂN TẠO (ANN)
# =============================================================================

# --- Bước 1: Gọi các thư viện cần thiết ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- Bước 2: Tải và chuẩn bị dữ liệu ---
print(">>> Bước 2: Đang tải và chuẩn bị dữ liệu...")

# Tải tập dữ liệu Pima Indians Diabetes Dataset từ URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df = pd.read_csv(url, names=columns)

# Xử lý các giá trị 0 không hợp lệ, thay thế bằng giá trị trung bình của cột
# Đây là một bước quan trọng để cải thiện độ chính xác của mô hình
cols_to_replace = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_to_replace:
    # Thay thế 0 bằng giá trị NA (Not Available) để tính toán
    df[col].replace(0, pd.NA, inplace=True)
    # Tính giá trị trung bình (bỏ qua các giá trị NA)
    mean_val = df[col].mean()
    # Điền giá trị trung bình vào các ô bị thiếu (NA)
    df[col].fillna(mean_val, inplace=True)

# Tách biến đầu vào (X - features) và biến mục tiêu (y - target)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# --- Bước 3: Chia và chuẩn hóa dữ liệu ---
print(">>> Bước 3: Đang chia và chuẩn hóa dữ liệu...")

# Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)
# stratify=y để đảm bảo tỷ lệ outcome trong tập train và test là tương đồng
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Chuẩn hóa dữ liệu: Đưa tất cả các feature về cùng một thang đo
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Bước 4: Xây dựng mô hình Mạng Nơ-ron (ANN) ---
print(">>> Bước 4: Đang xây dựng mô hình ANN...")

model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(X_train.shape[1],))) # Lớp vào và lớp ẩn 1
model.add(Dense(8, activation='relu'))                                  # Lớp ẩn 2
model.add(Dense(1, activation='sigmoid'))                               # Lớp ra (cho bài toán phân loại nhị phân)

# In ra cấu trúc của mô hình
print("\n✅ Cấu trúc mô hình ANN:")
model.summary()
print("-" * 40)

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Bước 5: Huấn luyện mô hình ---
print(">>> Bước 5: Bắt đầu quá trình huấn luyện mô hình...")

# Huấn luyện mô hình với dữ liệu training
# verbose=1 để hiển thị thanh tiến trình cho mỗi epoch
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,
                    verbose=1)

print("\n✅ Hoàn tất huấn luyện!")
print("-" * 40)

# --- Bước 6: Đánh giá mô hình ---
print(">>> Bước 6: Đánh giá hiệu suất mô hình...")

# Đánh giá hiệu suất của mô hình trên tập dữ liệu kiểm tra
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'📊 Độ mất mát trên tập kiểm tra (Loss): {loss:.4f}')
print(f'🎯 Độ chính xác trên tập kiểm tra (Accuracy): {accuracy*100:.2f}%')
print("-" * 40)