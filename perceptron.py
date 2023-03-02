import numpy as np
import pandas as pd

from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

df = pd.read_csv('drug.csv')     #kiểu dữ liệu DataFrame: cấu trúc dữ liệu 2 chiều
data1 = np.array(df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K','Drug']].values) #trả về mảng 2 chiều[n,5]

print(data1)
#hàm chuẩn hóa dữ liệu
def data_encoder(data):
    for i, j in enumerate(data):
        for k in range(0, 6):
            if (j[k] == "F"):
                j[k] = 0
            elif (j[k] == "M"):
                j[k] = 1
            elif (j[k] == "LOW"):
                j[k] = 2
            elif (j[k] == "NORMAL"):
                j[k] = 3
            elif (j[k] == "HIGH"):
                j[k] = 4
    return data

#chuẩn hóa dữ liệu trong data1
data = data_encoder(data1)
print('Tap du lieu sau khi chuyen doi: ')
print(data)

#chia dữ liệu thành 2 phần: train 70%, test 30%
dt_Train, dt_Test = train_test_split(data, test_size=0.3 , shuffle = True)

X_train = dt_Train[:, :5]
y_train = dt_Train[:, 5]
X_test = dt_Test[:, :5]
y_test = dt_Test[:, 5]

#perceptron: thuật toán phân lớp dữ liệu
pla = Perceptron()
pla.fit(X_train, y_train)   #huấn luyện mô hình dựa trên tập x_train, y_train

#tính giá trị dự đoán trên dữ liệu đầu vào là X_test
y_predict = pla.predict(X_test)

count = 0       #lấy biến đếm
print("Thực tế\t\tDự đoán")
for i in range(0,len(y_predict)): 
    print(y_test[i],"\t\t", y_predict[i])
    if(y_test[i] == y_predict[i]):
        count = count +1

print('Ty le du doan dung: ', count/len(y_predict)*100, '%')
print('Ti le du doan sai:' , 100-count/len(y_predict)*100, '%') 


