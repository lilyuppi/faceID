
Tiền xử lý dữ liệu để cắt khuôn mặt từ ảnh gốc

1. cắt riêng khuôn mặt ra để train: 

python3 src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25

2. Train model

python3 src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000

3. Nhận diện bằng webcam

python3 src/face_rec_cam.py 
