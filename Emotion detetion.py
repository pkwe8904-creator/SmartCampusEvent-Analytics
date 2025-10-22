# import tensorflow as tf
# from tensorflow.keras import layers, models, datasets
# import numpy as np

# # Load FER-2013 dataset (or any 48x48 dataset you have)
# # Example assumes you already preprocessed data into X_train, y_train, etc.

# # Normalize
# X_train = X_train / 255.0
# X_test = X_test / 255.0

# # Model (same as your design)
# emotion_model = models.Sequential([
#     layers.Input(shape=(48, 48, 1)),
#     layers.Conv2D(32, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D(2,2),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(7, activation='softmax')
# ])

# emotion_model.compile(optimizer='adam',
#                       loss='sparse_categorical_crossentropy',
#                       metrics=['accuracy'])

# # Train
# emotion_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)

# # Save weights
# emotion_model.save_weights("emotion_model.h5")




from fer import FER
import cv2

detector = FER(mtcnn=True)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions
    result = detector.detect_emotions(frame)
    if result:
        (x, y, w, h) = result[0]["box"]
        emotions = result[0]["emotions"]
        emotion = max(emotions, key=emotions.get)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (255,0,0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
