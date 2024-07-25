import cv2
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import h5py


def load_yolo_model():
    net = cv2.dnn.readNet('yolo-coco/yolov3.weights', 'yolo-coco/yolov3.cfg')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, output_layers


def load_classes():
    with open('yolo-coco/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def load_age_model():
    age_net = cv2.dnn.readNetFromCaffe('age/deploy_age.prototxt', 'age/age_net.caffemodel')
    return age_net


def load_emotion_model(filepath):
    with h5py.File(filepath, 'r') as f:
        if 'optimizer_weights' in f.attrs:
            del f.attrs['optimizer_weights']
    emotion_model = load_model(filepath, compile=False)
    emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return emotion_model


def detect_objects(frame, net, output_layers, classes):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.3  # Lowered confidence threshold
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces


def predict_age(face_img, age_net):
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]
    return age


def predict_emotion(face_img, emotion_model):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img.astype("float") / 255.0
    face_img = img_to_array(face_img)
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    preds = emotion_model.predict(face_img)[0]
    emotion = emotion_labels[preds.argmax()]
    return emotion


def process_image(image_path, net, output_layers, classes, face_cascade, age_net, emotion_model):
    img = cv2.imread(image_path)
    if img is None:
        print("Could not open or find the image:", image_path)
        return
    img = detect_objects(img, net, output_layers, classes)
    faces = detect_faces(img, face_cascade)
    for (x, y, w, h) in faces:
        face_img = img[y:y + h, x:x + w]
        age = predict_age(face_img, age_net)
        emotion = predict_emotion(face_img, emotion_model)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, f'Age: {age}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(img, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path, net, output_layers, classes, face_cascade, age_net, emotion_model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video file:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    process_fps = max(int(fps / 2), 1)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % process_fps == 0:
            frame = cv2.resize(frame, (640, 480))
            frame = detect_objects(frame, net, output_layers, classes)
            faces = detect_faces(frame, face_cascade)
            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                age = predict_age(face_img, age_net)
                emotion = predict_emotion(face_img, emotion_model)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'Age: {age}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow('Video', frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main(file_path):
    net, output_layers = load_yolo_model()
    classes = load_classes()
    age_net = load_age_model()
    emotion_model = load_emotion_model('emotion/emotion_model.hdf5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        process_image(file_path, net, output_layers, classes, face_cascade, age_net, emotion_model)
    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
        process_video(file_path, net, output_layers, classes, face_cascade, age_net, emotion_model)
    else:
        print(f"Unsupported file format: {ext}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <image_or_video_path>")
        sys.exit(1)
    file_path = sys.argv[1]
main(file_path)