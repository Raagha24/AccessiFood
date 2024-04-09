import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk
import pytesseract
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
import pyttsx3

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#pre-trained DenseNet121 model 
model = DenseNet121(weights='imagenet')

# app
app = tk.Tk()
app.geometry("500x500")
app.title("AccessiFood")

#To take snapshot
def take_snapshot():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('snapshot.jpg', frame)
    cap.release()

#preprocessing image
def preprocess(path):
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expanded = np.expand_dims(img_array, axis=0)
    img_processed = preprocess_input(img_expanded)
    return img_processed


#classifying image
def classify(processed_img):
    preds = model.predict(processed_img)
    results = decode_predictions(preds, top=5)[0]
    return results

#detecting text from image
def detect_text(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    return text

def show_snapshot(image_path):
    img = Image.open(image_path)
    img = img.resize((300, 300), Image.ANTIALIAS) 
    img = ImageTk.PhotoImage(img)
    snapshot_label.config(image=img)
    snapshot_label.image = img

# image prediction
def image_prediction():
    take_snapshot()  #snapshot using the webcam
    path = 'snapshot.jpg'  
    input_img = preprocess(path)
    results = classify(input_img)
    text = detect_text(path)
    result_str = f"The product you have is: {results[0][1]} \n\nIt reads:\n{text}"
    result_label.config(text=result_str)

# TTS conversion
def convert_text_to_audio(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

#tkinter buttons
snapshot_button = tk.Button(app, text="Take Snapshot", command=image_prediction)
snapshot_button.pack(pady=10)

snapshot_label = tk.Label(app, bg='white')
snapshot_label.pack(pady=10)

#tkinter labels
result_label = tk.Label(app, text="", wraplength=300, justify="left")
result_label.pack(pady=10)

def quit_program(event=None):
    app.quit()

play_audio_button = tk.Button(app, text="Play Audio", command=lambda: convert_text_to_audio(result_label.cget("text")))
play_audio_button.pack(pady=10)

app.bind('q', quit_program)

app.mainloop()
