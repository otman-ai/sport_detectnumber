import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
import easyocr
import re
from PIL import Image

MODEL_PATH = "yolov8n.pt"
SRC_PATH = "uploaded_video.mp4"

@st.cache
def load_model():
  return YOLO(MODEL_PATH)

model = load_model()
reader = easyocr.Reader(["en"])
flat_list = {}
for c in labels:
  flat_list[c] = []
  
st.title("Uplaod the play here")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov"])

if uploaded_file!=None:
  with open("uploaded_video.mp4", "wb") as f:
    f.write(video_data)
  st.success("Video saved successfully.")
  results = model.track(source=SRC_PATH, show=False,save=True) 
  classes = np.array(results[0].boxes.cls)
  
  numbers = {}
  labels = list(results[0].names.values())
  for c in labels:
    numbers[c] = []
  for result in results:
    for k,i in enumerate(result.boxes.data[:,:5]):
      x_min,y_min,x_max,y_max ,id = np.array(i).astype(int)
      box = result.orig_img[y_min:y_max, x_min:x_max]
      try:
        if box != np.array([]):
          if reader.readtext(box)!=[]:
            numbers[labels[k]].append(re.findall(r'\d+', reader.readtext(box)[0][1]))
      except:
        print("Error")
  for v,k in zip(numbers.values(),numbers.keys()):
    for n in list(set([item for sublist in v for item in sublist if item])):
      if len(str(n)) <=2 and n !=0:
        flat_list[k].append(int(n))

  st.text(flat_list)
