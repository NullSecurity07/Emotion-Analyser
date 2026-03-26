<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:050510,50:101030,100:3b82f6&height=220&section=header&text=Emotion%20Analyser&fontSize=56&fontColor=ffffff&fontAlignY=38&fontAlign=50&desc=Real-Time%20CNN%20Facial%20Expression%20Detection&descSize=16&descAlignY=58&descAlign=50&descColor=93c5fd80&animation=fadeIn" width="100%"/>

</div>

<br/>

<div align="center">

*From raw camera feed to real-time emotion classification — powered by deep learning.*

</div>

<br/>

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=0:3b82f6,100:101030&height=1&section=header" width="40%"/>
</div>

<br/>

<div align="center">

### — Overview —

</div>

<br/>

Emotion Analyser is a computer vision project that implements a Convolutional Neural Network (CNN) for real-time facial expression recognition. It processes live webcam feeds, isolates human faces, and classifies the subject's expression into one of seven distinct emotional states. 

The architecture separates model training from execution, allowing for rapid iterations on custom datasets while maintaining a lightweight runtime for live inference.

<br/>

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=0:3b82f6,100:101030&height=1&section=header" width="40%"/>
</div>

<br/>

<div align="center">

### — Architecture —

</div>

<br/>

| Component | Function |
|:---|:---|
| `trainmodel.py` | CNN compilation, dataset ingestion, and model training |
| `realtimedetection.py` | Live webcam feed capture and bounding-box inference overlay |
| `emotiondetector.json` | Exported neural network architecture |
| `emotiondetector.h5` | Compiled and saved model weights |
| `history.json` | Serialized training metrics (accuracy and loss) |

<br/>

**Inference Pipeline**

```text
Webcam Input (Live Feed)
    └─ Face Detection (Haarcascade / OpenCV)
        └─ Image Preprocessing (Grayscale, Resize, Normalize)
            └─ CNN Inference (emotiondetector.h5)
                ├─ Angry
                ├─ Disgust
                ├─ Fear
                ├─ Happy
                ├─ Neutral
                ├─ Sad
                └─ Surprise
                    └─ Display overlay on bounding box
```

<br/>

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=0:3b82f6,100:101030&height=1&section=header" width="40%"/>
</div>

<br/>

<div align="center">

### — Setup —

</div>

<br/>

**Requirements**

- Python 3.8+
- A working webcam

<br/>

```bash
# 1. Clone the repository
git clone https://github.com/NullSecurity07/Emotion-Analyser
cd emotion_analyser

# 2. Initialize virtual environment
python -m venv venv

# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

<br/>

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=0:3b82f6,100:101030&height=1&section=header" width="40%"/>
</div>

<br/>

<div align="center">

### — Usage —

</div>

<br/>

**1. Dataset Preparation**

Ensure your training data is structured in the `assets/images/` directory before initiating a training run:

```text
assets/images/
├── train/
│   ├── angry/
│   ├── disgust/
│   └── ... (fear, happy, neutral, sad, surprise)
└── validation/
    ├── angry/
    ├── disgust/
    └── ...
```

<br/>

**2. Model Training**

Train the CNN from scratch. You can adjust the directories, epochs, and batch sizes via CLI arguments:

```bash
python trainmodel.py \
  --train_dir assets/images/images/train \
  --test_dir assets/images/images/validation \
  --epochs 50 \
  --batch_size 128
```

*Note: This generates `emotiondetector.json`, `emotiondetector.h5`, and `history.json` upon completion.*

<br/>

**3. Real-Time Detection**

Launch the inference script to open your webcam and begin live classification:

```bash
python realtimedetection.py
```

<br/>

**4. Performance Visualization**

Plot the training accuracy and loss curves from a previous run:

```bash
python trainmodel.py --plot_history history.json
```

<br/>

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=0:3b82f6,100:101030&height=1&section=header" width="40%"/>
</div>

<br/>

<div align="center">

### — Stack —

</div>

<br/>

| Technology | Purpose |
|:---|:---|
| **Python** | Core runtime |
| **TensorFlow / Keras** | CNN architecture, compilation, and model training |
| **OpenCV** | Computer vision, webcam interfacing, and face isolation |
| **NumPy** | Array manipulation and image data handling |
| **Matplotlib** | Training history and performance visualization |

<br/>

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,50:101030,100:050510&height=120&section=footer&animation=fadeIn" width="100%"/>

</div>
