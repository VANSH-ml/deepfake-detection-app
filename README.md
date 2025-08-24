# 🎭 DeepFake Detector

A DeepFake Detection system powered by **ResNet18** and deployed with **Gradio**.  
Upload an **image** or a **video**, and the model will classify it as **Real** or **Fake**.  

---

## 🚀 Features
- 🔍 Detects DeepFake in **images** and **videos**
- 🎥 Video frame sampling & majority voting
- 📷 Webcam support (optional)
- 🖥️ Interactive **Gradio UI**
- ⚡ Runs on **CPU / GPU**

---

## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/deepfake-detector.git
cd deepfake-detector
Create virtual environment & install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
python app.py
📂 Project Structure
bash
Copy
Edit
deepfake-detector/
│── app.py                  # Main Gradio app
│── requirements.txt        # Dependencies
│── deepfake_detector.pth   # Model weights
│── README.md               # Documentation
│── examples/               # Example images/videos
🔧 Usage
Image Tab → Upload an image & get prediction

Video Tab → Upload a video, system checks multiple frames

Output → Probability scores + Final label (Real / Fake)

📊 Example
Input	Output
<img src="examples/real_face.jpg" width="250"/>	✅ Real
<img src="examples/fake_face.jpg" width="250"/>	❌ Fake
