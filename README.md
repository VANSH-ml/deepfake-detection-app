🎭 DeepFake Detector

A state-of-the-art DeepFake Detection System built using PyTorch (ResNet18 backbone) and deployed via Gradio.

This project allows users to detect AI-manipulated media in both images and videos. With the rising threats of misinformation, identity fraud, and malicious content creation, DeepFake detection systems play a vital role in ensuring trust and safety in digital media.

📖 Background

DeepFakes use generative models to create hyper-realistic fake content, which can be exploited for:

Political misinformation 📰

Identity fraud & scams 💳

Harassment & blackmail 📢

Our detector analyzes uploaded images or video frames, extracts features via ResNet18, and outputs whether the content is Real or Fake.

By combining frame sampling, majority voting, and a user-friendly Gradio UI, the system makes detection accessible to both technical and non-technical users.

🚀 Features

🔍 Image Detection – Upload any face image to classify as Real or Fake

🎥 Video Detection – Upload videos (MP4/AVI), frames are sampled and analyzed

🖥 Web Interface – Simple drag-and-drop Gradio interface

📷 Webcam Support – Take a live picture and check authenticity instantly

⚡ Fast Inference – Optimized preprocessing & model evaluation

🧩 Modular Codebase – Easily replace backbone model (ResNet, EfficientNet, etc.)

📂 Project Structure
deepfake-detector/
│── app.py                  # Main Gradio app
│── requirements.txt        # Dependencies
│── deepfake_detector.pth   # Trained model weights
│── README.md               # Documentation
│── .gitignore              # Ignore unnecessary files
│── examples/               # Example images/videos
│── notebooks/              # Training / experimentation notebooks
│── utils/                  # Helper functions (video frame extraction etc.)

🛠 Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/deepfake-detector.git
cd deepfake-detector


Install dependencies:

pip install -r requirements.txt


Run the application:

python app.py

📊 Usage
▶ Image Mode

Upload an image (.jpg, .png)

Model outputs Probability (Real vs Fake) and final classification

▶ Video Mode

Upload a video (.mp4, .avi)

Frames are extracted → Processed → Classified

Majority vote decides final result

▶ Webcam Mode

Capture a live picture via Gradio webcam input

Model checks authenticity in real-time

📈 Example Results
Input	Output

	✅ Real

	❌ Fake
🔬 Technical Details

Model Backbone: ResNet18 (transfer learning on custom dataset)

Preprocessing: Resize → Normalize (ImageNet mean & std)

Loss Function: CrossEntropyLoss

Optimizer: Adam

Video Handling: OpenCV for frame extraction, sampling every n frames

Deployment: Gradio for interactive frontend

📌 Future Work

🔧 Add multi-face detection (handle group photos)

🛡 Improve robustness against adversarial attacks

🌍 Deploy as a web service / API

📱 Mobile app integration
