import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import gradio as gr
import os

# ==========================
# 1. Device Setup
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# 2. Define Model (ResNet18)
# ==========================
model = models.resnet18(weights=None)   # Initialize ResNet18 (no pretrained)
model.fc = nn.Linear(512, 2)            # 2 classes: Real / Fake

# ==========================
# 3. Load Saved Weights
# ==========================
checkpoint_path = "deepfake_detector.pth"

if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device)

    # ✅ Handle cases where model was saved with DataParallel (keys prefixed with "module.")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
else:
    print(f"⚠️ Warning: Checkpoint '{checkpoint_path}' not found. Using random weights!")

model = model.to(device)
model.eval()

# ==========================
# 4. Image Preprocessing
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================
# 5. Prediction Function
# ==========================
def predict_image(image):
    try:
        img = Image.fromarray(image).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)[0]  # ✅ Use softmax for proper scores
            _, predicted = torch.max(outputs, 1)

        label = "Real" if predicted.item() == 1 else "Fake"
        return {"Real": float(probs[1].item()), "Fake": float(probs[0].item())}, label

    except Exception as e:
        return {"Error": str(e)}, "Prediction Failed"


def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []
    preview_frames = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every 10th frame
        if frame_count % 10 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preview_frames.append(rgb_frame)

            img = Image.fromarray(rgb_frame).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img)
                _, predicted = torch.max(outputs, 1)
                frame_predictions.append(predicted.item())

        frame_count += 1

    cap.release()

    if len(frame_predictions) == 0:
        return "Error: No frames processed", []

    # Majority voting
    final_prediction = 1 if frame_predictions.count(1) > frame_predictions.count(0) else 0
    label = "Real" if final_prediction == 1 else "Fake"

    return label, preview_frames[:10]  # Show up to 10 preview frames

# ==========================
# 6. Gradio Interface
# ==========================
with gr.Blocks() as demo:
    gr.Markdown("# 🎭 DeepFake Detector (ResNet18)")
    gr.Markdown("Upload an **image** or a **video** to check if it's Real or Fake.")

    with gr.Tab("Image"):
        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload Image")
            image_output_label = gr.Label(num_top_classes=2, label="Prediction Scores")
            final_label = gr.Textbox(label="Final Label")

        image_button = gr.Button("Predict Image")
        image_button.click(
            fn=predict_image,
            inputs=image_input,
            outputs=[image_output_label, final_label]
        )

    with gr.Tab("Video"):
        with gr.Row():
            video_input = gr.Video(label="Upload Video")
            video_output_label = gr.Textbox(label="Final Label")
        
        output_gallery = gr.Gallery(label="Preview Frames / Image")
        video_button = gr.Button("Predict Video")

        video_button.click(
            fn=predict_video,
            inputs=video_input,
            outputs=[video_output_label, output_gallery]
        )

# ==========================
# 7. Launch App
# ==========================
if __name__ == "__main__":
    demo.launch(debug=True, share=True)
