import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import cv2
import numpy as np
from cnn_architecture import CNN
import torchvision.transforms as transforms

# Load the model and set it to evaluation mode
model = CNN(3, 2)
model.load_state_dict(torch.load('./Malaria-model.pth', map_location=torch.device('cpu')))
model.eval()

def get_hotspot(image):
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Perform inference
    image_tensor.requires_grad = True
    logits = model(image_tensor)
    probs = nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1)

    # Compute gradients
    model.zero_grad()
    logits[0, predicted_class].backward()

    # Compute the heatmap
    gradients = image_tensor.grad[0].detach().cpu().numpy()
    heatmap = np.mean(gradients, axis=0)

    # Normalize the heatmap
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (image.width, image.height))

    # Apply colormap to the heatmap (optional)
    heatmap = cv2.applyColorMap(np.uint8(heatmap * 255), cv2.COLORMAP_JET)

    # Overlay the heatmap on the original image
    image_np = np.array(image)
    overlay = cv2.addWeighted(image_np, 0.7, heatmap, 0.3, 0)

    return image_np, overlay, predicted_class.item()

# Create a Streamlit app
def main():
    st.title("Malaria Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        original_image, overlayed_image, predicted_class = get_hotspot(image)

        # Display the original image and overlayed image side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_column_width=True)
        with col2:
            st.subheader("Overlayed Image")
            st.image(overlayed_image, use_column_width=True)

        # Display the predicted class in block letters
        class_mapping = {0: "Parasitized Image", 1: "Uninfected Image"}
        st.subheader("Predicted Class")
        st.markdown(f"<h2 style='text-align: center;'>{class_mapping[predicted_class]}</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
