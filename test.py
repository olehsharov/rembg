from typing import Any
import streamlit as st
from loadimg import load_img
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image

torch.set_float32_matmul_precision(["high", "highest"][0])

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

@st.cache_resource
def load_model(device:str) -> Any:
    print("Loading model...")
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    birefnet.to(device)
    return birefnet

def process(image: Image.Image, device:str) -> Image.Image:
    print("Loading model")
    birefnet = load_model(device)
    print("Model loaded")
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    white_bg = Image.new("RGBA", image_size, (255, 255, 255, 255))
    result = Image.alpha_composite(white_bg, image)
    return result.convert("RGB")

def process_file(file:Path, save_to: Path, device:str):
    im:Image = load_img(str(file), output_type="pil")
    im = im.convert("RGB")
    if device == "cuda":
        im = im.to("cuda")
    output_path = save_to / file.name
    transparent = process(im, device)
    transparent.save(output_path)
    return output_path

def process_folder(input_folder:Path, output_folder:Path, device:str, progress:st, remove_input:bool):
    input_files = list(input_folder.iterdir())
    for i, file in enumerate(input_files):
        if file.is_file() and file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp"]:
            try:
                process_file(file, output_folder, device)
                progress.progress((i + 1) / len(input_files), text=f"Processing {file.name}")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(file)
                with col2:
                    st.image(output_folder / file.name)
                if remove_input:
                    file.unlink()
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
    progress.progress(1.0, text="Processing completed")

st.set_page_config(page_title="Remove Background", layout="wide")

st.title("Remove Background")

remove_input = st.checkbox("Remove input files", value=False)
device = st.selectbox("Device", options=["cpu", "mps", "cuda"], index=0)
input_folder = st.text_input("Input folder", value="./input")
output_folder = st.text_input("Output folder", value="./output")

if st.button("Process images"):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    progress = st.progress(0, text="Processing images...")
    process_folder(input_folder, output_folder, device, progress, remove_input)
