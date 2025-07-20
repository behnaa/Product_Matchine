import os
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# --- Config ---
MODEL_PATH = "workspace/llava_finetuned_2"
IMAGE_PATH = "/workspace/images/images/00a3aa58-5742-4178-8ea3-2d4b99156dfc.jpg"
QUESTION = "Give me decription?"

device = "cpu"  # Force CPU usage

# --- Load model and processor ---
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = LlavaForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)  # Use float32 for CPU
model.to(device)
model.eval()

# --- Load image and prepare input ---
image = Image.open(IMAGE_PATH).convert("RGB")
prompt = f"<image>\n{QUESTION}."

MAX_LEN = 1024  # Define MAX_LEN if not defined already

def extract_category(text):
    lines = text.strip().split("\n")
    for line in lines:
        if line.lower().startswith("category:"):
            return line.split(":", 1)[1].strip()
    return text.strip()

def run_inference(image, prompt):
    inputs = processor(
        text=prompt,
        images=image,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    raw_answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    category = extract_category(raw_answer)
    return category

category = run_inference(image, prompt)
print("Predicted category:", category)
