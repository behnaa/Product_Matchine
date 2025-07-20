import os
import json
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, LlavaForConditionalGeneration, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Allow memory expansion on CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ----------- Config ------------
JSON_PATH = "/workspace/updated_with_qna.json"
IMG_DIR = "/workspace/images/images/"
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
MAX_LEN = 1024
EPOCHS = 5
BATCH_SIZE = 2
OUTPUT_DIR = "workspace/llava_finetuned_2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------- Read and Prepare JSON Data ------------
with open(JSON_PATH, 'r') as f:
    raw_data = json.load(f)

data = []
for item in raw_data:
    image_file = item["image"]
    for convo in item["conversations"]:
        if convo["from"] == "human":
            caption = convo["value"]
            data.append({"image_path": image_file, "caption": caption})
            break

df = pd.DataFrame(data)
df = df.dropna()
df = df[~df['caption'].str.contains("A picture of", case=False)]

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# ----------- Processor ------------
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# ----------- Dataset Class ------------
class ImageCaptionDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(IMG_DIR, row['image_path'])
        image = Image.open(image_path).convert("RGB")
        prompt = "<image>\n" + row["caption"]

        inputs = processor(
            text=prompt,
            images=image,
            padding=False,  # we pad manually in collate_fn
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": inputs["input_ids"].squeeze(0)
        }

# ----------- Custom Collate Function ------------
def custom_collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# ----------- Datasets ------------
train_dataset = ImageCaptionDataset(train_df)
val_dataset = ImageCaptionDataset(val_df)

# ----------- Load Model & Apply LoRA ------------
model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# ----------- Training Config ------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_strategy="epoch",
    num_train_epochs=EPOCHS,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    bf16=True,
    report_to="none"
)

# ----------- Trainer ------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,
    data_collator=custom_collate_fn
)

# ----------- Train ------------
trainer.train()

# Save model and processor
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
