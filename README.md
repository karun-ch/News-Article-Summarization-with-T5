# 📰 News Article Summarization with T5 (Text Generation)

## 📌 Project Overview
This project demonstrates **news article summarization** using **Google's T5 (Text-To-Text Transfer Transformer)**. The model takes a long-form article and generates a concise summary while preserving key information.

## 🚀 Features
- Uses **T5-small** pre-trained model from Hugging Face.
- Summarizes long-form news articles efficiently.
- Implements **CNN/DailyMail dataset** for real-world summarization.
- Runs on **Google Colab** with GPU acceleration.
- Generates high-quality, extractive summaries.

## 📂 Dataset
- **CNN/DailyMail** dataset is used for training/testing.
- It contains news articles and corresponding human-written summaries.
- Automatically loaded from **Hugging Face datasets**.

## 🛠️ Installation & Setup
### 1️⃣ Install Required Libraries
Run the following command in **Google Colab**:
```bash
pip install transformers datasets torch
```

### 2️⃣ Load Pre-Trained T5 Model
```python
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load T5 tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
```

## 📜 Summarizing News Articles
```python
# Load CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:1%]")  # Small subset

# Function to summarize text
def summarize(text, max_input_length=512, max_output_length=150):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=max_input_length, truncation=True).to(device)
    summary_ids = model.generate(inputs.input_ids, max_length=max_output_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Test summarization
sample_text = dataset[0]["article"]
print("Original Article:\n", sample_text[:1000], "...")  # Show first 1000 chars
print("\nGenerated Summary:\n", summarize(sample_text))
```

## 🖥️ Running on Google Colab
- Upload the **notebook** to **Google Colab**.
- Ensure **GPU is enabled** (`Runtime > Change runtime type > GPU`).

## 🔮 Future Improvements
- Fine-tune T5 on **custom datasets**.
- Experiment with **T5-base or T5-large** for better summaries.
- Deploy as an **API or Web App** for real-world use.

---
🚀 **Get Started Now!** Experiment with different articles and tweak parameters to generate better summaries. 🎯

