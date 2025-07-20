# 🛒 Product_Matching

This project focuses on **matching similar or duplicate products** using a fine-tuned **Llava (Large Language and Vision Assistant)** model. It includes the following core components:

- 📦 Data deployment to MongoDB  
- 🧠 Model fine-tuning on custom product data  
- 🔎 Inference for real-world product predictions  
- 🌐 Flask-based API setup (in progress)

---

## 🧱 Tech Stack

- 🐍 Python 3.8+
- 🍃 MongoDB (Local or Atlas)
- 🔗 Llava / Transformers
- ⚙️ PyTorch
- 🧪 scikit-learn
- 🐼 pandas, numpy
- 🌐 Flask (for serving inference)

---

## 📁 Project Files Overview

| File           | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `Train.py`     | Uses a pre-trained Llava model and fine-tunes it on custom product data.   |
| `Mdb.py`       | Full implementation to upload and store product data in MongoDB.            |
| `Test.py`      | Loads the trained model and performs predictions on test inputs.            |
| `Inference.py` | Flask-based API for prediction (not fully tested due to memory limitations).|

---

## 🚀 How to Use This Project

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/behnaa/Product_Matchine.git
cd product_matching
