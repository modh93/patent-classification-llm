# Patent Classification with PatentBERT

This repository provides a complete pipeline for classifying patent abstracts using a fine-tuned BERT-based model (PatentBERT). It supports uploading datasets to Google Cloud Storage (GCS), fine-tuning the model on labeled data, and running inference at scale.

---

## Project Structure

```
├── patent_classification_pipeline.py   # Main script with training, inference, and GCS upload
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── LICENSE                            # MIT License file
```

---

## Features

- Upload patent datasets to Google Cloud Storage
- Fine-tune PatentBERT for binary classification
- Tokenize and preprocess patent abstracts
- Run inference on new patent data
- Fully modular and reusable components

---

## Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/patent-classification-llm.git
cd patent-classification
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Google Cloud SDK** (for GCS access)
```bash
gcloud auth login
gcloud config set project [YOUR_PROJECT_ID]
```

---

## Training the Model
```python
from patent_classification_pipeline import train_model

train_model(
    model_name="bert-base-uncased",
    dataset_path="path/to/labeled_data.csv",
    output_dir="model_output",
    epochs=5,
    batch_size=8
)
```

---

## Inference Example
```python
from patent_classification_pipeline import run_inference

preds = run_inference(
    model_path="model_output",
    tokenizer_path="model_output",
    texts=["A quantum system using entanglement..."]
)
print(preds)
```

---

## Upload Files to GCS
```python
from patent_classification_pipeline import upload_to_gcs

upload_to_gcs(
    bucket_name="your-gcs-bucket",
    source_folder="path/to/csv/files"
)
```

---

## Requirements
See `requirements.txt` for a full list of dependencies.

---
© 2025 Mohamed Keita
This project is licensed under the [MIT License](./LICENSE). 
