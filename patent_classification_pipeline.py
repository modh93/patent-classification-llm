
# patent_classification_pipeline.py

import os
import logging
from typing import List
from google.cloud import storage
from transformers import BertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# === CONFIGURATION LOGGING ===
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === GCS UPLOAD ===
def upload_to_gcs(bucket_name: str, source_folder: str) -> None:
    """
    Upload all files from a local folder to a Google Cloud Storage bucket.

    Args:
        bucket_name (str): Name of the GCS bucket.
        source_folder (str): Path to the local folder containing files to upload.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for file_name in os.listdir(source_folder):
        local_path = os.path.join(source_folder, file_name)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded {file_name} to gs://{bucket_name}/")

# === TOKENIZATION ===
def tokenize_and_encode(dataset: Dataset, tokenizer_path: str) -> Dataset:
    """
    Tokenize and encode a dataset using a specified tokenizer.

    Args:
        dataset (Dataset): The raw Hugging Face dataset with text and labels.
        tokenizer_path (str): Path to a pretrained tokenizer.

    Returns:
        Dataset: Tokenized and label-encoded dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    def encode_labels(example):
        example["label"] = int(example["label"])
        return example

    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.map(encode_labels)
    return tokenized_dataset

# === TRAINING ===
def train_model(model_name: str, dataset_path: str, output_dir: str, epochs: int = 5, batch_size: int = 8) -> None:
    """
    Fine-tune a BERT model for binary classification on a labeled patent dataset.

    Args:
        model_name (str): Path or name of a pretrained model (e.g., 'bert-base-uncased').
        dataset_path (str): Path to the CSV file containing labeled patent texts.
        output_dir (str): Directory where the trained model and tokenizer will be saved.
        epochs (int): Number of training epochs. Default is 5.
        batch_size (int): Batch size for training and evaluation. Default is 8.
    """
    logger.info("Loading dataset...")
    dataset = load_dataset("csv", data_files={"data": dataset_path})["data"]
    tokenized = tokenize_and_encode(dataset, model_name)
    split_dataset = tokenized.train_test_split(test_size=0.2)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        acc = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, output_dict=True)
        logger.info(f"\nClassification Report:\n{classification_report(labels, predictions)}")
        return {"accuracy": acc, "f1": report['weighted avg']['f1-score']}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed. Saving model...")
    model.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(model_name).save_pretrained(output_dir)
    logger.info(f"Model and tokenizer saved to {output_dir}")

# === INFERENCE ===
def run_inference(model_path: str, tokenizer_path: str, texts: List[str]) -> List[int]:
    """
    Run inference on a list of patent abstracts using a trained model.

    Args:
        model_path (str): Path to the trained model directory.
        tokenizer_path (str): Path to the tokenizer directory.
        texts (List[str]): List of patent abstracts to classify.

    Returns:
        List[int]: List of predicted labels (0 or 1).
    """
    logger.info("Loading model and tokenizer for inference...")
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model.eval()

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    logger.info(f"Inference done for {len(texts)} entries")
    return predictions.tolist()
