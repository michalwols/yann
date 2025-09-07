#!/usr/bin/env python3
"""
BERT Text Classification Example using Yann Trainer

This example demonstrates how to:
1. Use a HuggingFace BERT model for text classification
2. Load and preprocess a HuggingFace dataset
3. Train the model using Yann's Trainer class with Params configuration
4. Evaluate the model performance

Requirements:
    uv add --optional transformers
    # or: pip install yann[transformers]

Usage:
    python examples/bert_classifier.py --dataset imdb --max_length 512 --epochs 3
    python examples/bert_classifier.py --dataset ag_news --lr 1e-5 --batch_size 32
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

from yann.train import Trainer
from yann.callbacks import History, Logger, Checkpoint, ProgressBar
from yann.data import Classes
from yann.params import Choice


class BertClassifier(nn.Module):
    """BERT model wrapper for text classification using Yann."""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, inputs):
        # Handle both dict and positional args
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            token_type_ids = inputs.get('token_type_ids')
        else:
            # Assume positional arguments (backward compatibility)
            input_ids, attention_mask, token_type_ids = inputs, None, None
            
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class HuggingFaceTextDataset(Dataset):
    """Wrapper to convert HuggingFace dataset to PyTorch Dataset."""
    
    def __init__(self, dataset, tokenizer, max_length=512, text_column='text', label_column='label'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item[self.text_column]
        label = item[self.label_column]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch):
    """Custom collate function for BERT inputs."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # Return as dict - new Yann trainer supports this!
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'labels': labels,
    }


def bert_loss(outputs, targets):
    """Custom loss function that handles dict targets."""
    if isinstance(targets, dict):
        targets = targets['labels']
    return nn.CrossEntropyLoss()(outputs, targets)


def accuracy_metric(outputs, targets):
    """Compute accuracy metric."""
    # Handle dict targets
    if isinstance(targets, dict):
        targets = targets['labels']
    
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == targets).float()
    return correct.mean()


class BertParams(Trainer.Params):
    """Configuration parameters for BERT text classification."""
    
    # Dataset configuration
    dataset: Choice(['imdb', 'ag_news', 'sst2']) = 'imdb'
    max_length: int = 512
    
    # Model configuration  
    model_name: str = 'bert-base-uncased'
    dropout: float = 0.1
    
    # Training configuration
    epochs: int = 3
    batch_size: int = 16
    val_batch_size: int = 32
    lr: float = 2e-5
    warmup_steps: int = 500
    
    # Optimization settings
    optimizer: str = 'AdamW'
    loss: str = 'CrossEntropyLoss'
    clip_grad_max_norm: float = 1.0
    
    # Device and performance
    device: Choice(['auto', 'cpu', 'cuda']) = 'auto'
    amp: bool = True
    
    # Output configuration
    root: str = './runs/bert_classifier'
    save_best: bool = True
    checkpoint_freq: int = 1


def load_text_classification_dataset(dataset_name, tokenizer, max_length=512):
    """Load and prepare a text classification dataset from HuggingFace."""
    
    if dataset_name == 'imdb':
        # Load IMDB movie reviews dataset
        dataset = load_dataset('imdb')
        train_dataset = HuggingFaceTextDataset(
            dataset['train'], 
            tokenizer, 
            max_length=max_length,
            text_column='text',
            label_column='label',
        )
        val_dataset = HuggingFaceTextDataset(
            dataset['test'], 
            tokenizer, 
            max_length=max_length,
            text_column='text', 
            label_column='label',
        )
        num_classes = 2
        class_names = ['negative', 'positive']
        
    elif dataset_name == 'ag_news':
        # Load AG News dataset
        dataset = load_dataset('ag_news')
        train_dataset = HuggingFaceTextDataset(
            dataset['train'],
            tokenizer,
            max_length=max_length,
            text_column='text',
            label_column='label',
        )
        val_dataset = HuggingFaceTextDataset(
            dataset['test'],
            tokenizer,
            max_length=max_length,
            text_column='text',
            label_column='label',
        )
        num_classes = 4
        class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        
    elif dataset_name == 'sst2':
        # Load Stanford Sentiment Treebank dataset
        dataset = load_dataset('glue', 'sst2')
        train_dataset = HuggingFaceTextDataset(
            dataset['train'],
            tokenizer,
            max_length=max_length,
            text_column='sentence',
            label_column='label',
        )
        val_dataset = HuggingFaceTextDataset(
            dataset['validation'],
            tokenizer,
            max_length=max_length,
            text_column='sentence',
            label_column='label',
        )
        num_classes = 2
        class_names = ['negative', 'positive']
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset, val_dataset, num_classes, class_names


def main():
    # Parse parameters using Yann's Params system
    params = BertParams.from_command()
    
    # Set device
    if params.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = params.device
    
    # Update device and amp settings
    params.device = device
    if device == 'cpu':
        params.amp = False  # Disable mixed precision on CPU
    
    print(f"Using device: {device}")
    print(f"Loading dataset: {params.dataset}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(params.model_name)
    
    # Load dataset
    train_dataset, val_dataset, num_classes, class_names = load_text_classification_dataset(
        params.dataset, tokenizer, params.max_length
    )
    
    print(f"Dataset loaded:")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Classes: {class_names}")
    
    # Create model
    model = BertClassifier(
        model_name=params.model_name,
        num_classes=num_classes,
        dropout=params.dropout,
    )
    
    # Create classes object for Yann
    classes = Classes(class_names)
    
    # Set up callbacks
    callbacks = [
        History(),
        Logger(batch_freq=100),
        ProgressBar(),
        Checkpoint(freq=params.checkpoint_freq),
    ]
    
    # Create Trainer with params
    trainer = Trainer(
        params,  # Pass the params object directly
        model=model,
        dataset=train_dataset,
        val_dataset=val_dataset,
        classes=classes,
        loss=bert_loss,  # Use custom loss that handles dict targets
        callbacks=callbacks,
        collate_fn=collate_fn,
        metrics={'accuracy': accuracy_metric},
        # BERT-specific optimizations
        clip_grad={'value': params.clip_grad_max_norm, 'mode': 'norm'},
    )
    
    print(f"\nTraining Configuration:")
    print(f"  - Model: {params.model_name}")
    print(f"  - Dataset: {params.dataset}")
    print(f"  - Epochs: {params.epochs}")
    print(f"  - Batch size: {params.batch_size}")
    print(f"  - Learning rate: {params.lr}")
    print(f"  - Max sequence length: {params.max_length}")
    print(f"  - Device: {device}")
    print(f"  - Mixed precision: {params.amp}")
    print(f"  - Optimizer: {params.optimizer}")
    
    # Train the model
    print(f"\nStarting training...")
    trainer(epochs=params.epochs)
    
    print(f"\nTraining completed!")
    print(f"Final metrics:")
    if hasattr(trainer, 'history') and trainer.history.metrics:
        latest_metrics = {k: v[-1] if v else 0 for k, v in trainer.history.metrics.items()}
        for metric, value in latest_metrics.items():
            print(f"  - {metric}: {value:.4f}")
    
    # Export the trained model
    print(f"\nExporting model...")
    export_path = trainer.export()
    print(f"Model exported to: {export_path}")
    
    return trainer


if __name__ == '__main__':
    try:
        trainer = main()
        
        # Example of how to use the trained model
        print(f"\nExample predictions:")
        if hasattr(trainer, 'model'):
            model = trainer.model
            model.eval()
            
            # Example texts for different datasets
            if trainer.params.dataset == 'imdb':
                examples = [
                    "This movie was absolutely fantastic! Great acting and plot.",
                    "Terrible film, waste of time. Very disappointing.",
                ]
            elif trainer.params.dataset == 'ag_news':
                examples = [
                    "The stock market rallied today as tech companies reported strong earnings.",
                    "Scientists discover new species in the Amazon rainforest.",
                ]
            else:  # sst2
                examples = [
                    "The movie was great!",
                    "This film is boring.",
                ]
            
            tokenizer = AutoTokenizer.from_pretrained(trainer.params.model_name)
            
            with torch.no_grad():
                for i, text in enumerate(examples):
                    # Tokenize
                    encoding = tokenizer(
                        text,
                        add_special_tokens=True,
                        max_length=trainer.params.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt',
                    )
                    
                    # Move to device
                    input_ids = encoding['input_ids'].to(trainer.device)
                    attention_mask = encoding['attention_mask'].to(trainer.device)
                    token_type_ids = encoding.get('token_type_ids', torch.zeros_like(input_ids)).to(trainer.device)
                    
                    # Predict
                    outputs = model(input_ids, attention_mask, token_type_ids)
                    probabilities = torch.softmax(outputs, dim=-1)
                    predicted_class = torch.argmax(probabilities, dim=-1).item()
                    confidence = probabilities[0][predicted_class].item()
                    
                    print(f"  Text: '{text}'")
                    print(f"  Predicted: {trainer.classes.names[predicted_class]} (confidence: {confidence:.3f})")
                    
    except ImportError as e:
        if 'transformers' in str(e):
            print("Error: transformers library not found.")
            print("Please install it with: pip install transformers")
        elif 'datasets' in str(e):
            print("Error: datasets library not found.")
            print("Please install it with: pip install datasets")
        else:
            print(f"Import error: {e}")
        exit(1)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        exit(1)