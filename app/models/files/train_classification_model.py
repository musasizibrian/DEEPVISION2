# train_classification_model.py
import os
import sys
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from app.utils.dataset_processing import prepare_classification_dataset
from app.models.scene_classification import SceneClassificationModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
    print("Starting scene classification model training...")
    
    # Set up paths
    source_dir = 'data/SCVD/SCVD_converted'
    target_dir = 'data/processed_classification'
    models_dir = 'models'
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Prepare dataset
    print("Preparing dataset...")
    classes = prepare_classification_dataset(source_dir, target_dir)
    print(f"Dataset prepared with classes: {classes}")
    
    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"scene_classifier_{timestamp}"
    
    # Training parameters
    epochs = 30
    batch_size = 32
    learning_rate = 0.001
    
    # Create and train the model
    print("Initializing model...")
    model = SceneClassificationModel(num_classes=len(classes))
    
    print(f"Training model for {epochs} epochs with batch size {batch_size}...")
    history, model_path = model.train(
        data_dir=target_dir, 
        epochs=epochs, 
        batch_size=batch_size, 
        learning_rate=learning_rate
    )
    
    # Plot training history
    plot_training_history(history, model_name)
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    evaluate_model(model, target_dir, batch_size)
    
    print(f"Training complete! Model saved to: {model_path}")
    
    return model_path

def plot_training_history(history, model_name):
    """Plot and save training history graphs"""
    # Create figure directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'])
    plt.plot(history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f'figures/{model_name}_training_history.png')
    plt.close()
    
    print(f"Training plots saved to figures/{model_name}_training_history.png")

def evaluate_model(model, data_dir, batch_size):
    """Evaluate the model on the test set"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up test data
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Switch to evaluation mode
    model.model.eval()
    
    # Initialize variables for metrics
    test_correct = 0
    test_total = 0
    
    # Create confusion matrix
    num_classes = len(test_dataset.classes)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    # Evaluate with no gradient
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model.model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Update metrics
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    # Calculate accuracy
    accuracy = test_correct / test_total
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Calculate per-class metrics
    class_names = test_dataset.classes
    print("\nPer-Class Metrics:")
    
    for i, class_name in enumerate(class_names):
        # Calculate true positives, false positives, and false negatives
        true_pos = confusion_matrix[i, i].item()
        false_pos = confusion_matrix[:, i].sum().item() - true_pos
        false_neg = confusion_matrix[i, :].sum().item() - true_pos
        
        # Calculate precision, recall, and F1 score
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("Predicted →")
    print("            " + " ".join(f"{name:>10}" for name in class_names))
    print("Actual ↓")
    
    for i, name in enumerate(class_names):
        row = f"{name:10}" + " ".join(f"{confusion_matrix[i, j].int().item():10d}" for j in range(num_classes))
        print(row)

if __name__ == "__main__":
    main()