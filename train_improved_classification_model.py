# train_improved_classification_model.py
import os
import sys
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import traceback

# Try to import the necessary modules, with clear error handling
try:
    from app.utils.dataset_processing import prepare_classification_dataset
    from app.models.improved_scene_classification import SceneClassificationModel
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Make sure the following files exist:")
    print("  - app/utils/dataset_processing.py")
    print("  - app/models/improved_scene_classification.py")
    sys.exit(1)

def main():
    """Main function with hardcoded optimal parameters"""
    try:
        print("Starting improved scene classification model training...")
        print(f"Current working directory: {os.getcwd()}")

        # Set up paths
        source_dir = 'data/SCVD/SCVD_converted'
        target_dir = 'data/processed_classification'
        models_dir = 'models'

        # Training parameters - hardcoded with optimal values
        model_type = 'efficientnet'  # Options: 'mobilenet', 'resnet', 'efficientnet'
        frame_count = 5
        use_weighted_sampler = True
        epochs = 50
        batch_size = 32
        learning_rate = 0.0005
        dropout_rate = 0.5
        weight_decay = 1e-4
        patience = 15
        mixup_alpha = 0.2
        label_smoothing = 0.1

        print(f"Using model type: {model_type}")
        print(f"Frame count: {frame_count}")
        print(f"Using weighted sampler: {use_weighted_sampler}")
        print(f"Training for {epochs} epochs with learning rate {learning_rate}")

        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

        # Prepare dataset
        print("Preparing dataset...")
        video_data = prepare_classification_dataset(source_dir, target_dir)
        classes = ['Normal', 'Violence', 'Weaponized']

        print(f"Dataset prepared with classes: {classes}")

        # Create timestamp for model versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"scene_classifier_{model_type}_{timestamp}"

        # Create and train the model
        print(f"Initializing {model_type} model...")
        model = SceneClassificationModel(
            num_classes=len(classes), 
            video_data=video_data, 
            dropout_rate=dropout_rate,
            model_type=model_type,
            frame_count=frame_count,
            use_weighted_sampler=use_weighted_sampler
        )
        
        print(f"Training model for {epochs} epochs with batch size {batch_size}...")
        history, model_path = model.train(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            mixup_alpha=mixup_alpha,
            label_smoothing=label_smoothing
        )

        # Plot training history
        plot_training_history(history, model_name)

        # Evaluate model on test set
        print("Evaluating model on test set...")
        metrics = model.evaluate(verbose=True)

        # Save evaluation metrics
        save_evaluation_metrics(metrics, model_name, classes)

        print(f"Training complete! Model saved to: {model_path}")

        return model_path
    
    except Exception as e:
        print(f"ERROR: An exception occurred during training: {e}")
        traceback.print_exc()
        return None

def plot_training_history(history, model_name):
    """Plot and save enhanced training history graphs"""
    try:
        # Create figure directory if it doesn't exist
        os.makedirs('figures', exist_ok=True)

        # Create a figure with multiple subplots
        plt.figure(figsize=(15, 10))
        
        # Plot accuracy
        plt.subplot(2, 2, 1)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Validation')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Plot loss
        plt.subplot(2, 2, 2)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Plot learning rate
        if 'learning_rates' in history:
            plt.subplot(2, 2, 3)
            plt.plot(history['learning_rates'])
            plt.title('Learning Rate Schedule')
            plt.ylabel('Learning Rate')
            plt.xlabel('Epoch')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.yscale('log')  # Learning rates often best viewed on log scale

        # Add accuracy vs loss scatter plot
        if len(history['train_loss']) > 0:
            plt.subplot(2, 2, 4)
            plt.scatter(history['train_loss'], history['train_acc'], label='Train', alpha=0.5)
            plt.scatter(history['val_loss'], history['val_acc'], label='Validation', alpha=0.5)
            plt.title('Accuracy vs Loss')
            plt.xlabel('Loss')
            plt.ylabel('Accuracy')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)

        # Save figure
        plt.tight_layout()
        plt.savefig(f'figures/{model_name}_training_history.png', dpi=300)
        plt.close()

        print(f"Enhanced training plots saved to figures/{model_name}_training_history.png")
    except Exception as e:
        print(f"WARNING: Failed to plot training history: {e}")

def save_evaluation_metrics(metrics, model_name, class_names):
    """Save evaluation metrics to a text file"""
    try:
        os.makedirs('evaluations', exist_ok=True)
        
        with open(f'evaluations/{model_name}_metrics.txt', 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Test Loss: {metrics['loss']:.4f}\n\n")
            
            f.write("Per-Class Metrics:\n")
            confusion_matrix = metrics['confusion_matrix']
            
            for i, class_name in enumerate(class_names):
                # Calculate true positives, false positives, and false negatives
                true_pos = confusion_matrix[i, i]
                false_pos = np.sum(confusion_matrix[:, i]) - true_pos
                false_neg = np.sum(confusion_matrix[i, :]) - true_pos
                
                # Calculate precision, recall, and F1 score
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {precision:.4f}\n")
                f.write(f"  Recall:    {recall:.4f}\n")
                f.write(f"  F1-Score:  {f1:.4f}\n\n")
            
            # Write confusion matrix
            f.write("Confusion Matrix:\n")
            header = "            " + " ".join(f"{name:>10}" for name in class_names)
            f.write(header + "\n")
            
            for i, name in enumerate(class_names):
                row = f"{name:10}" + " ".join([f"{int(confusion_matrix[i, j]):10d}" for j in range(len(class_names))])
                f.write(row + "\n")
        
        print(f"Evaluation metrics saved to evaluations/{model_name}_metrics.txt")
    except Exception as e:
        print(f"WARNING: Failed to save evaluation metrics: {e}")

if __name__ == "__main__":
    main()