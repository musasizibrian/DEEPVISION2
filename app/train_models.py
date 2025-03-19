# train_models.py
from app.utils.dataset_processing import prepare_classification_dataset
from app.models.scene_classification import SceneClassificationModel
import os

def main():
    # Prepare dataset for classification
    source_dir = 'data/SCVD/SCVD_converted'  # Path to your dataset
    target_dir = 'data/processed_classification'
    classes = prepare_classification_dataset(source_dir, target_dir)
    
    print(f"Dataset prepared for classification with classes: {classes}")
    
    # Train classification model
    model = SceneClassificationModel(num_classes=len(classes))
    history, model_path = model.train(target_dir, epochs=20)
    
    print(f"Model training completed. Model saved to: {model_path}")

if __name__ == "__main__":
    main()