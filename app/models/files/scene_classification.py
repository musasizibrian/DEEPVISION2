# app/models/scene_classification.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import os
import numpy as np
from PIL import Image

class SceneClassificationModel:
    def __init__(self, model_path=None, num_classes=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['Normal', 'Violence', 'Weaponized']
        
        if model_path and os.path.exists(model_path):
            # Load the saved model
            self.model = models.mobilenet_v2(pretrained=False)
            self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            # Create a new model based on MobileNetV2
            self.model = models.mobilenet_v2(pretrained=True)
            # Replace the classifier with a new one for our classes
            self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def train(self, data_dir, epochs=10, batch_size=32, learning_rate=0.001):
        """Train the classification model"""
        # Set model to training mode
        self.model.train()
        
        # Create data transformations for training with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Initialize tracking variables
        best_val_loss = float('inf')
        best_model_path = os.path.join('models', 'scene_classification_best.pth')
        
        # Make sure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Calculate average training loss and accuracy
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Track statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate average validation loss and accuracy
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save the model if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Saved best model at epoch {epoch+1}")
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print statistics
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save the final model
        final_model_path = os.path.join('models', 'scene_classification_final.pth')
        torch.save(self.model.state_dict(), final_model_path)
        
        # Load the best model for subsequent use
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        
        return history, best_model_path
    
    def classify(self, image):
        """Classify a single image"""
        # Convert image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        # Apply transformations
        image = self.transforms(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Get top prediction
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        confidence = probabilities[class_idx].item()
        class_name = self.class_names[class_idx]
        
        # Return prediction details
        return {
            "class": class_name,
            "confidence": confidence,
            "predictions": {self.class_names[i]: probabilities[i].item() for i in range(len(self.class_names))}
        }