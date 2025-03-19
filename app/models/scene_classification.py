import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import os
import numpy as np
import cv2
from PIL import Image
from sklearn.utils import class_weight

class VideoDataset(Dataset):
    def __init__(self, video_data, transform=None, frame_rate=1, split='train'):
        """
        Args:
            video_data (dict): Dictionary of video paths returned by prepare_classification_dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            frame_rate (int):  Frames per second to extract.
            split (str): 'train', 'test', or 'val'.
        """
        self.video_paths = []
        self.labels = []
        self.transform = transform
        self.frame_rate = frame_rate

        # Check if the split is valid
        if split not in ['train', 'test']:
            raise ValueError("Split must be 'train' or 'test'")

        self.split = split
        self.class_names = list(video_data[split].keys())  # Extract class names

        # Populate video_paths and labels from the video_data dictionary
        for class_name in self.class_names:
            for video_path in video_data[split][class_name]:
                self.video_paths.append(video_path)
                self.labels.append(class_name)

        # Create class_to_idx mapping
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self._extract_frames(video_path, self.frame_rate)

        if len(frames) > 0:
            # Take the first frame only for simplicity (can be modified for temporal information)
            frame = frames[0]

            # Convert to RGB (OpenCV loads as BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame) #Convert the numpy array to a PIL image

            if self.transform:
                frame = self.transform(frame)
        else:
            print(f"Warning: No frames extracted from {video_path}")
            return torch.zeros((3, 224, 224)), -1  # Placeholder

        return frame, self.class_to_idx[label]

    def _extract_frames(self, video_path, frame_rate):
        """Extracts frames from a video."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / frame_rate)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frames.append(frame)
            frame_count += 1

        cap.release()
        return frames

class SceneClassificationModel:
    def __init__(self, model_path=None, num_classes=3, video_data=None, dropout_rate=0.3):  # Added dropout_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate  # Store dropout rate

        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),  # Increased rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Added saturation and hue
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.2), shear=(-15, 15)), #random zoom/sheer
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize Datasets
        self.train_dataset = VideoDataset(video_data, transform=self.train_transform, split='train')
        self.test_dataset = VideoDataset(video_data, transform=self.test_transform, split='test')

        # Initialize DataLoaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False, num_workers=4)

        # Model initialization
        self.model = models.mobilenet_v2(pretrained=True)
        # Add dropout layer
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),  # Add dropout
            nn.Linear(self.model.last_channel, num_classes)
        )
        self.model = self.model.to(self.device)

        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

    def train(self, epochs=10, batch_size=32, learning_rate=0.001, weight_decay=1e-4, patience=5): #Added Patience

        # Compute class weights
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(self.train_dataset.labels),
            y=self.train_dataset.labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay) #Added weight decay
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5)  # Added patience

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

        # Early stopping variables
        counter = 0 #counter of how many epochs the model did not improve in
        best_epoch = 0 #best epoch to restore the weights

        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in self.train_loader:
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
                for inputs, labels in self.test_loader: # Use test loader as validation data
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
                best_epoch = epoch
                counter = 0
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Saved best model at epoch {epoch+1}")
            else:
                counter +=1

            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Print statistics
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            #Early Stopping Check
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load the best model for subsequent use
        print(f"Loading best model from epoch {best_epoch+1}")
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        return history, best_model_path

    def classify(self, image):
        """Classify a single image"""
        # Convert image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))

        # Apply transformations
        image = self.test_transform(image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Get top prediction
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        confidence = probabilities[class_idx].item()
        class_name = self.train_dataset.class_names[class_idx]

        # Return prediction details
        return {
            "class": class_name,
            "confidence": confidence,
            "predictions": {self.train_dataset.class_names[i]: probabilities[i].item() for i in range(len(self.train_dataset.class_names))}
        }