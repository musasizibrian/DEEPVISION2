import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
import os
import numpy as np
import cv2
from PIL import Image
from sklearn.utils import class_weight
from collections import Counter

class VideoDataset(Dataset):
    def __init__(self, video_data, transform=None, frame_count=5, split='train'):
        """
        Args:
            video_data (dict): Dictionary of video paths returned by prepare_classification_dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            frame_count (int): Number of frames to extract per video.
            split (str): 'train', 'test', or 'val'.
        """
        self.video_paths = []
        self.labels = []
        self.transform = transform
        self.frame_count = frame_count

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
        
        # Print class distribution
        label_counts = Counter(self.labels)
        print(f"Class distribution in {split} set:")
        for class_name, count in label_counts.items():
            print(f"  {class_name}: {count}")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self._extract_frames(video_path, self.frame_count)

        if len(frames) > 0:
            # If we have frames, process them
            processed_frames = []
            for frame in frames:
                # Convert to RGB (OpenCV loads as BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                if self.transform:
                    frame = self.transform(frame)
                processed_frames.append(frame)
            
            # Stack frames along a new dimension
            if len(processed_frames) > 0:
                stacked_frames = torch.stack(processed_frames)
                # Average the features across frames
                avg_frame = torch.mean(stacked_frames, dim=0)
                return avg_frame, self.class_to_idx[label]
            else:
                print(f"Warning: No processed frames from {video_path}")
                return torch.zeros((3, 224, 224)), self.class_to_idx[label]  # Placeholder
        else:
            print(f"Warning: No frames extracted from {video_path}")
            return torch.zeros((3, 224, 224)), self.class_to_idx[label]  # Placeholder

    def _extract_frames(self, video_path, frame_count):
        """
        Extracts a specified number of evenly spaced frames from a video.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            print(f"Error: Could not read frames from {video_path}")
            cap.release()
            return frames
        
        # Calculate frame intervals for even spacing
        if frame_count > total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames-1, frame_count, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                
        cap.release()
        return frames

class SceneClassificationModel:
    def __init__(self, model_path=None, num_classes=3, video_data=None, dropout_rate=0.5, 
                 model_type='efficientnet', frame_count=5, use_weighted_sampler=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model_type = model_type
        self.frame_count = frame_count
        self.use_weighted_sampler = use_weighted_sampler

        # Define transforms with stronger augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize larger, then crop
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),  # Added vertical flip
            transforms.RandomRotation(degrees=30),  # Increased rotation
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-15, 15)),
            transforms.RandomGrayscale(p=0.1),  # Occasionally convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)  # Random erasing for robustness
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize Datasets
        self.train_dataset = VideoDataset(video_data, transform=self.train_transform, 
                                         frame_count=frame_count, split='train')
        self.test_dataset = VideoDataset(video_data, transform=self.test_transform, 
                                        frame_count=frame_count, split='test')

        # Create weighted sampler for training data to handle class imbalance
        if use_weighted_sampler:
            class_counts = Counter([self.train_dataset.class_to_idx[label] for label in self.train_dataset.labels])
            class_weights = {class_idx: 1.0 / count for class_idx, count in class_counts.items()}
            sample_weights = [class_weights[self.train_dataset.class_to_idx[label]] for label in self.train_dataset.labels]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle = False  # Don't shuffle when using sampler
        else:
            sampler = None
            shuffle = True

        # Initialize DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=32, 
            shuffle=shuffle,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )

        # Model initialization based on model_type
        if model_type == 'mobilenet':
            self.model = models.mobilenet_v2(pretrained=True)
            # Modify classifier with dropout
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.model.last_channel, num_classes)
            )
        elif model_type == 'resnet':
            self.model = models.resnet50(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_ftrs, num_classes)
            )
        elif model_type == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=True)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Print model summary
        print(f"Initialized {model_type} model with {dropout_rate} dropout rate")
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded pre-trained weights from {model_path}")
            self.model.eval()

    def train(self, epochs=50, batch_size=32, learning_rate=0.0005, weight_decay=1e-4, patience=15, 
              mixup_alpha=0.2, label_smoothing=0.1):
        """
        Train the model with enhanced training techniques including:
        - Learning rate scheduling
        - Mixup augmentation
        - Label smoothing
        - Gradient clipping
        """
        print(f"Starting training for {epochs} epochs with LR={learning_rate}, weight_decay={weight_decay}")
        
        # Compute class weights for loss function
        y_train = np.array([self.train_dataset.class_to_idx[label] for label in self.train_dataset.labels])
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        print(f"Class weights for loss function: {class_weights}")

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler - cosine annealing with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=learning_rate/20
        )

        # Initialize tracking variables
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_model_path = os.path.join('models', f'scene_classification_{self.model_type}_best.pth')

        # Make sure models directory exists
        os.makedirs('models', exist_ok=True)

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        # Early stopping variables
        no_improve_counter = 0
        best_epoch = 0

        # Helper function for mixup
        def mixup_data(x, y, alpha=0.2):
            """Applies mixup augmentation to the batch."""
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1

            batch_size = x.size()[0]
            index = torch.randperm(batch_size).to(self.device)

            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam

        def mixup_criterion(criterion, pred, y_a, y_b, lam):
            """Calculates loss for mixup."""
            return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

        # Training loop
        for epoch in range(epochs):
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Training loop with progress reporting
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Apply mixup with 50% probability
                use_mixup = (np.random.random() < 0.5) and (mixup_alpha > 0)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass with or without mixup
                if use_mixup:
                    inputs_mixed, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
                    outputs = self.model(inputs_mixed)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    
                    # For tracking accuracy with mixup, we'll use the primary label
                    _, predicted = torch.max(outputs, 1)
                    train_correct += (lam * (predicted == labels_a).sum().item() + 
                                     (1 - lam) * (predicted == labels_b).sum().item())
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Track statistics
                    _, predicted = torch.max(outputs, 1)
                    train_correct += (predicted == labels).sum().item()
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Track statistics
                train_loss += loss.item() * inputs.size(0)
                train_total += labels.size(0)
                
                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(self.train_loader)} | "
                          f"Loss: {loss.item():.4f}")

            # Calculate average training loss and accuracy
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # Initialize confusion matrix for validation
            confusion_matrix = torch.zeros(self.num_classes, self.num_classes, device=self.device)
            
            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                    # Track statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Update confusion matrix
                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

            # Calculate average validation loss and accuracy
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            # Update scheduler
            scheduler.step()

            # Check if this is the best model by validation accuracy
            if val_acc > best_val_acc:
                print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}")
                best_val_acc = val_acc
                best_epoch = epoch
                no_improve_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Saved best model at epoch {epoch+1}")
                
                # Print confusion matrix for best model
                print("\nConfusion Matrix for Best Model:")
                class_names = self.train_dataset.class_names
                print("Predicted →")
                print("            " + " ".join(f"{name:>10}" for name in class_names))
                print("Actual ↓")
                for i, name in enumerate(class_names):
                    row = f"{name:10}" + " ".join(f"{confusion_matrix[i, j].int().item():10d}" for j in range(self.num_classes))
                    print(row)
            else:
                no_improve_counter += 1

            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Print statistics
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"LR: {current_lr:.6f}")

            # Early Stopping Check
            if no_improve_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load the best model for subsequent use
        print(f"Loading best model from epoch {best_epoch+1}")
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        return history, best_model_path

    def evaluate(self, verbose=True):
        """Comprehensive model evaluation function"""
        self.model.eval()
        device = self.device
        
        # Initialize variables for metrics
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        # Use CrossEntropyLoss without class weights for evaluation
        criterion = nn.CrossEntropyLoss()
        
        # Create confusion matrix
        confusion_matrix = torch.zeros(self.num_classes, self.num_classes, device=device)
        
        # Store all predictions and true labels
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        
        # Evaluate with no gradient
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Get probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                
                # Update metrics
                test_loss += loss.item() * inputs.size(0)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                # Update confusion matrix
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                
                # Store predictions and true labels
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate overall metrics
        accuracy = test_correct / test_total
        test_loss = test_loss / test_total
        
        if verbose:
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")
            
            # Calculate per-class metrics
            print("\nPer-Class Metrics:")
            class_names = self.train_dataset.class_names
            
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
                row = f"{name:10}" + " ".join(f"{confusion_matrix[i, j].int().item():10d}" for j in range(self.num_classes))
                print(row)
        
        # Return evaluation metrics
        return {
            'accuracy': accuracy,
            'loss': test_loss,
            'confusion_matrix': confusion_matrix.cpu().numpy(),
            'predictions': all_predictions,
            'true_labels': all_true_labels,
            'probabilities': all_probabilities
        }

    def classify(self, image):
        """Classify a single image"""
        # Convert image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))

        # Apply transformations
        image = self.test_transform(image).unsqueeze(0).to(self.device)

        # Make prediction
        self.model.eval()
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