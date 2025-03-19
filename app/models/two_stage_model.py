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
import torch.nn.functional as F

class VideoDataset(Dataset):
    def __init__(self, video_data, transform=None, frame_count=8, split='train', temporal_features=True):
        """
        Args:
            video_data (dict): Dictionary of video paths returned by prepare_classification_dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            frame_count (int): Number of frames to extract per video.
            split (str): 'train', 'test', or 'val'.
            temporal_features (bool): Whether to return temporal features (sequence of frames) or averaged frames.
        """
        self.video_paths = []
        self.labels = []
        self.transform = transform
        self.frame_count = frame_count
        self.temporal_features = temporal_features

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
                
                if self.temporal_features:
                    # Return sequence of frames for temporal processing
                    # If we don't have enough frames, repeat the last frame
                    if stacked_frames.size(0) < self.frame_count:
                        last_frame = stacked_frames[-1].unsqueeze(0)
                        padding = last_frame.repeat(self.frame_count - stacked_frames.size(0), 1, 1, 1)
                        stacked_frames = torch.cat([stacked_frames, padding], dim=0)
                    # If we have too many frames, take the first frame_count frames
                    elif stacked_frames.size(0) > self.frame_count:
                        stacked_frames = stacked_frames[:self.frame_count]
                    
                    return stacked_frames, self.class_to_idx[label]
                else:
                    # Average the features across frames for non-temporal processing
                    avg_frame = torch.mean(stacked_frames, dim=0)
                    return avg_frame, self.class_to_idx[label]
            else:
                print(f"Warning: No processed frames from {video_path}")
                if self.temporal_features:
                    # Return zero tensor of shape [frame_count, channels, height, width]
                    return torch.zeros((self.frame_count, 3, 224, 224)), self.class_to_idx[label]
                else:
                    return torch.zeros((3, 224, 224)), self.class_to_idx[label]
        else:
            print(f"Warning: No frames extracted from {video_path}")
            if self.temporal_features:
                # Return zero tensor of shape [frame_count, channels, height, width]
                return torch.zeros((self.frame_count, 3, 224, 224)), self.class_to_idx[label]
            else:
                return torch.zeros((3, 224, 224)), self.class_to_idx[label]

    def _extract_frames(self, video_path, frame_count):
        """
        Extracts a specified number of evenly spaced frames from a video.
        Uses temporal information by capturing motion between frames.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            print(f"Error: Could not read frames from {video_path}")
            cap.release()
            return frames
        
        # Calculate frame indices for even spacing
        if frame_count > total_frames:
            frame_indices = list(range(total_frames))
        else:
            # Take evenly spaced frames but ensure we capture the beginning, middle, and end
            # This helps with detecting action/motion patterns specific to each class
            frame_indices = np.linspace(0, total_frames-1, frame_count, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                
        cap.release()
        return frames

class TemporalCNN(nn.Module):
    """A CNN model that processes temporal information from video frames."""
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(TemporalCNN, self).__init__()
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Remove the classifier
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # Feature dimension from efficientnet_b0
        feature_dim = 1280
        
        # Temporal feature processing
        self.temporal_conv = nn.Conv1d(in_channels=feature_dim, out_channels=512, kernel_size=3, padding=1)
        self.temporal_bn = nn.BatchNorm1d(512)
        self.temporal_pool = nn.AdaptiveMaxPool1d(1)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, num_frames, channels, height, width]
        batch_size, num_frames, c, h, w = x.size()
        
        # Reshape to process each frame
        x = x.view(batch_size * num_frames, c, h, w)
        
        # Extract features for each frame
        x = self.features(x)
        
        # Reshape back to [batch_size, num_frames, features]
        x = x.view(batch_size, num_frames, -1)
        
        # Transpose to [batch_size, features, num_frames] for 1D convolution
        x = x.transpose(1, 2)
        
        # Apply temporal convolution
        x = F.relu(self.temporal_bn(self.temporal_conv(x)))
        
        # Pool across time dimension
        x = self.temporal_pool(x).squeeze(-1)
        
        # Classifier
        x = self.classifier(x)
        
        return x

class TwoStageModel:
    def __init__(self, model_path=None, video_data=None, dropout_rate=0.6, 
                 frame_count=8, use_weighted_sampler=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.dropout_rate = dropout_rate
        self.frame_count = frame_count
        self.use_weighted_sampler = use_weighted_sampler
        
        # Define transforms with stronger augmentation for violence class
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # Increased vertical flip probability
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Increased color jitter
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(-20, 20)),  # Increased translate
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3)  # Increased erasing probability
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize datasets for both stages
        # For the first stage: Normal vs Not Normal (Violence + Weaponized)
        self._prepare_first_stage_data(video_data)
        
        # For the second stage: Violence vs Weaponized
        self._prepare_second_stage_data(video_data)
        
        # Initialize models
        self.first_stage_model = self._create_model(num_classes=2)  # Normal vs Not Normal
        self.second_stage_model = TemporalCNN(num_classes=2, dropout_rate=dropout_rate)  # Violence vs Weaponized
        
        # Move models to device
        self.first_stage_model = self.first_stage_model.to(self.device)
        self.second_stage_model = self.second_stage_model.to(self.device)
        
        # Load pre-trained weights if provided
        if model_path and isinstance(model_path, dict):
            if 'first_stage' in model_path and os.path.exists(model_path['first_stage']):
                self.first_stage_model.load_state_dict(torch.load(model_path['first_stage'], map_location=self.device))
                print(f"Loaded first stage weights from {model_path['first_stage']}")
            
            if 'second_stage' in model_path and os.path.exists(model_path['second_stage']):
                self.second_stage_model.load_state_dict(torch.load(model_path['second_stage'], map_location=self.device))
                print(f"Loaded second stage weights from {model_path['second_stage']}")

    def _prepare_first_stage_data(self, video_data):
        """Prepare data for the first stage: Normal vs Not Normal"""
        # Create a modified version of video_data for binary classification
        binary_video_data = {'train': {}, 'test': {}}
        
        # Normal class stays as is
        binary_video_data['train']['Normal'] = video_data['train']['Normal']
        binary_video_data['test']['Normal'] = video_data['test']['Normal']
        
        # Combine Violence and Weaponized into "Not Normal"
        binary_video_data['train']['Not_Normal'] = []
        binary_video_data['test']['Not_Normal'] = []
        
        for split in ['train', 'test']:
            for class_name in ['Violence', 'Weaponized']:
                if class_name in video_data[split]:
                    binary_video_data[split]['Not_Normal'].extend(video_data[split][class_name])
        
        # Initialize datasets
        self.first_stage_train_dataset = VideoDataset(
            binary_video_data, transform=self.train_transform, 
            frame_count=self.frame_count, split='train', temporal_features=False
        )
        
        self.first_stage_test_dataset = VideoDataset(
            binary_video_data, transform=self.test_transform, 
            frame_count=self.frame_count, split='test', temporal_features=False
        )
        
        # Create weighted sampler for training data
        if self.use_weighted_sampler:
            class_counts = Counter([self.first_stage_train_dataset.class_to_idx[label] 
                                   for label in self.first_stage_train_dataset.labels])
            class_weights = {class_idx: 1.0 / count for class_idx, count in class_counts.items()}
            sample_weights = [class_weights[self.first_stage_train_dataset.class_to_idx[label]] 
                             for label in self.first_stage_train_dataset.labels]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        # Initialize DataLoaders
        self.first_stage_train_loader = DataLoader(
            self.first_stage_train_dataset, batch_size=32, shuffle=shuffle,
            sampler=sampler, num_workers=4, pin_memory=True
        )
        
        self.first_stage_test_loader = DataLoader(
            self.first_stage_test_dataset, batch_size=32, shuffle=False,
            num_workers=4, pin_memory=True
        )

    def _prepare_second_stage_data(self, video_data):
        """Prepare data for the second stage: Violence vs Weaponized"""
        # Create a filtered version of video_data for Violence vs Weaponized
        filtered_video_data = {'train': {}, 'test': {}}
        
        # Only include Violence and Weaponized classes
        for split in ['train', 'test']:
            for class_name in ['Violence', 'Weaponized']:
                if class_name in video_data[split]:
                    filtered_video_data[split][class_name] = video_data[split][class_name]
        
        # Initialize datasets with temporal features
        self.second_stage_train_dataset = VideoDataset(
            filtered_video_data, transform=self.train_transform, 
            frame_count=self.frame_count, split='train', temporal_features=True
        )
        
        self.second_stage_test_dataset = VideoDataset(
            filtered_video_data, transform=self.test_transform, 
            frame_count=self.frame_count, split='test', temporal_features=True
        )
        
        # Create weighted sampler for training data
        if self.use_weighted_sampler:
            class_counts = Counter([self.second_stage_train_dataset.class_to_idx[label] 
                                   for label in self.second_stage_train_dataset.labels])
            class_weights = {class_idx: 1.0 / count for class_idx, count in class_counts.items()}
            sample_weights = [class_weights[self.second_stage_train_dataset.class_to_idx[label]] 
                             for label in self.second_stage_train_dataset.labels]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        # Initialize DataLoaders
        self.second_stage_train_loader = DataLoader(
            self.second_stage_train_dataset, batch_size=16, shuffle=shuffle,  # Smaller batch size for temporal
            sampler=sampler, num_workers=4, pin_memory=True
        )
        
        self.second_stage_test_loader = DataLoader(
            self.second_stage_test_dataset, batch_size=16, shuffle=False,  # Smaller batch size for temporal
            num_workers=4, pin_memory=True
        )

    def _create_model(self, num_classes):
        """Create a model for the first stage"""
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )
        return model

    def train_first_stage(self, epochs=30, learning_rate=0.0003, weight_decay=2e-4, patience=10, 
                         label_smoothing=0.1):
        """Train the first stage model (Normal vs Not Normal)"""
        print("Training first stage model: Normal vs Not Normal")
        
        # Compute class weights for loss function
        y_train = np.array([self.first_stage_train_dataset.class_to_idx[label] 
                           for label in self.first_stage_train_dataset.labels])
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        print(f"First stage class weights: {class_weights}")
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        optimizer = optim.AdamW(self.first_stage_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
    )
        
        # Initialize tracking variables
        best_val_acc = 0.0
        no_improve_counter = 0
        best_model_path = os.path.join('models', 'first_stage_best.pth')
        
        # Make sure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rates': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            # Training phase
            self.first_stage_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, labels) in enumerate(self.first_stage_train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.first_stage_model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.first_stage_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    print(f"First Stage - Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(self.first_stage_train_loader)} | "
                          f"Loss: {loss.item():.4f}")
            
            # Calculate average training loss and accuracy
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            self.first_stage_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in self.first_stage_test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.first_stage_model(inputs)
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
            
            # Check if this is the best model
            if val_acc > best_val_acc:
                print(f"First Stage - Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}")
                best_val_acc = val_acc
                no_improve_counter = 0
                torch.save(self.first_stage_model.state_dict(), best_model_path)
                print(f"Saved best first stage model at epoch {epoch+1}")
            else:
                no_improve_counter += 1
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print statistics
            print(f"First Stage - Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"LR: {current_lr:.6f}")
            
            # Early stopping check
            if no_improve_counter >= patience:
                print(f"First Stage - Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load the best model
        print(f"Loading best first stage model")
        self.first_stage_model.load_state_dict(torch.load(best_model_path))
        self.first_stage_model.eval()
        
        return history, best_model_path

    def train_second_stage(self, epochs=40, learning_rate=0.0002, weight_decay=2e-4, patience=12, 
                          label_smoothing=0.1):
        """Train the second stage model (Violence vs Weaponized)"""
        print("Training second stage model: Violence vs Weaponized")
        
        # Compute class weights for loss function
        y_train = np.array([self.second_stage_train_dataset.class_to_idx[label] 
                           for label in self.second_stage_train_dataset.labels])
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        print(f"Second stage class weights: {class_weights}")
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        optimizer = optim.AdamW(self.second_stage_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=learning_rate/20
    )
        
        # Initialize tracking variables
        best_val_acc = 0.0
        no_improve_counter = 0
        best_model_path = os.path.join('models', 'second_stage_best.pth')
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rates': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            # Training phase
            self.second_stage_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, labels) in enumerate(self.second_stage_train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.second_stage_model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.second_stage_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Print progress every 5 batches (since batch size is smaller)
                if (batch_idx + 1) % 5 == 0:
                    print(f"Second Stage - Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(self.second_stage_train_loader)} | "
                          f"Loss: {loss.item():.4f}")
            
            # Calculate average training loss and accuracy
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            self.second_stage_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in self.second_stage_test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.second_stage_model(inputs)
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
            scheduler.step()
            
            # Check if this is the best model
            if val_acc > best_val_acc:
                print(f"Second Stage - Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}")
                best_val_acc = val_acc
                no_improve_counter = 0
                torch.save(self.second_stage_model.state_dict(), best_model_path)
                print(f"Saved best second stage model at epoch {epoch+1}")
            else:
                no_improve_counter += 1
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print statistics
            print(f"Second Stage - Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"LR: {current_lr:.6f}")
            
            # Early stopping check
            if no_improve_counter >= patience:
                print(f"Second Stage - Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load the best model
        print(f"Loading best second stage model")
        self.second_stage_model.load_state_dict(torch.load(best_model_path))
        self.second_stage_model.eval()
        
        return history, best_model_path

    def train(self, epochs_first=30, epochs_second=40, learning_rate_first=0.0003, 
              learning_rate_second=0.0002, weight_decay=2e-4, patience_first=10, 
              patience_second=12, label_smoothing=0.1):
        """Train both stages of the model"""
        # Train first stage
        first_stage_history, first_stage_path = self.train_first_stage(
            epochs=epochs_first, learning_rate=learning_rate_first, 
            weight_decay=weight_decay, patience=patience_first, 
            label_smoothing=label_smoothing
        )
        
        # Train second stage
        second_stage_history, second_stage_path = self.train_second_stage(
            epochs=epochs_second, learning_rate=learning_rate_second, 
            weight_decay=weight_decay, patience=patience_second, 
            label_smoothing=label_smoothing
        )
        
        # Return combined history and paths
        combined_history = {
            'first_stage': first_stage_history,
            'second_stage': second_stage_history
        }
        
        model_paths = {
            'first_stage': first_stage_path,
            'second_stage': second_stage_path
        }
        
        return combined_history, model_paths

    def _extract_frames_for_prediction(self, video_path):
        """Extract frames from a video for prediction"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            print(f"Error: Could not read frames from {video_path}")
            cap.release()
            return []
        
        # Extract frames for prediction
        frame_indices = np.linspace(0, total_frames-1, self.frame_count, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Apply transform
                frame = self.test_transform(frame)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def _predict_first_stage(self, frames):
        """Make a prediction with the first stage model"""
        if not frames:
            return 0  # Default to Normal if no frames
        
        # Convert frames to batch
        if len(frames) > 0:
            # Stack frames
            stacked_frames = torch.stack(frames)
            # Average frames for first stage model
            avg_frame = torch.mean(stacked_frames, dim=0).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.first_stage_model(avg_frame)
                _, predicted = torch.max(outputs, 1)
                
            return predicted.item()
        else:
            return 0  # Default to Normal if no valid frames
    
    def _predict_second_stage(self, frames):
        """Make a prediction with the second stage model"""
        if not frames:
            return 1  # Default to Weaponized if no frames (safer assumption)
        
        # Convert frames to batch
        if len(frames) > 0:
            # Stack frames and ensure we have enough
            stacked_frames = torch.stack(frames)
            
            # Ensure we have frame_count frames
            if stacked_frames.size(0) < self.frame_count:
                # Repeat the last frame if needed
                last_frame = stacked_frames[-1].unsqueeze(0)
                padding = last_frame.repeat(self.frame_count - stacked_frames.size(0), 1, 1, 1)
                stacked_frames = torch.cat([stacked_frames, padding], dim=0)
            elif stacked_frames.size(0) > self.frame_count:
                stacked_frames = stacked_frames[:self.frame_count]
            
            # Add batch dimension
            batched_frames = stacked_frames.unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.second_stage_model(batched_frames)
                _, predicted = torch.max(outputs, 1)
                
            return predicted.item()
        else:
            return 1  # Default to Weaponized if no valid frames
    
    def evaluate(self, verbose=True):
        """Evaluate the two-stage model on the test set"""
        self.first_stage_model.eval()
        self.second_stage_model.eval()
        
        # Original class names and mapping for final predictions
        original_classes = ['Normal', 'Violence', 'Weaponized']
        class_mapping = {
            (0, None): 0,  # Normal
            (1, 0): 1,     # Not Normal -> Violence
            (1, 1): 2      # Not Normal -> Weaponized
        }
        
        # Get all test data from the original datasets
        all_video_paths = []
        all_true_labels = []
        
        # Collect paths and true labels from first stage test dataset
        for i in range(len(self.first_stage_test_dataset)):
            path = self.first_stage_test_dataset.video_paths[i]
            label = self.first_stage_test_dataset.labels[i]
            
            # Map binary labels back to original classes
            if label == 'Normal':
                true_label = 0  # Normal
            else:  # Not_Normal
                # Find the original label in the source data
                if 'Violence' in path.lower() or path in self.second_stage_test_dataset.video_paths and \
                   self.second_stage_test_dataset.labels[self.second_stage_test_dataset.video_paths.index(path)] == 'Violence':
                    true_label = 1  # Violence
                else:
                    true_label = 2  # Weaponized
            
            all_video_paths.append(path)
            all_true_labels.append(true_label)
        
        # Create confusion matrix
        confusion_matrix = torch.zeros(3, 3, device=self.device)
        
        # Track all predictions
        all_predictions = []
        
        # Process each video
        for i, path in enumerate(all_video_paths):
            true_label = all_true_labels[i]
            
            # First stage prediction (Normal vs Not Normal)
            # Load and preprocess the image
            frames = self._extract_frames_for_prediction(path)
            
            # Make first stage prediction
            first_stage_pred = self._predict_first_stage(frames)
            
            # If Normal, we're done
            if first_stage_pred == 0:
                final_pred = 0
            else:
                # If Not Normal, use second stage to distinguish between Violence and Weaponized
                second_stage_pred = self._predict_second_stage(frames)
                
                # Map to final prediction
                final_pred = class_mapping[(1, second_stage_pred)]
            
            # Update confusion matrix
            confusion_matrix[true_label, final_pred] += 1
            all_predictions.append(final_pred)
        
        # Calculate accuracy
        correct = (confusion_matrix.diag().sum() / confusion_matrix.sum()).item()
        
        if verbose:
            print(f"Overall Accuracy: {correct:.4f}")
            
            # Calculate per-class metrics
            print("\nPer-Class Metrics:")
            
            for i, class_name in enumerate(original_classes):
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
            print("            " + " ".join(f"{name:>10}" for name in original_classes))
            print("Actual ↓")
            
            for i, name in enumerate(original_classes):
                row = f"{name:10}" + " ".join([f"{int(confusion_matrix[i, j].item()):10d}" for j in range(3)])
                print(row)
        
        # Return evaluation metrics
        return {
            'accuracy': correct,
            'confusion_matrix': confusion_matrix.cpu().numpy(),
            'predictions': all_predictions,
            'true_labels': all_true_labels
        }
        
    def classify(self, video_path):
        """Classify a single video"""
        original_classes = ['Normal', 'Violence', 'Weaponized']
        class_mapping = {
            (0, None): 0,  # Normal
            (1, 0): 1,     # Not Normal -> Violence
            (1, 1): 2      # Not Normal -> Weaponized
        }
        
        # Extract frames
        frames = self._extract_frames_for_prediction(video_path)
        
        # First stage prediction
        first_stage_pred = self._predict_first_stage(frames)
        
        # If Normal, we're done
        if first_stage_pred == 0:
            final_pred = 0
            final_class = original_classes[final_pred]
            confidence = self._get_first_stage_confidence(frames, first_stage_pred)
            return {
                "class": final_class,
                "confidence": confidence,
                "predictions": {
                    original_classes[0]: confidence,
                    original_classes[1]: 0.0,
                    original_classes[2]: 0.0
                }
            }
        else:
            # If Not Normal, use second stage to distinguish between Violence and Weaponized
            second_stage_pred = self._predict_second_stage(frames)
            
            # Map to final prediction
            final_pred = class_mapping[(1, second_stage_pred)]
            final_class = original_classes[final_pred]
            
            # Get confidences
            first_stage_confidence = self._get_first_stage_confidence(frames, first_stage_pred)
            second_stage_confidence = self._get_second_stage_confidence(frames, second_stage_pred)
            
            # Calculate combined confidence
            confidence = first_stage_confidence * second_stage_confidence
            
            # Calculate confidences for all classes
            if second_stage_pred == 0:  # Violence
                violence_confidence = confidence
                weaponized_confidence = first_stage_confidence * (1 - second_stage_confidence)
            else:  # Weaponized
                violence_confidence = first_stage_confidence * (1 - second_stage_confidence)
                weaponized_confidence = confidence
            
            normal_confidence = 1 - first_stage_confidence
            
            return {
                "class": final_class,
                "confidence": confidence,
                "predictions": {
                    original_classes[0]: normal_confidence,
                    original_classes[1]: violence_confidence,
                    original_classes[2]: weaponized_confidence
                }
            }
    
    def _get_first_stage_confidence(self, frames, predicted_class):
        """Get the confidence of the first stage prediction"""
        if not frames:
            return 0.5  # Default confidence if no frames
        
        # Convert frames to batch
        if len(frames) > 0:
            # Stack frames
            stacked_frames = torch.stack(frames)
            # Average frames for first stage model
            avg_frame = torch.mean(stacked_frames, dim=0).unsqueeze(0).to(self.device)
            
            # Get confidences
            with torch.no_grad():
                outputs = self.first_stage_model(avg_frame)
                probabilities = F.softmax(outputs, dim=1)[0]
                
            return probabilities[predicted_class].item()
        else:
            return 0.5  # Default confidence if no valid frames
    
    def _get_second_stage_confidence(self, frames, predicted_class):
        """Get the confidence of the second stage prediction"""
        if not frames:
            return 0.5  # Default confidence if no frames
        
        # Convert frames to batch
        if len(frames) > 0:
            # Stack frames and ensure we have enough
            stacked_frames = torch.stack(frames)
            
            # Ensure we have frame_count frames
            if stacked_frames.size(0) < self.frame_count:
                # Repeat the last frame if needed
                last_frame = stacked_frames[-1].unsqueeze(0)
                padding = last_frame.repeat(self.frame_count - stacked_frames.size(0), 1, 1, 1)
                stacked_frames = torch.cat([stacked_frames, padding], dim=0)
            elif stacked_frames.size(0) > self.frame_count:
                stacked_frames = stacked_frames[:self.frame_count]
            
            # Add batch dimension
            batched_frames = stacked_frames.unsqueeze(0).to(self.device)
            
            # Get confidences
            with torch.no_grad():
                outputs = self.second_stage_model(batched_frames)
                probabilities = F.softmax(outputs, dim=1)[0]
                
            return probabilities[predicted_class].item()
        else:
            return 0.5  # Default confidence if no valid frames