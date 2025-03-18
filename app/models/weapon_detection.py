# app/models/scene_classification.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class SceneClassificationModel:
    def __init__(self, model_path=None, num_classes=3):
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            # Create a new model
            self.model = self._build_model(num_classes)
        
        self.class_names = ['Normal', 'Violence', 'Weaponized']
    
    def _build_model(self, num_classes):
        """Build a classification model based on MobileNetV2"""
        # Base model - MobileNetV2 is lightweight and efficient
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        # Final model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, data_dir, epochs=10, batch_size=32, fine_tune=True):
        """Train the classification model"""
        # Data generators with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        validation_generator = val_datagen.flow_from_directory(
            os.path.join(data_dir, 'val'),
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Initial training with frozen base model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    'models/scene_classification_best.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        # Fine-tuning (optional)
        if fine_tune:
            # Unfreeze some top layers of the base model
            for layer in self.model.layers[0].layers[-20:]:  # Unfreeze last 20 layers
                layer.trainable = True
            
            # Recompile with a lower learning rate
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Continue training with unfrozen layers
            history = self.model.fit(
                train_generator,
                epochs=epochs // 2,  # Fewer epochs for fine-tuning
                validation_data=validation_generator,
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(
                        'models/scene_classification_best_finetuned.h5',
                        save_best_only=True,
                        monitor='val_accuracy'
                    ),
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_accuracy',
                        patience=5,
                        restore_best_weights=True
                    )
                ]
            )
        
        # Save the final model
        self.model.save('models/scene_classification_final.h5')
        
        return history, 'models/scene_classification_final.h5'
    
    def classify(self, image):
        """Classify a single image"""
        # Preprocess the image
        img = tf.image.resize(image, (224, 224))
        img = img / 255.0  # Normalize
        img = tf.expand_dims(img, 0)  # Add batch dimension
        
        # Make prediction
        predictions = self.model.predict(img)
        class_idx = tf.argmax(predictions[0]).numpy()
        confidence = float(predictions[0][class_idx])
        class_name = self.class_names[class_idx]
        
        return {
            "class": class_name,
            "confidence": confidence,
            "predictions": {self.class_names[i]: float(predictions[0][i]) for i in range(len(self.class_names))}
        }