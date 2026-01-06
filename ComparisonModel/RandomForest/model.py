import numpy as np
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    def __init__(self, in_channels=77, historical_channels=5):
        self.in_channels = in_channels
        self.historical_channels = historical_channels
        self.temporal_channels = in_channels - historical_channels
        
    def extract(self, image):
        C, H, W = image.shape
        features = image.reshape(C, -1).T.astype(np.float32)
        return features
    
    def get_feature_dim(self):
        return self.in_channels


class RandomForestModel(nn.Module):
    def __init__(
        self,
        in_channels=77,
        num_classes=6,
        img_size=128,
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        **kwargs
    ):
        super(RandomForestModel, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        
        self.feature_extractor = FeatureExtractor(
            in_channels=in_channels,
            historical_channels=5
        )
        
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt',
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=0
        )
        
        self.scaler = StandardScaler()
        
        self.is_fitted = False
        
        print(f"\nRandom Forest Model:")
        print(f"  Input: ({in_channels}, {img_size}, {img_size})")
        print(f"  Feature dimension: {self.feature_extractor.get_feature_dim()} (raw pixel values)")
        print(f"  Feature composition: {self.feature_extractor.temporal_channels} spectral + {self.feature_extractor.historical_channels} historical channels")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        print(f"  class_weight: {class_weight}")
    
    def forward(self, x):
        raise NotImplementedError(
            "Random Forest does not support forward pass during training. "
            "Use fit() and predict() methods instead."
        )
    
    def fit(self, train_loader, device='cpu'):
        print("\nTraining Random Forest...")
        print("="*80)
        
        X_train = []
        y_train = []
        
        print("Extracting features from training data...")
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 4:
                images, labels, _, _ = batch
            else:
                images, labels = batch
            
            images = images.numpy()
            labels = labels.numpy()
            
            batch_size = images.shape[0]
            
            for i in range(batch_size):
                img = images[i]
                lbl = labels[i]
                
                features = self.feature_extractor.extract(img)
                
                if len(lbl.shape) == 2:
                    lbl_flat = lbl.flatten()
                else:
                    lbl_flat = np.full(features.shape[0], lbl)
                
                X_train.append(features)
                y_train.append(lbl_flat)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(train_loader)} batches")
        
        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)
        
        print(f"\nTotal training samples: {X_train.shape[0]:,}")
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Class distribution: {np.bincount(y_train.astype(int))}")
        
        print("\nFitting StandardScaler...")
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        print("\nTraining Random Forest classifier...")
        self.rf.fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        
        print("\nTraining completed!")
        print("="*80)
        
        return self
    
    def predict(self, x):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        batch_size, C, H, W = x.shape
        predictions = np.zeros((batch_size, H, W), dtype=np.int64)
        
        for i in range(batch_size):
            img = x[i]
            
            features = self.feature_extractor.extract(img)
            
            features_scaled = self.scaler.transform(features)
            
            pred_flat = self.rf.predict(features_scaled)
            
            pred = pred_flat.reshape(H, W)
            predictions[i] = pred
        
        return predictions
    
    def predict_proba(self, x):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        batch_size, C, H, W = x.shape
        probabilities = np.zeros((batch_size, self.num_classes, H, W), dtype=np.float32)
        
        for i in range(batch_size):
            img = x[i]
            
            features = self.feature_extractor.extract(img)
            
            features_scaled = self.scaler.transform(features)
            
            proba_flat = self.rf.predict_proba(features_scaled)
            
            for c in range(self.num_classes):
                probabilities[i, c] = proba_flat[:, c].reshape(H, W)
        
        return probabilities
    
    def get_feature_importances(self):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first. Call fit() first.")
        
        return self.rf.feature_importances_
    
    def save(self, path):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving. Call fit() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'rf': self.rf,
            'scaler': self.scaler,
            'in_channels': self.in_channels,
            'num_classes': self.num_classes,
            'img_size': self.img_size,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {path}")
    
    def load(self, path):
        path = Path(path)
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.rf = model_data['rf']
        self.scaler = model_data['scaler']
        self.in_channels = model_data['in_channels']
        self.num_classes = model_data['num_classes']
        self.img_size = model_data['img_size']
        self.is_fitted = True
        
        print(f"Model loaded from {path}")


def build_random_forest(config):
    model = RandomForestModel(**config)
    return model


if __name__ == "__main__":
    print("Testing Random Forest model...")
    
    config = {
        'in_channels': 77,
        'num_classes': 6,
        'img_size': 128,
        'n_estimators': 100,
        'max_depth': 15,
    }
    
    model = build_random_forest(config)
    print("\nModel created successfully!")
    
    test_image = np.random.rand(77, 128, 128).astype(np.float32)
    features = model.feature_extractor.extract(test_image)
    print(f"\nFeature extraction test:")
    print(f"  Input shape: {test_image.shape}")
    print(f"  Output shape: {features.shape}")
    print(f"  Feature dimension: {model.feature_extractor.get_feature_dim()}")
