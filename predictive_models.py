import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib

class SolarPVPredictiveModels:
    def __init__(self, data_path='.'):
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
    def load_processed_data(self):
        """Load preprocessed training and test data"""
        try:
            self.train_df = pd.read_csv(f'{self.data_path}train_data.csv')
            self.test_df = pd.read_csv(f'{self.data_path}test_data.csv')
            
            # Feature columns (exclude target and metadata)
            self.feature_cols = [col for col in self.train_df.columns if col not in [
                'timestamp', 'inverter_id', 'inverter_status', 'fault_code',
                'will_fail', 'failure_type', 'days_until_failure'
            ]]
            
            # Prepare features and targets
            self.X_train = self.train_df[self.feature_cols].fillna(0)
            self.X_test = self.test_df[self.feature_cols].fillna(0)
            
            self.y_binary_train = self.train_df['will_fail']
            self.y_binary_test = self.test_df['will_fail']
            
            # Encode failure types for multi-class
            le = LabelEncoder()
            self.y_multiclass_train = le.fit_transform(self.train_df['failure_type'])
            self.y_multiclass_test = le.transform(self.test_df['failure_type'])
            self.label_encoders['failure_type'] = le
            
            self.y_regression_train = self.train_df['days_until_failure']
            self.y_regression_test = self.test_df['days_until_failure']
            
            print(f"Loaded data: {len(self.X_train)} train, {len(self.X_test)} test samples")
            print(f"Features: {len(self.feature_cols)}")
            print(f"Failure rate: {self.y_binary_train.mean():.3f}")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def build_binary_classification_models(self):
        """Build models for binary failure prediction"""
        print("Building binary classification models...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        self.scalers['binary'] = scaler
        
        # Model 1: Random Forest (handles non-linear patterns)
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42
        )
        rf_model.fit(self.X_train, self.y_binary_train)
        self.models['rf_binary'] = rf_model
        
        # Model 2: XGBoost (gradient boosting)
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=len(self.y_binary_train[self.y_binary_train==0]) / len(self.y_binary_train[self.y_binary_train==1]),
            random_state=42
        )
        xgb_model.fit(self.X_train, self.y_binary_train)
        self.models['xgb_binary'] = xgb_model
        
        # Model 3: Logistic Regression (baseline)
        lr_model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        lr_model.fit(X_train_scaled, self.y_binary_train)
        self.models['lr_binary'] = lr_model
        
        print("Binary classification models trained!")
    
    def build_anomaly_detection_model(self):
        """Build anomaly detection model for early warning"""
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        
        print("Building anomaly detection models...")
        
        # Use only normal operation data for training
        normal_data = self.X_train[self.y_binary_train == 0]
        
        # Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42
        )
        iso_forest.fit(normal_data)
        self.models['isolation_forest'] = iso_forest
        
        # One-Class SVM
        scaler_anomaly = StandardScaler()
        normal_data_scaled = scaler_anomaly.fit_transform(normal_data)
        self.scalers['anomaly'] = scaler_anomaly
        
        oc_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
        oc_svm.fit(normal_data_scaled)
        self.models['one_class_svm'] = oc_svm
        
        print("Anomaly detection models trained!")
    
    def build_time_series_model(self):
        """Build LSTM model for time series prediction"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            print("Building LSTM time series model...")
            
            # Prepare time series data (sequences of 24 hours = 288 intervals)
            sequence_length = 288  # 24 hours of 5-min intervals
            
            def create_sequences(data, target, seq_length):
                X, y = [], []
                for i in range(seq_length, len(data)):
                    X.append(data[i-seq_length:i])
                    y.append(target[i])
                return np.array(X), np.array(y)
            
            # Scale data
            scaler_lstm = StandardScaler()
            X_train_scaled = scaler_lstm.fit_transform(self.X_train)
            X_test_scaled = scaler_lstm.transform(self.X_test)
            self.scalers['lstm'] = scaler_lstm
            
            # Create sequences
            X_train_seq, y_train_seq = create_sequences(
                X_train_scaled, self.y_binary_train.values, sequence_length
            )
            X_test_seq, y_test_seq = create_sequences(
                X_test_scaled, self.y_binary_test.values, sequence_length
            )
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, len(self.feature_cols))),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Train model
            history = model.fit(
                X_train_seq, y_train_seq,
                batch_size=32,
                epochs=10,
                validation_split=0.2,
                verbose=0
            )
            
            self.models['lstm'] = model
            self.lstm_sequences = {
                'X_train': X_train_seq,
                'y_train': y_train_seq,
                'X_test': X_test_seq,
                'y_test': y_test_seq
            }
            
            print("LSTM model trained!")
            
        except ImportError:
            print("TensorFlow not available, skipping LSTM model")
    
    def build_rule_based_baseline(self):
        """Build rule-based baseline model"""
        print("Building rule-based baseline...")
        
        def rule_based_prediction(X):
            """Simple rule-based failure prediction"""
            predictions = []
            
            for _, row in X.iterrows():
                score = 0
                
                # High temperature rule
                if row.get('module_temp_c', 0) > 70:
                    score += 0.3
                
                # Low performance rule
                if row.get('performance_ratio', 1) < 0.7:
                    score += 0.2
                
                # Recent fault activity
                if row.get('fault_count_30d', 0) > 2:
                    score += 0.3
                
                # Long time since maintenance
                if row.get('days_since_maintenance', 0) > 180:
                    score += 0.2
                
                # Long time since cleaning
                if row.get('days_since_cleaning', 0) > 90:
                    score += 0.1
                
                # High temperature deviation
                if row.get('temp_deviation', 0) > 40:
                    score += 0.2
                
                predictions.append(score > 0.5)
            
            return np.array(predictions)
        
        self.models['rule_based'] = rule_based_prediction
        print("Rule-based baseline created!")
    
    def evaluate_models(self):
        """Evaluate all models and compare performance"""
        print("\n=== MODEL EVALUATION ===")
        
        results = {}
        
        # Binary classification models
        for model_name in ['rf_binary', 'xgb_binary', 'lr_binary']:
            if model_name in self.models:
                model = self.models[model_name]
                
                if model_name == 'lr_binary':
                    # Use scaled data for logistic regression
                    X_test_eval = self.scalers['binary'].transform(self.X_test)
                else:
                    X_test_eval = self.X_test
                
                y_pred = model.predict(X_test_eval)
                y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
                
                auc_score = roc_auc_score(self.y_binary_test, y_pred_proba)
                
                results[model_name] = {
                    'accuracy': (y_pred == self.y_binary_test).mean(),
                    'auc': auc_score,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"\n{model_name.upper()}:")
                print(f"Accuracy: {results[model_name]['accuracy']:.3f}")
                print(f"AUC: {results[model_name]['auc']:.3f}")
                print(classification_report(self.y_binary_test, y_pred, target_names=['No Failure', 'Failure']))
        
        # Anomaly detection models
        for model_name in ['isolation_forest', 'one_class_svm']:
            if model_name in self.models:
                model = self.models[model_name]
                
                if model_name == 'one_class_svm':
                    X_test_eval = self.scalers['anomaly'].transform(self.X_test)
                else:
                    X_test_eval = self.X_test
                
                anomaly_pred = model.predict(X_test_eval)
                # Convert to binary (1 = normal, -1 = anomaly -> 0 = normal, 1 = anomaly)
                y_pred = (anomaly_pred == -1).astype(int)
                
                results[model_name] = {
                    'accuracy': (y_pred == self.y_binary_test).mean(),
                    'predictions': y_pred
                }
                
                print(f"\n{model_name.upper()}:")
                print(f"Accuracy: {results[model_name]['accuracy']:.3f}")
        
        # LSTM model
        if 'lstm' in self.models:
            model = self.models['lstm']
            X_test_seq = self.lstm_sequences['X_test']
            y_test_seq = self.lstm_sequences['y_test']
            
            y_pred_proba = model.predict(X_test_seq).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            auc_score = roc_auc_score(y_test_seq, y_pred_proba)
            
            results['lstm'] = {
                'accuracy': (y_pred == y_test_seq).mean(),
                'auc': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"\nLSTM:")
            print(f"Accuracy: {results['lstm']['accuracy']:.3f}")
            print(f"AUC: {results['lstm']['auc']:.3f}")
        
        # Rule-based baseline
        if 'rule_based' in self.models:
            rule_func = self.models['rule_based']
            y_pred = rule_func(self.test_df)
            
            results['rule_based'] = {
                'accuracy': (y_pred == self.y_binary_test).mean(),
                'predictions': y_pred
            }
            
            print(f"\nRULE-BASED BASELINE:")
            print(f"Accuracy: {results['rule_based']['accuracy']:.3f}")
        
        return results
    
    def get_feature_importance(self):
        """Get feature importance from tree-based models"""
        importance_results = {}
        
        for model_name in ['rf_binary', 'xgb_binary']:
            if model_name in self.models:
                model = self.models[model_name]
                
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': self.feature_cols,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    importance_results[model_name] = importance_df
                    
                    print(f"\nTop 10 Features - {model_name.upper()}:")
                    print(importance_df.head(10).to_string(index=False))
        
        return importance_results
    
    def save_models(self):
        """Save trained models"""
        for model_name, model in self.models.items():
            if model_name != 'lstm':  # Save LSTM separately
                joblib.dump(model, f'{self.data_path}model_{model_name}.pkl')
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, f'{self.data_path}scaler_{scaler_name}.pkl')
        
        # Save LSTM model
        if 'lstm' in self.models:
            self.models['lstm'].save(f'{self.data_path}model_lstm.h5')
        
        print("Models saved successfully!")

if __name__ == "__main__":
    predictor = SolarPVPredictiveModels()
    
    if predictor.load_processed_data():
        predictor.build_binary_classification_models()
        predictor.build_anomaly_detection_model()
        predictor.build_time_series_model()
        predictor.build_rule_based_baseline()
        
        results = predictor.evaluate_models()
        importance = predictor.get_feature_importance()
        
        predictor.save_models()
        
        print("\nModel training and evaluation complete!")
