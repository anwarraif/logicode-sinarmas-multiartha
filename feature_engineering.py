import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime, timedelta

class SolarPVFeatureEngineer:
    def __init__(self, data_path='.'):
        self.data_path = data_path
        self.performance_df = None
        self.maintenance_df = None
        self.failures_df = None
        self.parts_df = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare data"""
        self.performance_df = pd.read_csv('performance_data.csv')
        self.maintenance_df = pd.read_csv('maintenance_logs.csv')
        self.failures_df = pd.read_csv('failure_records.csv')
        self.parts_df = pd.read_csv('spare_parts.csv')
        
        # Convert datetime columns
        self.performance_df['timestamp'] = pd.to_datetime(self.performance_df['timestamp'])
        self.maintenance_df['service_date'] = pd.to_datetime(self.maintenance_df['service_date'])
        self.failures_df['failure_start'] = pd.to_datetime(self.failures_df['failure_start'])
        self.failures_df['failure_end'] = pd.to_datetime(self.failures_df['failure_end'])
        
    def create_temporal_split(self, test_size=0.2):
        """Create temporal train/test split to prevent data leakage"""
        # Sort by timestamp
        self.performance_df = self.performance_df.sort_values('timestamp')
        
        # Find split point based on time (not random)
        split_date = self.performance_df['timestamp'].quantile(1 - test_size)
        
        train_mask = self.performance_df['timestamp'] < split_date
        test_mask = self.performance_df['timestamp'] >= split_date
        
        print(f"Train period: {self.performance_df[train_mask]['timestamp'].min()} to {self.performance_df[train_mask]['timestamp'].max()}")
        print(f"Test period: {self.performance_df[test_mask]['timestamp'].min()} to {self.performance_df[test_mask]['timestamp'].max()}")
        
        return train_mask, test_mask, split_date
    
    def create_performance_features(self, df):
        """Create performance-based features"""
        df = df.copy()
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Performance ratios
        df['performance_ratio'] = df['ac_power_kw'] / (df['irradiance_wm2'] / 1000 * 100)  # Assuming 100kW nominal
        df['performance_ratio'] = df['performance_ratio'].fillna(0)
        
        # Temperature-related features
        df['temp_deviation'] = df['module_temp_c'] - df['ambient_temp_c']
        df['high_temp_flag'] = (df['module_temp_c'] > 70).astype(int)
        
        # Rolling statistics (prevent leakage by using only past data)
        for inverter in df['inverter_id'].unique():
            inv_mask = df['inverter_id'] == inverter
            
            # 24-hour rolling averages (past data only)
            df.loc[inv_mask, 'energy_24h_avg'] = df.loc[inv_mask, 'energy_kwh'].rolling(window=288, min_periods=1).mean()
            df.loc[inv_mask, 'temp_24h_avg'] = df.loc[inv_mask, 'module_temp_c'].rolling(window=288, min_periods=1).mean()
            df.loc[inv_mask, 'irradiance_24h_avg'] = df.loc[inv_mask, 'irradiance_wm2'].rolling(window=288, min_periods=1).mean()
            
            # Performance degradation trend (7-day rolling)
            df.loc[inv_mask, 'performance_trend'] = df.loc[inv_mask, 'performance_ratio'].rolling(window=2016, min_periods=1).mean()
            
            # Fault frequency (past 30 days)
            df.loc[inv_mask, 'fault_count_30d'] = (df.loc[inv_mask, 'fault_code'] > 0).rolling(window=8640, min_periods=1).sum()
        
        return df
    
    def create_maintenance_features(self, perf_df, split_date):
        """Create maintenance-based features (only using past data)"""
        # Only use maintenance data before split_date for training features
        past_maintenance = self.maintenance_df[self.maintenance_df['service_date'] < split_date].copy()
        past_failures = self.failures_df[self.failures_df['failure_start'] < split_date].copy()
        
        maintenance_features = []
        
        for _, row in perf_df.iterrows():
            inverter_id = row['inverter_id']
            timestamp = row['timestamp']
            
            # Only look at maintenance/failures before current timestamp
            inv_maintenance = past_maintenance[
                (past_maintenance['inverter_id'] == inverter_id) & 
                (past_maintenance['service_date'] <= timestamp)
            ]
            
            inv_failures = past_failures[
                (past_failures['inverter_id'] == inverter_id) & 
                (past_failures['failure_start'] <= timestamp)
            ]
            
            # Days since last maintenance
            if len(inv_maintenance) > 0:
                last_maintenance = inv_maintenance['service_date'].max()
                days_since_maintenance = (timestamp - last_maintenance).days
            else:
                days_since_maintenance = 999  # No maintenance yet
            
            # Days since last cleaning
            last_cleaning = inv_maintenance[inv_maintenance['event_type'] == 'cleaning']
            if len(last_cleaning) > 0:
                days_since_cleaning = (timestamp - last_cleaning['service_date'].max()).days
            else:
                days_since_cleaning = 999
            
            # Failure history
            failure_count_6m = len(inv_failures[inv_failures['failure_start'] >= timestamp - timedelta(days=180)])
            failure_count_1y = len(inv_failures[inv_failures['failure_start'] >= timestamp - timedelta(days=365)])
            
            # Average downtime
            if len(inv_failures) > 0:
                avg_downtime = inv_failures['downtime_hours'].mean()
                total_downtime_6m = inv_failures[
                    inv_failures['failure_start'] >= timestamp - timedelta(days=180)
                ]['downtime_hours'].sum()
            else:
                avg_downtime = 0
                total_downtime_6m = 0
            
            maintenance_features.append({
                'days_since_maintenance': min(days_since_maintenance, 365),  # Cap at 1 year
                'days_since_cleaning': min(days_since_cleaning, 180),  # Cap at 6 months
                'failure_count_6m': failure_count_6m,
                'failure_count_1y': failure_count_1y,
                'avg_downtime_hours': avg_downtime,
                'total_downtime_6m': total_downtime_6m
            })
        
        return pd.DataFrame(maintenance_features)
    
    def create_failure_labels(self, perf_df, prediction_window_days=28):
        """Create failure labels for prediction (2-4 weeks ahead)"""
        labels = []
        
        for _, row in perf_df.iterrows():
            inverter_id = row['inverter_id']
            timestamp = row['timestamp']
            
            # Look for failures in the next 2-4 weeks
            future_start = timestamp + timedelta(days=14)  # Start looking 2 weeks ahead
            future_end = timestamp + timedelta(days=prediction_window_days)
            
            future_failures = self.failures_df[
                (self.failures_df['inverter_id'] == inverter_id) & 
                (self.failures_df['failure_start'] >= future_start) & 
                (self.failures_df['failure_start'] <= future_end)
            ]
            
            # Binary label: will fail in prediction window
            will_fail = len(future_failures) > 0
            
            # Failure type (for multi-class prediction)
            if will_fail:
                failure_type = future_failures.iloc[0]['cause']
            else:
                failure_type = 'no_failure'
            
            # Days until failure (for regression)
            if will_fail:
                days_until_failure = (future_failures.iloc[0]['failure_start'] - timestamp).days
            else:
                days_until_failure = prediction_window_days + 1  # Beyond prediction window
            
            labels.append({
                'will_fail': will_fail,
                'failure_type': failure_type,
                'days_until_failure': days_until_failure
            })
        
        return pd.DataFrame(labels)
    
    def engineer_features(self):
        """Main feature engineering pipeline"""
        print("Starting feature engineering...")
        
        # Create temporal split first (prevent leakage)
        train_mask, test_mask, split_date = self.create_temporal_split()
        
        # Create performance features
        print("Creating performance features...")
        perf_features = self.create_performance_features(self.performance_df)
        
        # Create maintenance features (using only past data)
        print("Creating maintenance features...")
        maintenance_features = self.create_maintenance_features(perf_features, split_date)
        
        # Combine features
        feature_df = pd.concat([perf_features.reset_index(drop=True), 
                               maintenance_features.reset_index(drop=True)], axis=1)
        
        # Create labels
        print("Creating failure labels...")
        labels_df = self.create_failure_labels(perf_features)
        
        # Combine features and labels
        final_df = pd.concat([feature_df.reset_index(drop=True), 
                             labels_df.reset_index(drop=True)], axis=1)
        
        # Split into train/test
        train_df = final_df[train_mask].copy()
        test_df = final_df[test_mask].copy()
        
        # Feature selection (remove non-predictive columns)
        feature_cols = [col for col in final_df.columns if col not in [
            'timestamp', 'inverter_id', 'inverter_status', 'fault_code',
            'will_fail', 'failure_type', 'days_until_failure'
        ]]
        
        X_train = train_df[feature_cols]
        y_train = train_df[['will_fail', 'failure_type', 'days_until_failure']]
        X_test = test_df[feature_cols]
        y_test = test_df[['will_fail', 'failure_type', 'days_until_failure']]
        
        # Handle missing values
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())  # Use train median for test
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {len(feature_cols)}")
        print(f"Failure rate in training: {y_train['will_fail'].mean():.3f}")
        
        return X_train, X_test, y_train, y_test, feature_cols, train_df, test_df

if __name__ == "__main__":
    engineer = SolarPVFeatureEngineer()
    engineer.load_data()
    X_train, X_test, y_train, y_test, feature_cols, train_df, test_df = engineer.engineer_features()
    
    # Save processed data
    train_df.to_csv('train_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    
    print("Feature engineering complete!")
