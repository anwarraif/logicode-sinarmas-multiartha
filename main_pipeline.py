#!/usr/bin/env python3
"""
Solar PV Predictive Maintenance Pipeline
Complete end-to-end solution for predictive maintenance of solar PV assets
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logger_config import setup_logger
from model_selector import ModelSelector

# Setup logger
logger = setup_logger("main_pipeline")

def check_data_files():
    """Check if all required data files exist"""
    data_files = [
        'performance_data.csv',
        'maintenance_logs.csv', 
        'failure_records.csv',
        'spare_parts.csv',
        'financial_data.csv'
    ]
    
    missing_files = []
    existing_files = []
    
    for file in data_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            existing_files.append((file, size))
            logger.info(f"Found {file} - {size:,} bytes")
        else:
            missing_files.append(file)
            logger.warning(f"Missing file: {file}")
    
    return existing_files, missing_files

def run_complete_pipeline():
    """Run the complete predictive maintenance pipeline"""
    
    print("=" * 60)
    print("SOLAR PV PREDICTIVE MAINTENANCE PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    logger.info("Starting Solar PV Predictive Maintenance Pipeline")
    print()
    
    # Step 0: Check data files
    print("STEP 0: CHECKING DATA FILES")
    print("-" * 40)
    logger.info("Step 0: Data Files Check")
    
    existing_files, missing_files = check_data_files()
    
    if missing_files:
        print(f"[WARNING] Missing data files: {missing_files}")
        logger.warning(f"Missing data files: {missing_files}")
    else:
        print(f"[OK] All data files present ({len(existing_files)} files)")
        logger.info(f"All data files present: {len(existing_files)} files")
    
    # Import modules with error handling
    try:
        print("\nImporting pipeline modules...")
        logger.info("Importing pipeline modules")
        
        from data_generator import RealisticSolarPVDataGenerator
        print("[OK] Data generator imported")
        logger.info("Data generator module imported successfully")
        
        from data_exploration import SolarPVDataExplorer
        print("[OK] Data explorer imported")
        logger.info("Data explorer module imported successfully")
        
        from feature_engineering import SolarPVFeatureEngineer
        print("[OK] Feature engineer imported")
        logger.info("Feature engineer module imported successfully")
        
        from predictive_models import SolarPVPredictiveModels
        print("[OK] Predictive models imported")
        logger.info("Predictive models module imported successfully")
        
        from evaluation import ModelEvaluator
        print("[OK] Model evaluator imported")
        logger.info("Model evaluator module imported successfully")
        
        from prescriptive_insights import PrescriptiveInsights
        print("[OK] Prescriptive insights imported")
        logger.info("Prescriptive insights module imported successfully")
        
        print("[OK] All pipeline modules imported successfully")
        logger.info("All pipeline modules imported successfully")
        
    except ImportError as e:
        print(f"[ERROR] Failed to import pipeline modules: {e}")
        logger.error(f"Import error: {e}")
        print("\nThis error typically occurs when required packages are not installed.")
        print("Please ensure you're running in the correct environment with all dependencies.")
        return False
    
    # Step 1: Generate realistic data
    print(f"\nSTEP 1: GENERATING REALISTIC SOLAR PV DATA")
    print("-" * 40)
    logger.info("Step 1: Data Generation")
    
    try:
        # Check if data files already exist
        data_files = ['performance_data.csv', 'maintenance_logs.csv', 'failure_records.csv', 'spare_parts.csv', 'financial_data.csv']
        if all(os.path.exists(f) for f in data_files):
            print("[SKIP] Data files already exist, skipping generation")
            logger.info("Data files already exist, skipping generation")
        else:
            print("Generating missing data files...")
            logger.info("Starting data generation for missing files")
            
            generator = RealisticSolarPVDataGenerator(n_inverters=50)
            
            if not os.path.exists('performance_data.csv'):
                print("Generating performance data...")
                logger.info("Generating performance data")
                performance_df = generator.generate_performance_data()
                performance_df.to_csv('performance_data.csv', index=False)
                print(f"[OK] Generated {len(performance_df):,} performance records")
                logger.info(f"Generated {len(performance_df):,} performance records")
            
            if not os.path.exists('maintenance_logs.csv'):
                print("Generating maintenance data...")
                logger.info("Generating maintenance data")
                maintenance_df, failures_df, parts_df = generator.generate_maintenance_data()
                maintenance_df.to_csv('maintenance_logs.csv', index=False)
                failures_df.to_csv('failure_records.csv', index=False)
                parts_df.to_csv('spare_parts.csv', index=False)
                print(f"[OK] Generated {len(maintenance_df)} maintenance, {len(failures_df)} failure, {len(parts_df)} parts records")
                logger.info(f"Generated maintenance data: {len(maintenance_df)} maintenance, {len(failures_df)} failure, {len(parts_df)} parts records")
            
            if not os.path.exists('financial_data.csv'):
                print("Generating financial data...")
                logger.info("Generating financial data")
                financial_df = generator.generate_financial_data()
                financial_df.to_csv('financial_data.csv', index=False)
                print(f"[OK] Generated {len(financial_df)} financial records")
                logger.info(f"Generated {len(financial_df)} financial records")
        
    except Exception as e:
        print(f"[ERROR] Error in data generation: {e}")
        logger.error(f"Error in data generation: {e}")
        return False
    
    # Step 2: Data exploration and quality assessment
    print(f"\nSTEP 2: DATA EXPLORATION & QUALITY ASSESSMENT")
    print("-" * 40)
    logger.info("Step 2: Data Exploration & Quality Assessment")
    
    try:
        explorer = SolarPVDataExplorer()
        print("Loading data for exploration...")
        logger.info("Loading data for exploration")
        
        if explorer.load_data():
            print("[OK] Data loaded successfully")
            logger.info("Data loaded successfully for exploration")
            
            print("Generating data quality report...")
            logger.info("Generating data quality report")
            explorer.generate_data_quality_report()
            
            print("[OK] Data quality assessment completed")
            logger.info("Data quality assessment completed successfully")
        else:
            print("[ERROR] Failed to load data for exploration")
            logger.error("Failed to load data for exploration")
            return False
    except Exception as e:
        print(f"[ERROR] Error in data exploration: {e}")
        logger.error(f"Error in data exploration: {e}")
        return False
    
    # Step 3: Feature engineering
    print(f"\nSTEP 3: FEATURE ENGINEERING")
    print("-" * 40)
    logger.info("Step 3: Feature Engineering")
    
    try:
        engineer = SolarPVFeatureEngineer()
        print("Loading data for feature engineering...")
        logger.info("Loading data for feature engineering")
        engineer.load_data()
        
        print("Engineering features...")
        logger.info("Starting feature engineering")
        X_train, X_test, y_train, y_test, feature_cols = engineer.engineer_features()
        
        # Save processed data
        train_df = X_train.copy()
        train_df['target'] = y_train
        test_df = X_test.copy()
        test_df['target'] = y_test
        
        train_df.to_csv('train_data.csv', index=False)
        test_df.to_csv('test_data.csv', index=False)
        
        print("[OK] Feature engineering completed")
        print(f"  - Training samples: {len(X_train):,}")
        print(f"  - Test samples: {len(X_test):,}")
        print(f"  - Features: {len(feature_cols)}")
        logger.info(f"Feature engineering completed: {len(X_train):,} train, {len(X_test):,} test samples, {len(feature_cols)} features")
        
    except Exception as e:
        print(f"[ERROR] Error in feature engineering: {e}")
        logger.error(f"Error in feature engineering: {e}")
        return False
    
    # Step 4: Model development
    print(f"\nSTEP 4: PREDICTIVE MODEL DEVELOPMENT")
    print("-" * 40)
    logger.info("Step 4: Predictive Model Development")
    
    try:
        predictor = SolarPVPredictiveModels()
        
        print("Loading processed data...")
        logger.info("Loading processed data for modeling")
        
        if predictor.load_processed_data():
            print("[OK] Processed data loaded")
            logger.info("Processed data loaded successfully")
            
            # Build models
            print("Building binary classification models...")
            logger.info("Building binary classification models")
            predictor.build_binary_classification_models()
            print("[OK] Binary classification models built")
            
            print("Building anomaly detection model...")
            logger.info("Building anomaly detection model")
            predictor.build_anomaly_detection_model()
            print("[OK] Anomaly detection model built")
            
            print("Building time series model...")
            logger.info("Building time series model")
            predictor.build_time_series_model()
            print("[OK] Time series model built")
            
            print("Building rule-based baseline...")
            logger.info("Building Rule-based baseline")
            predictor.build_rule_based_baseline()
            print("[OK] Rule-based baseline built")
            
            print("[OK] All models trained successfully")
            logger.info("All models trained successfully")
            
            # Evaluate models
            print("Evaluating models...")
            logger.info("Starting model evaluation")
            results = predictor.evaluate_models()
            print("[OK] Model evaluation completed")
            logger.info("Model evaluation completed")
            
            # Save models
            print("Saving models...")
            logger.info("Saving trained models")
            predictor.save_models()
            print("[OK] Models saved")
            logger.info("Models saved successfully")
            
        else:
            print("[ERROR] Failed to load processed data for modeling")
            logger.error("Failed to load processed data for modeling")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error in model development: {e}")
        logger.error(f"Error in model development: {e}")
        return False
    
    # Step 5: Model evaluation and selection
    print(f"\nSTEP 5: MODEL EVALUATION & BEST MODEL SELECTION")
    print("-" * 40)
    logger.info("Step 5: Model Evaluation & Best Model Selection")
    
    try:
        evaluator = ModelEvaluator()
        model_selector = ModelSelector()
        
        # Generate evaluation report
        print("Generating evaluation report...")
        logger.info("Generating evaluation report")
        comparison_df = evaluator.generate_evaluation_report(results)
        
        # Select best model based on business metrics
        print("Selecting best model based on business metrics...")
        logger.info("Selecting best model based on business metrics")
        best_model_name, best_score = model_selector.select_best_model(results)
        
        print(f"[OK] Best model selected: {best_model_name} (score: {best_score:.4f})")
        logger.info(f"Best model selected: {best_model_name} with score: {best_score:.4f}")
        
        # Save best model information
        print("Saving best model information...")
        logger.info("Saving best model information")
        model_info = model_selector.save_best_model_info()
        
        print("[OK] Model evaluation and selection completed")
        logger.info("Model evaluation and selection completed successfully")
        
    except Exception as e:
        print(f"[ERROR] Error in model evaluation: {e}")
        logger.error(f"Error in model evaluation: {e}")
        return False
    
    # Step 6: Generate predictions with best model
    print(f"\nSTEP 6: GENERATING PREDICTIONS WITH BEST MODEL")
    print("-" * 40)
    logger.info("Step 6: Generating Predictions with Best Model")
    
    try:
        # Generate predictions using best model
        print(f"Generating predictions with {best_model_name}...")
        logger.info(f"Generating predictions with best model: {best_model_name}")
        predictions_df = model_selector.generate_predictions(predictor)
        
        print(f"[OK] Generated {len(predictions_df)} failure predictions")
        logger.info(f"Generated {len(predictions_df)} failure predictions")
        
    except Exception as e:
        print(f"[ERROR] Error generating predictions: {e}")
        logger.error(f"Error generating predictions: {e}")
        return False
    
    # Step 7: Prescriptive insights and recommendations
    print(f"\nSTEP 7: PRESCRIPTIVE INSIGHTS & RECOMMENDATIONS")
    print("-" * 40)
    logger.info("Step 7: Prescriptive Insights & Recommendations")
    
    try:
        insights = PrescriptiveInsights()
        
        # Generate maintenance recommendations using actual predictions
        print("Generating maintenance recommendations...")
        logger.info("Generating maintenance recommendations")
        recommendations_df = insights.generate_maintenance_recommendations(predictions_df)
        recommendations_df.to_csv('maintenance_recommendations.csv', index=False)
        print(f"[OK] Generated {len(recommendations_df)} maintenance recommendations")
        logger.info(f"Generated {len(recommendations_df)} maintenance recommendations")
        
        # Optimize maintenance schedule
        print("Optimizing maintenance schedule...")
        logger.info("Optimizing maintenance schedule")
        schedule_df = insights.optimize_maintenance_schedule(recommendations_df)
        schedule_df.to_csv('maintenance_schedule.csv', index=False)
        print(f"[OK] Generated maintenance schedule with {len(schedule_df)} entries")
        logger.info(f"Generated maintenance schedule with {len(schedule_df)} entries")
        
        # Calculate business impact
        print("Calculating business impact...")
        logger.info("Calculating business impact")
        business_impact = insights.calculate_business_impact(recommendations_df)
        
        print("[OK] Prescriptive insights generated")
        print(f"  - Maintenance recommendations: {len(recommendations_df)} inverters")
        print(f"  - Maintenance schedule: {len(schedule_df)} entries planned")
        print(f"  - Estimated cost savings: ${business_impact.get('total_savings', 0):,.2f}")
        logger.info(f"Prescriptive insights generated: {len(recommendations_df)} recommendations, {len(schedule_df)} schedule entries")
        
    except Exception as e:
        print(f"[ERROR] Error in prescriptive insights: {e}")
        logger.error(f"Error in prescriptive insights: {e}")
        return False
    
    # Pipeline completion summary
    print(f"\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    
    print("[OK] Data Generation: Realistic solar PV operational data created")
    print("[OK] Data Exploration: Quality issues identified and reconciliation methods proposed")
    print("[OK] Feature Engineering: Predictive features created with proper temporal split")
    print("[OK] Model Development: Multiple algorithms trained and compared")
    print("[OK] Model Evaluation: Best model selected based on business metrics")
    print("[OK] Failure Predictions: Generated using best performing model")
    print("[OK] Prescriptive Insights: Actionable maintenance recommendations generated")
    
    print(f"\nKey Outputs:")
    print(f"- Performance data: performance_data.csv")
    print(f"- Maintenance logs: maintenance_logs.csv")
    print(f"- Failure records: failure_records.csv")
    print(f"- Best model info: best_model_info.pkl")
    print(f"- Model evaluation: model_evaluation_results.csv")
    print(f"- Failure predictions: failure_predictions.csv")
    print(f"- Maintenance recommendations: maintenance_recommendations.csv")
    print(f"- Maintenance schedule: maintenance_schedule.csv")
    
    print(f"\nCompleted at: {datetime.now()}")
    logger.info("Solar PV Predictive Maintenance Pipeline completed successfully")
    
    return True

if __name__ == "__main__":
    success = run_complete_pipeline()
    
    if success:
        print(f"\n Pipeline executed successfully!")
        print(f"All components are ready for production deployment.")
        logger.info("Pipeline executed successfully - ready for production")
    else:
        print(f"\n Pipeline execution failed.")
        print(f"Please check the error messages above.")
        logger.error("Pipeline execution failed")
        sys.exit(1)
