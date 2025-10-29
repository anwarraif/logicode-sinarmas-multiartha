#!/usr/bin/env python3
"""
Solar PV Predictive Maintenance Pipeline - Safe Version
Complete end-to-end solution for predictive maintenance of solar PV assets
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logger_config import setup_logger

# Setup logger
logger = setup_logger("main_pipeline")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

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
    
    # Step 0: Check dependencies
    print("STEP 0: CHECKING SYSTEM REQUIREMENTS")
    print("-" * 40)
    logger.info("Step 0: System Requirements Check")
    
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"[ERROR] Missing required packages: {', '.join(missing_deps)}")
        print(f"Please install with: pip install {' '.join(missing_deps)}")
        logger.error(f"Missing dependencies: {missing_deps}")
        print("\nTo install dependencies, run:")
        print("pip install pandas numpy matplotlib seaborn scikit-learn")
        return False
    else:
        print("[OK] All required packages are installed")
        logger.info("All dependencies are available")
    
    # Check data files
    existing_files, missing_files = check_data_files()
    
    if missing_files:
        print(f"[WARNING] Missing data files: {missing_files}")
        logger.warning(f"Missing data files: {missing_files}")
    else:
        print(f"[OK] All data files present ({len(existing_files)} files)")
        logger.info(f"All data files present: {len(existing_files)} files")
    
    # Import modules after dependency check
    try:
        from data_generator import RealisticSolarPVDataGenerator
        from data_exploration import SolarPVDataExplorer
        from feature_engineering import SolarPVFeatureEngineer
        from predictive_models import SolarPVPredictiveModels
        from evaluation import ModelEvaluator
        from prescriptive_insights import PrescriptiveInsights
        logger.info("Successfully imported all pipeline modules")
    except ImportError as e:
        print(f"[ERROR] Failed to import pipeline modules: {e}")
        logger.error(f"Import error: {e}")
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
        if explorer.load_data():
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
        X_train, X_test, y_train, y_test, feature_cols = engineer.prepare_features()
        
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
        
        if predictor.load_processed_data():
            # Train models
            print("Training Random Forest...")
            predictor.train_random_forest()
            logger.info("Random Forest model trained")
            
            print("Training Gradient Boosting...")
            predictor.train_gradient_boosting()
            logger.info("Gradient Boosting model trained")
            
            print("Training Neural Network...")
            predictor.train_neural_network()
            logger.info("Neural Network model trained")
            
            print("Building Rule-based Baseline...")
            predictor.build_rule_based_baseline()
            logger.info("Rule-based baseline built")
            
            print("[OK] Models trained successfully")
            logger.info("All models trained successfully")
            
            # Evaluate models
            print("Evaluating models...")
            results = predictor.evaluate_models()
            logger.info("Model evaluation completed")
            
            # Save models
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
    
    # Step 5: Model evaluation
    print(f"\nSTEP 5: MODEL EVALUATION & COMPARISON")
    print("-" * 40)
    logger.info("Step 5: Model Evaluation & Comparison")
    
    try:
        evaluator = ModelEvaluator()
        
        # Generate comprehensive evaluation
        print("Generating evaluation report...")
        results = evaluator.evaluate_all_models()
        comparison_df = evaluator.generate_evaluation_report(results)
        
        print("[OK] Model evaluation completed")
        logger.info("Model evaluation completed successfully")
        
    except Exception as e:
        print(f"[ERROR] Error in model evaluation: {e}")
        logger.error(f"Error in model evaluation: {e}")
        return False
    
    # Step 6: Prescriptive insights
    print(f"\nSTEP 6: PRESCRIPTIVE INSIGHTS & RECOMMENDATIONS")
    print("-" * 40)
    logger.info("Step 6: Prescriptive Insights & Recommendations")
    
    try:
        insights = PrescriptiveInsights()
        
        # Generate maintenance recommendations
        print("Generating maintenance recommendations...")
        recommendations_df = insights.generate_maintenance_recommendations()
        recommendations_df.to_csv('maintenance_recommendations.csv', index=False)
        logger.info(f"Generated {len(recommendations_df)} maintenance recommendations")
        
        # Generate maintenance schedule
        print("Generating maintenance schedule...")
        schedule_df = insights.generate_maintenance_schedule()
        schedule_df.to_csv('maintenance_schedule.csv', index=False)
        logger.info(f"Generated maintenance schedule with {len(schedule_df)} entries")
        
        print("[OK] Prescriptive insights generated")
        print(f"  - Maintenance recommendations: {len(recommendations_df)} inverters")
        print(f"  - Maintenance schedule: {len(schedule_df)} weeks planned")
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
    print("[OK] Model Evaluation: Performance metrics and business impact calculated")
    print("[OK] Prescriptive Insights: Actionable maintenance recommendations generated")
    
    print(f"\nKey Outputs:")
    print(f"- Performance data: performance_data.csv")
    print(f"- Maintenance logs: maintenance_logs.csv")
    print(f"- Failure records: failure_records.csv")
    print(f"- Trained models: model_*.pkl")
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
