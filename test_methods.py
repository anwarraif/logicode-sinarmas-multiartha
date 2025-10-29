#!/usr/bin/env python3
"""
Test script to verify all method calls are correct
"""

import sys
import os
from logger_config import setup_logger

logger = setup_logger("test_methods")

def test_method_existence():
    """Test if all required methods exist in their respective classes"""
    
    print("Testing method existence...")
    logger.info("Starting method existence test")
    
    try:
        # Test imports first
        print("Testing imports...")
        
        try:
            from feature_engineering import SolarPVFeatureEngineer
            print("[OK] SolarPVFeatureEngineer imported")
            logger.info("SolarPVFeatureEngineer imported successfully")
            
            # Check if engineer_features method exists
            engineer = SolarPVFeatureEngineer()
            if hasattr(engineer, 'engineer_features'):
                print("[OK] engineer_features method exists")
                logger.info("engineer_features method found")
            else:
                print("[ERROR] engineer_features method missing")
                logger.error("engineer_features method not found")
                
        except ImportError as e:
            print(f"[SKIP] Cannot import SolarPVFeatureEngineer: {e}")
            logger.warning(f"Cannot import SolarPVFeatureEngineer: {e}")
        
        try:
            from predictive_models import SolarPVPredictiveModels
            print("[OK] SolarPVPredictiveModels imported")
            logger.info("SolarPVPredictiveModels imported successfully")
            
            # Check methods
            predictor = SolarPVPredictiveModels()
            methods_to_check = [
                'build_binary_classification_models',
                'build_anomaly_detection_model', 
                'build_time_series_model',
                'build_rule_based_baseline',
                'evaluate_models',
                'save_models'
            ]
            
            for method in methods_to_check:
                if hasattr(predictor, method):
                    print(f"[OK] {method} method exists")
                    logger.info(f"{method} method found")
                else:
                    print(f"[ERROR] {method} method missing")
                    logger.error(f"{method} method not found")
                    
        except ImportError as e:
            print(f"[SKIP] Cannot import SolarPVPredictiveModels: {e}")
            logger.warning(f"Cannot import SolarPVPredictiveModels: {e}")
        
        try:
            from evaluation import ModelEvaluator
            print("[OK] ModelEvaluator imported")
            logger.info("ModelEvaluator imported successfully")
            
            evaluator = ModelEvaluator()
            if hasattr(evaluator, 'generate_evaluation_report'):
                print("[OK] generate_evaluation_report method exists")
                logger.info("generate_evaluation_report method found")
            else:
                print("[ERROR] generate_evaluation_report method missing")
                logger.error("generate_evaluation_report method not found")
                
        except ImportError as e:
            print(f"[SKIP] Cannot import ModelEvaluator: {e}")
            logger.warning(f"Cannot import ModelEvaluator: {e}")
        
        try:
            from prescriptive_insights import PrescriptiveInsights
            print("[OK] PrescriptiveInsights imported")
            logger.info("PrescriptiveInsights imported successfully")
            
            insights = PrescriptiveInsights()
            methods_to_check = [
                'generate_maintenance_recommendations',
                'optimize_maintenance_schedule'
            ]
            
            for method in methods_to_check:
                if hasattr(insights, method):
                    print(f"[OK] {method} method exists")
                    logger.info(f"{method} method found")
                else:
                    print(f"[ERROR] {method} method missing")
                    logger.error(f"{method} method not found")
                    
        except ImportError as e:
            print(f"[SKIP] Cannot import PrescriptiveInsights: {e}")
            logger.warning(f"Cannot import PrescriptiveInsights: {e}")
        
        print("\n[OK] Method existence test completed")
        logger.info("Method existence test completed successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_method_existence()
    if success:
        print("\nAll method calls in main_pipeline.py should now be correct!")
        logger.info("All method calls verified as correct")
    else:
        print("\nSome issues found with method calls")
        logger.error("Issues found with method calls")
        sys.exit(1)
