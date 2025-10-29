import subprocess
import sys
import os
from datetime import datetime
from logger_config import setup_logger

def run_script(script_name):
    """Run Python script with logging"""
    logger = setup_logger("script_runner", f"execution_{datetime.now().strftime('%Y%m%d')}.log")
    
    logger.info(f"Starting execution: {script_name}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            logger.info(f"SUCCESS: {script_name} completed successfully")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
        else:
            logger.error(f"FAILED: {script_name} failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Error: {result.stderr}")
                
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"EXCEPTION: Error running {script_name}: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_with_logging.py <script_name.py>")
        sys.exit(1)
    
    script = sys.argv[1]
    success = run_script(script)
    sys.exit(0 if success else 1)
