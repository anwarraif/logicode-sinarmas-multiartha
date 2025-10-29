#!/usr/bin/env python3
"""
Streamlit Runner for Solar PV Predictive Maintenance Dashboard
"""

import subprocess
import sys
import os
from logger_config import setup_logger

logger = setup_logger("streamlit_runner")

def run_streamlit():
    """Run Streamlit dashboard"""
    
    print("=" * 60)
    print("SOLAR PV PREDICTIVE MAINTENANCE DASHBOARD")
    print("=" * 60)
    print()
    
    logger.info("Starting Streamlit dashboard")
    
    try:
        # Check if streamlit is installed
        import streamlit
        print("Streamlit is available")
        logger.info("Streamlit package found")
        
        # Check if streamlit_app.py exists
        if os.path.exists('streamlit_app.py'):
            print("Dashboard file found")
            logger.info("streamlit_app.py found")
        else:
            print("streamlit_app.py not found")
            logger.error("streamlit_app.py not found")
            return False
        
        print("\nStarting dashboard...")
        print("Dashboard will open in your browser")
        print("URL: http://localhost:8501")
        print("\nPress Ctrl+C to stop the dashboard")
        print("-" * 60)
        
        logger.info("Launching Streamlit dashboard")
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except ImportError:
        print("Streamlit not installed")
        print("Install with: pip install streamlit")
        logger.error("Streamlit not installed")
        return False
    
    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user")
        logger.info("Dashboard stopped by user")
        return True
    
    except Exception as e:
        print(f"Error running dashboard: {e}")
        logger.error(f"Error running dashboard: {e}")
        return False

if __name__ == "__main__":
    success = run_streamlit()
    
    if success:
        print("\nDashboard session completed")
        logger.info("Dashboard session completed successfully")
    else:
        print("\nDashboard failed to start")
        logger.error("Dashboard failed to start")
        sys.exit(1)
