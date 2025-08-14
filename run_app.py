#!/usr/bin/env python3
"""
Runner script for the NLG Paper Trends Analysis UI

Usage:
    python run_app.py

This will start the Streamlit server on http://localhost:8501
"""

import subprocess
import sys
import os

def main():
    # Change to the directory containing the app
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check if data file exists
    if not os.path.exists('data/all_papers.json'):
        print("❌ Error: data/all_papers.json not found!")
        print("Please ensure the data file is in the correct location.")
        sys.exit(1)
    
    print("🚀 Starting NLG Paper Trends Analysis UI...")
    print("📊 The app will open at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run(['streamlit', 'run', 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Streamlit not found!")
        print("Please install Streamlit: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()