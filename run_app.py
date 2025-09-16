#!/usr/bin/env python3
"""
Launcher script for Supply Chain Analytics App
"""
import subprocess
import sys
import os

def main():
    # Find the app file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "supply_chain_app.py")
    
    print("Starting Supply Chain Analytics Dashboard...")
    print("The app will open in your default web browser")
    print("URL: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Launch the web app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\nApp stopped. Thanks for using Supply Chain Analytics!")
    except Exception as e:
        print(f"Error running app: {e}")
        print("Make sure you have Python installed and try again")

if __name__ == "__main__":
    main()