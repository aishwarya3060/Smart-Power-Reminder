#!/usr/bin/env python3
"""
Smart Power Reminder - Green AI Desktop App
Main launcher script
"""

import os
import sys
import subprocess

def main():
    print("🌱 Smart Power Reminder - Green AI Desktop App")
    print("=" * 50)

    # Check if we're in the right directory
    if not os.path.exists("src/frontend/app.py"):
        print("Error: Please run this script from the smart_power_reminder directory")
        return

    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit found")
    except ImportError:
        print("❌ Streamlit not found. Please install requirements:")
        print("pip install -r requirements.txt")
        return

    print("🚀 Launching Smart Power Reminder...")
    print("📊 Dashboard will open in your default web browser")
    print("🔧 Use Ctrl+C to stop the application")
    print("-" * 50)

    # Launch Streamlit app
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "src/frontend/app.py",
        "--server.headless", "false",
        "--server.runOnSave", "true",
        "--theme.base", "light"
    ])

if __name__ == "__main__":
    main()
