#!/usr/bin/env python3
"""
Demo runner script với fallback model
Chạy web demo với model gốc CodeT5 nếu model train không load được
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Kiểm tra và cài đặt dependencies"""
    required_packages = [
        'streamlit',
        'transformers',
        'torch',
        'pandas'
    ]
    
    print("🔍 Checking dependencies...")
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} missing")
    
    if missing:
        print("\n📦 Installing missing packages...")
        for package in missing:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        print("✅ All packages installed!")
    
    return True

def run_demo():
    """Chạy Streamlit demo"""
    demo_path = Path(__file__).parent / "app_simple.py"
    
    print("\n" + "="*60)
    print("🏦 BANKING AI TEST GENERATOR - WEB DEMO")
    print("="*60)
    print("\n🚀 Starting web interface...")
    print("📍 URL will be: http://localhost:8501")
    print("\n💡 Tips:")
    print("  - Model sẽ load trong lần generate đầu tiên")
    print("  - Có thể mất 30-60 giây để load model")
    print("  - Nếu model train không load được, sẽ dùng model gốc")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(demo_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n✅ Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🎯 Banking Test Generator Demo Launcher\n")
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Failed to setup dependencies")
        sys.exit(1)
    
    # Run demo
    if not run_demo():
        print("❌ Failed to run demo")
        sys.exit(1)
    
    print("\n✅ Demo completed successfully!")
