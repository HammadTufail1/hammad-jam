#!/usr/bin/env python3
"""
Test script for FIBO-LightRAG system.
"""

# Import the complete system
from fibo_lightrag_complete import FiboLightRAGSystem, run_system_check, run_demo

def main():
    print("Testing FIBO-LightRAG System")
    print("=" * 40)
    
    # Run system check
    print("Running system check...")
    check_result = run_system_check()
    
    if check_result:
        print("\nRunning demo...")
        run_demo()
    else:
        print("\nSystem check failed, skipping demo.")

if __name__ == "__main__":
    main()