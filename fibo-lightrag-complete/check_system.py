#!/usr/bin/env python3
"""
System check script for FIBO-LightRAG.
Verifies that all components are working correctly.
"""

import os
import sys
import importlib

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_imports():
    """Check that all required modules can be imported."""
    print("🔍 Checking imports...")
    
    modules_to_check = [
        'fibo_lightrag',
        'fibo_lightrag.fibo.parser',
        'fibo_lightrag.lightrag.entity_extractor',
        'fibo_lightrag.graph.graph_builder',
        'fibo_lightrag.retrieval.document_processor',
        'fibo_lightrag.retrieval.vector_store',
        'fibo_lightrag.retrieval.retrieval_engine',
        'fibo_lightrag.retrieval.query_processor',
        'fibo_lightrag.integration.config',
        'fibo_lightrag.integration.fibo_lightrag_system'
    ]
    
    for module in modules_to_check:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            return False
    
    return True

def check_dependencies():
    """Check that all required dependencies are available."""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        'rdflib',
        'requests',
        'typing_extensions',
        'dataclasses',
        'json',
        'logging',
        'hashlib',
        'math',
        're'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            return False
    
    return True

def check_system_initialization():
    """Check that the system can be initialized."""
    print("\n🔍 Checking system initialization...")
    
    try:
        from fibo_lightrag.integration.fibo_lightrag_system import FiboLightRAGSystem
        from fibo_lightrag.integration.config import FiboLightRAGConfig
        
        # Create system
        config = FiboLightRAGConfig()
        system = FiboLightRAGSystem(config)
        print("  ✅ System object created")
        
        # Initialize system
        if system.initialize():
            print("  ✅ System initialized successfully")
            
            # Check statistics
            stats = system.get_statistics()
            print(f"  ✅ System statistics: {stats['status']}")
            
            return True
        else:
            print("  ❌ System initialization failed")
            return False
            
    except Exception as e:
        print(f"  ❌ System initialization error: {e}")
        return False

def check_basic_functionality():
    """Check basic system functionality."""
    print("\n🔍 Checking basic functionality...")
    
    try:
        from fibo_lightrag.integration.fibo_lightrag_system import FiboLightRAGSystem
        
        # Initialize system
        system = FiboLightRAGSystem()
        if not system.initialize():
            print("  ❌ Could not initialize system")
            return False
        
        # Test document addition
        test_doc = "Apple Inc. reported revenue of $10 billion in Q1 2024. The company's stock price increased by 5%."
        success = system.add_document(test_doc, "test_doc")
        
        if success:
            print("  ✅ Document addition works")
        else:
            print("  ❌ Document addition failed")
            return False
        
        # Test query
        response = system.query("What is Apple's revenue?")
        
        if response.results is not None:
            print(f"  ✅ Query processing works ({len(response.results)} results)")
        else:
            print("  ❌ Query processing failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality error: {e}")
        return False

def check_file_structure():
    """Check that all expected files exist."""
    print("\n🔍 Checking file structure...")
    
    expected_files = [
        'src/fibo_lightrag/__init__.py',
        'src/fibo_lightrag/fibo/__init__.py',
        'src/fibo_lightrag/fibo/parser.py',
        'src/fibo_lightrag/lightrag/__init__.py',
        'src/fibo_lightrag/lightrag/entity_extractor.py',
        'src/fibo_lightrag/graph/__init__.py',
        'src/fibo_lightrag/graph/graph_builder.py',
        'src/fibo_lightrag/graph/graph_operations.py',
        'src/fibo_lightrag/retrieval/__init__.py',
        'src/fibo_lightrag/retrieval/document_processor.py',
        'src/fibo_lightrag/retrieval/vector_store.py',
        'src/fibo_lightrag/retrieval/retrieval_engine.py',
        'src/fibo_lightrag/retrieval/query_processor.py',
        'src/fibo_lightrag/integration/__init__.py',
        'src/fibo_lightrag/integration/config.py',
        'src/fibo_lightrag/integration/fibo_lightrag_system.py',
        'requirements.txt',
        'README.md'
    ]
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing")
            return False
    
    return True

def run_system_check():
    """Run comprehensive system check."""
    print("🏦 FIBO-LightRAG System Check")
    print("=" * 50)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Dependencies", check_dependencies),
        ("Imports", check_imports),
        ("System Initialization", check_system_initialization),
        ("Basic Functionality", check_basic_functionality)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"  ❌ {check_name} check failed with error: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n📊 System Check Summary")
    print("-" * 30)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python demo_system.py' to see the system in action")
        print("2. Check the documentation in the docs/ folder")
        print("3. Try adding your own financial documents")
        return True
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure all required dependencies are installed")
        print("2. Check that Python path includes the src directory")
        print("3. Verify that all source files are present")
        return False

if __name__ == "__main__":
    try:
        success = run_system_check()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏸️  System check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ System check failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)