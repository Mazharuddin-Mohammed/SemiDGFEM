#!/usr/bin/env python3
"""
Test script to verify documentation build configuration
"""

import sys
import subprocess
import os

def test_requirements():
    """Test if all required packages can be installed"""
    print("Testing documentation requirements...")
    
    try:
        # Test importing key packages
        import sphinx
        print(f"✅ Sphinx version: {sphinx.__version__}")
        
        import sphinx_rtd_theme
        print(f"✅ RTD theme available")
        
        import myst_parser
        print(f"✅ MyST parser available")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_sphinx_build():
    """Test if Sphinx can build the documentation"""
    print("\nTesting Sphinx build...")
    
    try:
        # Change to docs directory
        os.chdir('docs')
        
        # Run sphinx-build
        result = subprocess.run([
            'sphinx-build', '-b', 'html', '-W', '.', '_build/html'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Sphinx build successful")
            return True
        else:
            print(f"❌ Sphinx build failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Build test error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 SemiDGFEM Documentation Build Test")
    print("=" * 50)
    
    # Test 1: Requirements
    req_ok = test_requirements()
    
    # Test 2: Build (only if requirements are OK)
    build_ok = False
    if req_ok:
        build_ok = test_sphinx_build()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"Requirements: {'✅ PASS' if req_ok else '❌ FAIL'}")
    print(f"Build Test: {'✅ PASS' if build_ok else '❌ FAIL'}")
    
    if req_ok and build_ok:
        print("\n🎉 All tests passed! Documentation should build on ReadTheDocs.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
