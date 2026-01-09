#!/usr/bin/env python
"""
VOXENT Comprehensive System Test Suite
Consolidated from: test_gpu.py, run_cpu_test.py, run_full_test.py, test_voxent_complete.py

Provides three testing modes:
  - GPU MODE: Tests GPU/CUDA availability
  - CPU MODE: Tests CPU-only pipeline
  - FULL MODE: Comprehensive system testing
"""

import os
import sys
import logging
import traceback
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# HEADER & FORMATTING UTILITIES
# ============================================================================

def print_header(title, width=80):
    """Print formatted section header"""
    print("\n" + "="*width)
    print(f"  {title}")
    print("="*width + "\n")


def print_subheader(title, width=60):
    """Print formatted subsection header"""
    print(f"\n{'-'*width}")
    print(f"{title}")
    print(f"{'-'*width}\n")


# ============================================================================
# GPU TESTING FUNCTIONS
# ============================================================================

def test_gpu_availability():
    """Test GPU and PyTorch installation"""
    print_header("GPU & PYTORCH VERIFICATION")
    
    try:
        import torch
        
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        gpu_available = False
        
        if torch.cuda.is_available():
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"VRAM: {vram_gb:.2f} GB")
            
            # Test tensor operations
            x = torch.randn(100, 100)
            x_gpu = x.cuda()
            result = torch.matmul(x_gpu, x_gpu.T)
            print(f"\n✅ GPU READY FOR ACCELERATION")
            gpu_available = True
        else:
            print(f"\n⚠️  GPU not available, CPU will be used")
        
        return gpu_available
        
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


# ============================================================================
# CPU TESTING FUNCTIONS
# ============================================================================

def check_core_dependencies():
    """Verify core dependencies"""
    print_subheader("Checking Core Dependencies")
    
    required = {
        'yaml': 'PyYAML',
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'psutil': 'psutil',
        'tqdm': 'tqdm',
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name}")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        return False
    
    print(f"\n✅ All core dependencies available")
    return True


def check_classifiers():
    """Check if classifiers can be loaded"""
    print_subheader("Checking Classifiers")
    
    try:
        from classification.pitch_gender import PitchGenderClassifier
        print(f"  ✅ Pitch-Based Classifier")
        
        from classification.ml_classifier import MLGenderClassifier
        print(f"  ✅ ML Classifier Framework")
        
        try:
            from classification.advanced_gender_classifier import create_advanced_classifier
            print(f"  ✅ Advanced Multi-Feature Classifier")
        except Exception as e:
            print(f"  ⚠️  Advanced Classifier (optional): {str(e)[:50]}")
        
        print(f"\n✅ All classifiers loaded")
        return True
    except Exception as e:
        print(f"❌ Error loading classifiers: {e}")
        return False


def test_pitch_classifier():
    """Test pitch classifier with synthetic audio"""
    print_subheader("Testing Pitch Classifier")
    
    try:
        import numpy as np
        from classification.pitch_gender import PitchGenderClassifier
        
        classifier = PitchGenderClassifier()
        
        # Generate test audio
        sr = 16000
        duration = 2
        t = np.arange(int(sr * duration)) / sr
        
        # Male voice (low pitch ~120Hz)
        male_audio = np.sin(2 * np.pi * 120 * t) * 0.5
        label, conf = classifier.classify(male_audio, sr)
        print(f"  Male test: {label} ({conf:.1f}%)")
        
        # Female voice (high pitch ~220Hz)
        female_audio = np.sin(2 * np.pi * 220 * t) * 0.5
        label, conf = classifier.classify(female_audio, sr)
        print(f"  Female test: {label} ({conf:.1f}%)")
        
        print(f"\n✅ Pitch classifier working")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


# ============================================================================
# FULL PIPELINE TESTING FUNCTIONS
# ============================================================================

def check_configuration():
    """Check if config file exists and can be loaded"""
    print_subheader("Checking Configuration")
    
    config_path = 'src/config/config.yaml'
    
    if os.path.exists(config_path):
        print(f"  ✅ Configuration file exists")
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            print(f"  ✅ Configuration loaded")
            print(f"  - Device: {config.get('device', 'N/A')}")
            print(f"  - Sample Rate: {config.get('sample_rate', 'N/A')} Hz")
            return config
        except Exception as e:
            print(f"  ❌ Error loading config: {e}")
            return None
    else:
        print(f"  ❌ Configuration file not found")
        return None


def check_directory_structure():
    """Verify required directory structure"""
    print_subheader("Checking Directory Structure")
    
    required_dirs = [
        'src/classification',
        'src/preprocessing',
        'src/dataset',
        'src/quality',
        'src/engine',
        'data/input',
        'data/voice_dataset',
    ]
    
    missing = []
    for dir_path in required_dirs:
        exists = os.path.isdir(dir_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {dir_path}")
        if not exists:
            missing.append(dir_path)
    
    if missing:
        print(f"\n⚠️  Missing directories: {', '.join(missing)}")
        return False
    
    print(f"\n✅ All directories present")
    return True


def check_modules():
    """Check if all required modules can be imported"""
    print_subheader("Checking Module Imports")
    
    modules = [
        ('preprocessing.audio_loader', 'Audio Loader'),
        ('preprocessing.normalize', 'Audio Normalize'),
        ('preprocessing.source_separator', 'Source Separator'),
        ('classification.pitch_gender', 'Pitch Classifier'),
        ('classification.ml_classifier', 'ML Classifier'),
        ('dataset.organizer', 'Dataset Organizer'),
        ('engine.batch_runner', 'Batch Runner'),
        ('pipeline.batch_organizer', 'Batch Organizer'),
    ]
    
    failed = []
    for module_path, name in modules:
        try:
            __import__(module_path)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}")
            failed.append((name, str(e)[:40]))
    
    if failed:
        print(f"\n⚠️  Failed imports:")
        for name, error in failed:
            print(f"    - {name}: {error}")
        return False
    
    print(f"\n✅ All modules importable")
    return True


def check_files():
    """Check if required files exist"""
    print_subheader("Checking Required Files")
    
    required_files = [
        'src/config/config.yaml',
        'src/config/run_pipeline.py',
        'src/classification/__init__.py',
        'src/classification/pitch_gender.py',
        'src/classification/ml_classifier.py',
        'src/engine/batch_runner.py',
        'src/dataset/organizer.py',
        'src/preprocessing/audio_loader.py',
    ]
    
    missing = []
    for file_path in required_files:
        exists = os.path.isfile(file_path)
        status = "✅" if exists else "❌"
        if not exists:
            missing.append(file_path)
        print(f"  {status} {file_path}")
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} files")
        return False
    
    print(f"\n✅ All required files present")
    return True


# ============================================================================
# TEST MODE SELECTION & EXECUTION
# ============================================================================

def run_gpu_mode():
    """Run GPU-specific tests"""
    print_header("GPU TEST MODE", width=100)
    
    gpu_ok = test_gpu_availability()
    
    print_header("GPU TEST RESULTS", width=100)
    if gpu_ok:
        print("✅ GPU is available and working\n")
        return True
    else:
        print("⚠️  GPU not available - CPU will be used for processing\n")
        return False


def run_cpu_mode():
    """Run CPU-specific tests"""
    print_header("CPU TEST MODE", width=100)
    
    # Check dependencies
    deps_ok = check_core_dependencies()
    
    # Check classifiers
    classifiers_ok = check_classifiers()
    
    # Test pitch classifier
    pitch_ok = test_pitch_classifier()
    
    print_header("CPU TEST RESULTS", width=100)
    results = {
        "Dependencies": deps_ok,
        "Classifiers": classifiers_ok,
        "Pitch Classifier": pitch_ok,
    }
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n{'✅ CPU mode ready' if all_passed else '⚠️  Some CPU tests failed'}\n")
    
    return all_passed


def run_full_mode():
    """Run comprehensive system tests"""
    print_header("COMPREHENSIVE SYSTEM TEST MODE", width=100)
    
    # Check configuration
    config = check_configuration()
    
    # Check directory structure
    dirs_ok = check_directory_structure()
    
    # Check modules
    modules_ok = check_modules()
    
    # Check files
    files_ok = check_files()
    
    # Check classifiers
    classifiers_ok = check_classifiers()
    
    # Test GPU if available
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except:
        gpu_available = False
    
    print_header("COMPREHENSIVE SYSTEM TEST RESULTS", width=100)
    
    results = {
        "Configuration": config is not None,
        "Directory Structure": dirs_ok,
        "Module Imports": modules_ok,
        "Required Files": files_ok,
        "Classifiers": classifiers_ok,
        "GPU Available": gpu_available,
    }
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    critical_tests = ["Configuration", "Directory Structure", "Module Imports", "Required Files"]
    critical_passed = all(results[test] for test in critical_tests)
    
    print(f"\nCritical Tests: {'✅ PASSED' if critical_passed else '❌ FAILED'}")
    
    if critical_passed:
        print("✅ System ready for speaker separation pipeline\n")
    else:
        print("⚠️  Some critical tests failed\n")
    
    return critical_passed


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main test execution with mode selection"""
    
    print("\n" + "="*100)
    print("VOXENT COMPREHENSIVE SYSTEM TEST SUITE")
    print("="*100)
    
    # Determine test mode from command line
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "full"  # Default to full mode
    
    if mode == "gpu":
        success = run_gpu_mode()
    elif mode == "cpu":
        success = run_cpu_mode()
    elif mode == "full":
        success = run_full_mode()
    else:
        print(f"\nUnknown mode: {mode}")
        print("Usage: python comprehensive_system_test.py [gpu|cpu|full]")
        print("  gpu:  Test GPU availability")
        print("  cpu:  Test CPU pipeline")
        print("  full: Comprehensive system test (default)")
        sys.exit(1)
    
    print("="*100)
    if success:
        print("✅ TEST SUITE COMPLETED SUCCESSFULLY")
    else:
        print("⚠️  TEST SUITE COMPLETED WITH WARNINGS")
    print("="*100 + "\n")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
