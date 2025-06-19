#!/usr/bin/env python3
"""
Test script to verify your model integration works correctly.
Run this after placing your model files in reconstruction_model/
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_model_files():
    """Check if all required model files are present."""
    print("=" * 60)
    print("TESTING MODEL FILE PRESENCE")
    print("=" * 60)
    
    model_dir = "reconstruction_model"
    required_files = [
        'config.json',
        'model.safetensors', 
        'tokenizer_config.json',
        'tokenizer.json'
    ]
    
    if not os.path.exists(model_dir):
        print(f"❌ Model directory '{model_dir}' not found")
        return False
    
    all_present = True
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} - MISSING")
            all_present = False
    
    return all_present

def test_dependencies():
    """Check if PyTorch and transformers are available."""
    print("\n" + "=" * 60)
    print("TESTING DEPENDENCIES")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        return False
    
    return True

def test_model_loading():
    """Test loading your model."""
    print("\n" + "=" * 60)
    print("TESTING MODEL LOADING")
    print("=" * 60)
    
    try:
        from model_service import TextReconstructionService
        
        print("Creating TextReconstructionService...")
        service = TextReconstructionService()
        
        if service.model_loaded:
            print("✅ Your trained model loaded successfully!")
            print(f"   Model type: {type(service.model).__name__}")
            print(f"   Tokenizer type: {type(service.tokenizer).__name__}")
            return service
        else:
            print("❌ Model not loaded (check logs above)")
            return None
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_reconstruction(service):
    """Test text reconstruction with your model."""
    print("\n" + "=" * 60)
    print("TESTING TEXT RECONSTRUCTION")
    print("=" * 60)
    
    test_texts = [
        "Zаmfіrаkе. nꙋ'lꙋ꙼ аltъ",
        "In nꙄmine P꙲tris et F꙲lii",
        "Ѕъ nꙋ'lꙋ꙼ аltъ ԁаtъ"
    ]
    
    for text in test_texts:
        print(f"\nInput: {text}")
        try:
            result = service.reconstruct(text)
            print(f"Output: {result}")
            
            if service.model_loaded:
                print("✅ Used your trained model")
            else:
                print("⚠️  Used demo mode (model not loaded)")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def test_translation(service):
    """Test translation functionality."""
    print("\n" + "=" * 60)
    print("TESTING TRANSLATION")
    print("=" * 60)
    
    test_text = "Съ nulla alta data"
    print(f"Testing translation for: {test_text}")
    
    try:
        translation = service.translate_with_gemini(test_text)
        print(f"Translation: {translation}")
        
        if service.translation_available:
            print("✅ Gemini translation available")
        else:
            print("⚠️  Translation not available (API key missing)")
            
    except Exception as e:
        print(f"❌ Translation error: {e}")

def main():
    """Run all tests."""
    print("HISTORICAL TEXT RECONSTRUCTION - MODEL INTEGRATION TEST")
    
    # Test 1: Check files
    files_ok = test_model_files()
    
    # Test 2: Check dependencies
    deps_ok = test_dependencies()
    
    if not files_ok:
        print("\n❌ MISSING MODEL FILES")
        print("Please place your model files in reconstruction_model/:")
        print("- config.json")
        print("- model.safetensors")
        print("- tokenizer_config.json") 
        print("- tokenizer.json")
        return
    
    if not deps_ok:
        print("\n❌ MISSING DEPENDENCIES")
        print("Please install: pip install torch transformers")
        return
    
    # Test 3: Load model
    service = test_model_loading()
    if service is None:
        print("\n❌ MODEL LOADING FAILED")
        return
    
    # Test 4: Test reconstruction
    test_reconstruction(service)
    
    # Test 5: Test translation
    test_translation(service)
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 60)
    
    if service.model_loaded:
        print("🎉 SUCCESS: Your model is fully integrated!")
        print("The web application will now use your trained model.")
    else:
        print("⚠️  WARNING: Using demo mode")
        print("Your model files are present but couldn't be loaded.")

if __name__ == "__main__":
    main()