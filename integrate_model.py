#!/usr/bin/env python3
"""
Model Integration Helper Script
This script helps you integrate your trained model into the web application.
"""

import os
import sys
import shutil

def check_model_files():
    """Check if model files are present in the reconstruction_model directory."""
    model_dir = "reconstruction_model"
    required_files = [
        'config.json',
        'model.safetensors', 
        'tokenizer_config.json',
        'tokenizer.json'
    ]
    
    print("🔍 Checking for model files...")
    
    if not os.path.exists(model_dir):
        print(f"❌ Model directory '{model_dir}' not found.")
        print("   Please create it and place your model files there.")
        return False
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n📋 Missing files: {missing_files}")
        print("   Please add these files to the reconstruction_model/ directory")
        return False
    
    print("\n✅ All required model files found!")
    return True

def update_model_service():
    """Update model_service.py to enable the actual model."""
    print("\n🔧 Updating model_service.py...")
    
    service_file = "model_service.py"
    backup_file = "model_service.py.backup"
    
    # Create backup
    shutil.copy2(service_file, backup_file)
    print(f"📋 Created backup: {backup_file}")
    
    with open(service_file, 'r') as f:
        content = f.read()
    
    # Enable imports
    content = content.replace(
        "# import torch\n# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM",
        "import torch\nfrom transformers import AutoTokenizer, AutoModelForSeq2SeqLM"
    )
    
    # Enable device initialization
    content = content.replace(
        "        # self.device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")",
        "        self.device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")"
    )
    
    # Enable model loading
    model_loading_commented = """            # logging.info(f"Loading model from {self.model_path}")
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            # self.model.to(self.device)
            # self.model.eval()"""
    
    model_loading_enabled = """            logging.info(f"Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()"""
    
    content = content.replace(model_loading_commented, model_loading_enabled)
    
    # Enable inference code
    inference_commented = """            # inputs = self.tokenizer(
            #     damaged_text, 
            #     return_tensors="pt", 
            #     truncation=True, 
            #     padding=True
            # )
            
            # # Move inputs to the same device as model
            # device = next(self.model.parameters()).device
            # inputs = {key: val.to(device) for key, val in inputs.items()}
            
            # output = self.model.generate(**inputs, max_new_tokens=50)
            # reconstructed = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # return reconstructed"""
    
    inference_enabled = """            inputs = self.tokenizer(
                damaged_text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            )
            
            # Move inputs to the same device as model
            device = next(self.model.parameters()).device
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            output = self.model.generate(**inputs, max_new_tokens=50)
            reconstructed = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            return reconstructed"""
    
    content = content.replace(inference_commented, inference_enabled)
    
    # Remove the demo mode fallback
    content = content.replace(
        '            # Placeholder return for when you integrate your model\n            return damaged_text + " [Model integration needed]"',
        '            # This line should not be reached if model is loaded properly\n            return damaged_text'
    )
    
    with open(service_file, 'w') as f:
        f.write(content)
    
    print("✅ Updated model_service.py")
    print("   - Enabled PyTorch and transformers imports")
    print("   - Enabled device initialization") 
    print("   - Enabled model loading")
    print("   - Enabled inference code")

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        import torch
        import transformers
        print("✅ PyTorch and transformers are already installed")
        return True
    except ImportError:
        print("❌ Missing dependencies. Please install them:")
        print("   pip install torch transformers")
        print("   or")
        print("   uv add torch transformers")
        return False

def main():
    """Main integration process."""
    print("🚀 Historical Text Reconstruction - Model Integration Helper")
    print("=" * 60)
    
    # Check dependencies
    if not install_dependencies():
        print("\n⚠️  Please install dependencies first, then run this script again.")
        return
    
    # Check model files
    if not check_model_files():
        print("\n⚠️  Please add your model files first, then run this script again.")
        return
    
    # Update code
    update_model_service()
    
    print("\n🎉 Integration complete!")
    print("   Your model is now integrated into the web application.")
    print("   Restart the server to see your model in action:")
    print("   - The demo mode will be replaced with your actual model")
    print("   - Text reconstruction will use your trained model")
    print("   - Highlighting will show actual model predictions")
    
    print("\n💡 Next steps:")
    print("   1. Restart the web application")
    print("   2. Test with your historical texts")
    print("   3. Adjust model parameters if needed")

if __name__ == "__main__":
    main()