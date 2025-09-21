#!/usr/bin/env python3
"""
Real Model Interface for Trained CodeT5 Model
Tích hợp model đã train từ Kaggle vào web demo
"""

import os
import sys
import time
import torch
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import transformers
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    logging
)

# Tắt warning của transformers
logging.set_verbosity_error()

class RealModelInterface:
    """Interface cho model CodeT5 đã train"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Khởi tạo model interface
        
        Args:
            model_path: Đường dẫn đến thư mục chứa model
        """
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mặc định sử dụng model_trained folder (model đã train từ Kaggle)
        if model_path is None:
            project_root = Path(__file__).parent.parent
            model_path = project_root / "model_trained"
        
        self.model_path = Path(model_path)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load model và tokenizer"""
        try:
            print(f"🔄 Loading model from {self.model_path}...")
            start_time = time.time()
            
            # Check if model path exists
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model path not found: {self.model_path}")
            
            # Load tokenizer
            print("📦 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            
            # Load model - ưu tiên safetensors nếu có
            print("🤖 Loading fine-tuned CodeT5 model...")
            
            # Check for safetensors file
            safetensors_path = self.model_path / "model.safetensors"
            pytorch_bin_path = self.model_path / "pytorch_model.bin"
            
            if safetensors_path.exists():
                print("✅ Found model.safetensors - using safetensors format")
            elif pytorch_bin_path.exists():
                print("✅ Found pytorch_model.bin - using PyTorch format")
            else:
                raise FileNotFoundError("No model weights file found (model.safetensors or pytorch_model.bin)")
            
            # Load với auto-detect format
            self.model = T5ForConditionalGeneration.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float32,  # Sử dụng float32 cho CPU
                local_files_only=True,  # Chỉ dùng local files
                use_safetensors=safetensors_path.exists()  # Auto use safetensors if available
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            load_time = time.time() - start_time
            print(f"✅ Model loaded successfully in {load_time:.1f} seconds!")
            print(f"📍 Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_test_case(
        self, 
        user_story: str, 
        test_type: str = "functional",
        max_length: int = 180,
        temperature: float = 0.7,
        num_beams: int = 4,
        **kwargs
    ) -> Dict:
        """
        Generate test case từ user story
        
        Args:
            user_story: User story cần generate test case
            test_type: Loại test (functional, security, performance, compliance)
            max_length: Độ dài tối đa của output
            temperature: Điều chỉnh creativity (0.1 = conservative, 1.0 = creative)
            num_beams: Số beam cho beam search
            
        Returns:
            Dictionary chứa test case và metadata
        """
        if not self.model or not self.tokenizer:
            return {
                'success': False,
                'error': 'Model not loaded'
            }
        
        try:
            start_time = time.time()
            
            # Chuẩn bị input với test type
            input_text = self._prepare_input(user_story, test_type)
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                max_length=180,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate với no_grad để tiết kiệm memory
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    early_stopping=True,
                    do_sample=False  # Deterministic generation
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Format output theo Gherkin
            formatted_text = self._format_gherkin(generated_text)
            
            generation_time = time.time() - start_time
            
            return {
                'success': True,
                'test_case': formatted_text,
                'metadata': {
                    'test_type': test_type,
                    'confidence': self._calculate_confidence(generated_text),
                    'generation_time': generation_time,
                    'mode': 'production',
                    'model': 'My Fine-tuned Model (8 epochs, Kaggle)',
                    'device': str(self.device)
                }
            }
            
        except Exception as e:
            print(f"❌ Error generating test case: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _prepare_input(self, user_story: str, test_type: str) -> str:
        """
        Chuẩn bị input text với context phù hợp
        """
        # Thêm prefix theo test type để guide model
        prefixes = {
            'functional': 'Generate functional test case for:',
            'security': 'Generate security test case for:',
            'performance': 'Generate performance test case for:',
            'compliance': 'Generate compliance test case for:'
        }
        
        prefix = prefixes.get(test_type, 'Generate test case for:')
        return f"{prefix} {user_story}"
    
    def _format_gherkin(self, text: str) -> str:
        """
        Format output theo chuẩn Gherkin với style đẹp hơn
        """
        # Đảm bảo có cấu trúc Gherkin cơ bản
        if 'Scenario:' not in text:
            text = f"Scenario: {text}"
        
        # Format các keywords với emoji và style
        keyword_map = {
            'Feature:': '\n🎯 **Feature:**',
            'Scenario:': '\n📝 **Scenario:**',
            'Background:': '\n📋 **Background:**',
            'Given': '\n  ✅ **Given**',
            'When': '\n  ▶️ **When**',
            'Then': '\n  ✔️ **Then**',
            'And': '\n    **And**',
            'But': '\n    **But**'
        }
        
        # Replace keywords
        for old_kw, new_kw in keyword_map.items():
            text = text.replace(old_kw, new_kw)
        
        # Clean up and format
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                # Format test IDs
                if 'Test ID:' in line or 'TC-' in line:
                    line = f"🔖 {line}"
                # Format priority
                elif 'Priority:' in line:
                    if 'Critical' in line:
                        line = line.replace('Critical', '🔴 Critical')
                    elif 'High' in line:
                        line = line.replace('High', '🟠 High')
                    elif 'Medium' in line:
                        line = line.replace('Medium', '🟡 Medium')
                    elif 'Low' in line:
                        line = line.replace('Low', '🟢 Low')
                # Format preconditions
                elif 'Preconditions:' in line:
                    line = f"📌 **{line}**"
                
                lines.append(line)
        
        formatted_text = '\n'.join(lines)
        
        # Add header with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"🧪 **Test Case Generated**\n⏰ {timestamp}\n{'─'*40}"
        
        if not formatted_text.startswith('🧪'):
            formatted_text = f"{header}\n{formatted_text}"
        
        # Add footer
        footer = f"\n{'─'*40}\n✨ Generated by My Trained AI Model (8000+ samples)"
        formatted_text += footer
        
        return formatted_text
    
    def _calculate_confidence(self, text: str) -> float:
        """
        Tính confidence score dựa trên chất lượng output
        """
        score = 0.5  # Base score
        
        # Check for Gherkin keywords
        gherkin_keywords = ['Given', 'When', 'Then', 'Scenario']
        for keyword in gherkin_keywords:
            if keyword in text:
                score += 0.1
        
        # Check for completeness
        if len(text) > 50:
            score += 0.1
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def get_model_info(self) -> str:
        """Get model information"""
        if self.model:
            param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
            return f"My Trained Model Loaded - {param_count:.1f}M parameters (Fine-tuned on Kaggle, 8 epochs) on {self.device}"
        return "Model not loaded"
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None


# Test function
if __name__ == "__main__":
    print("🧪 Testing Real Model Interface...")
    
    # Initialize model
    model = RealModelInterface()
    
    # Test generation
    test_story = "As a user, I want to login to mobile banking using fingerprint"
    result = model.generate_test_case(test_story, test_type="security")
    
    if result['success']:
        print("\n✅ Generated Test Case:")
        print(result['test_case'])
        print("\n📊 Metadata:")
        for key, value in result['metadata'].items():
            print(f"  {key}: {value}")
    else:
        print(f"❌ Error: {result['error']}")
