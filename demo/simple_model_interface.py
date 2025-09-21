#!/usr/bin/env python3
"""
Simple Model Interface for Streamlit App
Sử dụng model CodeT5 đã train từ Kaggle (model_trained)
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src" / "utils"))

class SimpleModelInterface:
    """Interface để load model CodeT5 đã train"""
    
    def __init__(self, preload_model=False):
        self.model_interface = None
        self.initialized = False
        self.model = None  # Add compatibility attribute
        
        if preload_model:
            self._force_load()
        
    def _lazy_load(self):
        """Lazy load model interface khi cần thiết"""
        if not self.initialized:
            try:
                # Load model đã train
                from model_interface_real import RealModelInterface
                print("🔄 Loading trained CodeT5 model...")
                self.model_interface = RealModelInterface()
                self.model = self.model_interface.model if self.model_interface else None
                self.initialized = True
                return True
            except Exception as e:
                print(f"❌ Error loading trained model: {e}")
                # Không có fallback - chỉ dùng model đã train
                self.model = None
                self.initialized = True
                return False
        return self.model_interface is not None
    
    def _force_load(self):
        """Force load model immediately (for preload)"""
        import time
        start_time = time.time()
        print("🔄 Loading trained model...")
        
        try:
            # Load model đã train từ Kaggle
            from model_interface_real import RealModelInterface
            print("📦 Loading fine-tuned CodeT5 model from model_trained/...")
            self.model_interface = RealModelInterface()
            self.model = self.model_interface.model if self.model_interface else None
            self.initialized = True
            
            load_time = time.time() - start_time
            print(f"✅ Trained model loaded successfully in {load_time:.1f} seconds!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load trained model: {e}")
            print("Please ensure model_trained folder exists with model files.")
            self.model = None
            self.initialized = True
            return False
    
    def generate_test_case(self, user_story: str, test_type: str = "functional", **kwargs) -> dict:
        """Generate test case với fallback - accepts all parameters"""
        if self._lazy_load() and self.model_interface:
            try:
                return self.model_interface.generate_test_case(user_story, test_type, **kwargs)
            except Exception as e:
                print(f"Error generating with model: {e}")
                return self._demo_test_case(user_story, test_type)
        else:
            return self._demo_test_case(user_story, test_type)
    
    @property
    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.model_interface is not None
    
    def get_model_info(self):
        """Get model information"""
        if self.initialized and self.model_interface is not None:
            return "CodeT5 Model Loaded Successfully"
        else:
            return "Demo Mode - Model Not Available"
    
    def _demo_test_case(self, user_story: str, test_type: str) -> dict:
        """Generate demo test case khi model không available"""
        templates = {
            'functional': f"""
**Scenario:** {user_story}

**Given** the user is on the mobile banking application  
**When** the user performs the required action  
**Then** the system should respond appropriately  
**And** all functional requirements are met
            """,
            'security': f"""
**Scenario:** Security testing for {user_story}

**Given** the user has valid credentials  
**When** the user attempts secure operations  
**Then** the system should enforce security policies  
**And** sensitive data should be protected
            """,
            'performance': f"""
**Scenario:** Performance testing for {user_story}

**Given** the system is under normal load  
**When** the user performs the action  
**Then** the response time should be < 3 seconds  
**And** system resources should be within limits
            """,
            'compliance': f"""
**Scenario:** Compliance testing for {user_story}

**Given** the system follows banking regulations  
**When** the user performs the transaction  
**Then** audit logs should be created  
**And** compliance requirements should be met
            """
        }
        
        return {
            'success': True,
            'test_case': templates.get(test_type, templates['functional']).strip(),
            'metadata': {
                'test_type': test_type,
                'confidence': 0.5,
                'generation_time': 0.1,
                'mode': 'demo'
            }
        }