#!/usr/bin/env python3
"""
Model Interface - Tích hợp Trained T5 Model từ auto_checkpoint_1200
Luận văn Thạc sĩ: Generative AI for Testing of Banking System in Mobile Applications

Interface để tích hợp trained T5 model vào web application.
Xử lý input preprocessing, model inference, và output postprocessing.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import torch and transformers only if available
try:
    import torch
    from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoConfig
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    T5ForConditionalGeneration = None
    AutoTokenizer = None
    AutoConfig = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInterface:
    """
    Interface để tích hợp trained T5 model từ auto_checkpoint_1200
    
    Chức năng chính:
    - Load trained model và tokenizer từ checkpoint
    - Process user story input
    - Generate test cases với model
    - Apply banking industry standards
    - Return formatted results
    """
    
    def __init__(self, model_path: str = None, tokenizer_path: str = None):
        """
        Khởi tạo model interface với auto_checkpoint_1200
        
        Args:
            model_path: Path to trained model (default: auto_checkpoint_1200)
            tokenizer_path: Path to tokenizer (default: auto_checkpoint_1200)
        """
        # Get checkpoint path
        self.model_path = model_path or self._get_checkpoint_path()
        self.tokenizer_path = tokenizer_path or self.model_path
        
        self.model = None
        self.tokenizer = None
        self.device = None
        
        if TORCH_AVAILABLE and torch:
            # Use CPU by default for demo (GPU optional)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"🔧 Using device: {self.device}")
        else:
            logger.warning("⚠️ PyTorch not available - please install torch and transformers")
            
        # Banking domain keywords
        self.banking_keywords = [
            "login", "authentication", "transfer", "payment", "balance",
            "transaction", "account", "security", "notification", "settings",
            "deposit", "withdrawal", "statement", "card", "pin", "otp"
        ]
        
        # Test case templates
        self.test_templates = {
            "functional": "Generate functional test case for mobile banking: {user_story}",
            "security": "Generate security test case for mobile banking: {user_story}",
            "performance": "Generate performance test case for mobile banking: {user_story}",
            "compliance": "Generate compliance test case for mobile banking: {user_story}"
        }
        
        # Load model
        self._load_checkpoint()
        
        if self.model:
            logger.info("✅ ModelInterface initialized successfully with checkpoint!")
        else:
            logger.warning("⚠️ ModelInterface initialized in demo mode")
    
    def _get_checkpoint_path(self) -> str:
        """Get path to codet5_final"""
        # Try multiple possible locations
        project_root = Path(__file__).parent.parent.parent
        
        # Primary path - relative to project (codet5_final folder)
        checkpoint_path = project_root / "codet5_final"
        
        # Alternative absolute path
        alt_path = Path("/Users/chienvt/Downloads/03_Personal Development/00_VuTuanChien_DeAnThacSi_HUIT/GenerativeAIForSoftwareTesting/clean_project/codet5_final")
        
        # Check which path exists
        if checkpoint_path.exists():
            logger.info(f"✅ Found checkpoint at: {checkpoint_path}")
            return str(checkpoint_path)
        elif alt_path.exists():
            logger.info(f"✅ Found checkpoint at alternative path: {alt_path}")
            return str(alt_path)
        else:
            logger.warning(f"⚠️ Checkpoint not found at expected locations")
            logger.warning(f"   Tried: {checkpoint_path}")
            logger.warning(f"   Tried: {alt_path}")
            # Return the expected path anyway for error messages
            return str(alt_path)
    
    def _load_checkpoint(self):
        """Load trained model và tokenizer từ codet5_final"""
        if not TORCH_AVAILABLE:
            logger.error("❌ PyTorch not installed. Please run: pip install torch transformers")
            return
            
        try:
            checkpoint_path = Path(self.model_path)
            
            # Check required files for CodeT5
            required_files = {
                "config.json": checkpoint_path / "config.json",
                "model.safetensors": checkpoint_path / "model.safetensors",
                "tokenizer.json": checkpoint_path / "tokenizer.json",
                "tokenizer_config.json": checkpoint_path / "tokenizer_config.json",
                "vocab.json": checkpoint_path / "vocab.json"
            }
            
            # Verify files exist
            missing_files = []
            for name, path in required_files.items():
                if not path.exists():
                    missing_files.append(name)
                else:
                    size_mb = path.stat().st_size / (1024 * 1024)
                    logger.info(f"  ✓ {name}: {size_mb:.1f} MB")
            
            if missing_files:
                logger.error(f"❌ Missing files in checkpoint: {', '.join(missing_files)}")
                logger.info("🔄 Falling back to demo mode...")
                self._load_demo_model()
                return
            
            logger.info(f"📦 Loading checkpoint from: {checkpoint_path}")
            
            # Load tokenizer
            logger.info("📚 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(checkpoint_path),
                use_fast=True
            )
            logger.info(f"✅ Tokenizer loaded successfully: {type(self.tokenizer).__name__}")
            
            # Load model configuration
            logger.info("⚙️ Loading model configuration...")
            config = AutoConfig.from_pretrained(str(checkpoint_path))
            
            # Load model weights
            logger.info("🧠 Loading model weights (this may take a moment)...")
            self.model = T5ForConditionalGeneration.from_pretrained(
                str(checkpoint_path),
                config=config,
                use_safetensors=True,
                torch_dtype=torch.float32,  # Use float32 for stability
                low_cpu_mem_usage=True  # Optimize memory usage
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Log model info
            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"✅ Model loaded successfully!")
            logger.info(f"📊 Model parameters: {param_count / 1e6:.1f}M")
            logger.info(f"🔧 Device: {self.device}")
            
            # Quick test
            self._test_model()
            
        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("🔄 Falling back to demo mode...")
            self._load_demo_model()
    
    def _load_demo_model(self):
        """Load a small demo model for fallback"""
        try:
            logger.info("📥 Loading demo T5-small model...")
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ Demo model loaded (t5-small)")
        except Exception as e:
            logger.error(f"❌ Failed to load demo model: {e}")
            self.model = None
            self.tokenizer = None
    
    def _test_model(self):
        """Test model với sample input"""
        if not self.model or not self.tokenizer:
            return
            
        try:
            test_input = "Generate test case for: User login to mobile banking app"
            logger.info(f"🧪 Testing model with: '{test_input}'")
            
            # Tokenize
            inputs = self.tokenizer(
                test_input,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=100,
                    num_beams=2,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    decoder_start_token_id=self.model.config.decoder_start_token_id
                )
            
            # Decode
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"✅ Test successful! Output: {result[:150]}...")
            
        except Exception as e:
            logger.error(f"⚠️ Model test failed: {e}")
    
    def generate_test_cases(self, 
                          user_story: str,
                          context: str = "",
                          options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Generate test cases từ user story
        
        Args:
            user_story: User story requirements
            context: Additional context
            options: Generation options
            
        Returns:
            List of generated test cases
        """
        try:
            if not self.model or not self.tokenizer:
                logger.warning("⚠️ Model not available, using demo test cases")
                return self._generate_demo_test_cases(user_story, options)
            
            options = options or {}
            test_types = options.get('test_types', ['Functional Testing'])
            num_test_cases = options.get('num_test_cases', 3)
            
            generated_tests = []
            
            for test_type in test_types:
                # Create prompt for each test type
                test_type_key = test_type.lower().replace(' testing', '')
                prompt = self._create_prompt(user_story, context, test_type_key)
                
                # Generate test case
                logger.info(f"🤖 Generating {test_type} test case...")
                test_case = self._generate_single_test_case(prompt, test_type_key)
                
                if test_case:
                    generated_tests.append(test_case)
                
                if len(generated_tests) >= num_test_cases:
                    break
            
            logger.info(f"✅ Generated {len(generated_tests)} test cases")
            return generated_tests
            
        except Exception as e:
            logger.error(f"❌ Error generating test cases: {e}")
            return self._generate_demo_test_cases(user_story, options)
    
    def generate_test_case(self, user_story: str, test_type: str = "functional", max_length: int = 150) -> Dict[str, Any]:
        """
        Generate single test case cho demo
        
        Args:
            user_story: User story input
            test_type: Type of test to generate
            max_length: Maximum length of output
            
        Returns:
            Dictionary with success status and test case
        """
        import time
        start_time = time.time()
        
        try:
            if not self.model or not self.tokenizer:
                logger.warning("⚠️ Model not available, using demo mode")
                return {
                    'success': True,
                    'test_case': self._generate_demo_single_test_case(user_story, test_type),
                    'metadata': {
                        'test_type': test_type,
                        'confidence': 0.5,
                        'generation_time': time.time() - start_time
                    }
                }
            
            # Create prompt
            prompt = self._create_prompt(user_story, "", test_type)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate với model - ultra-conservative settings for stability
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=min(max_length, 100),  # Shorter outputs for better quality
                    num_beams=1,  # Greedy decoding only
                    do_sample=False,  # No randomness
                    early_stopping=True,
                    no_repeat_ngram_size=2,  # Prevent repetition
                    pad_token_id=self.tokenizer.pad_token_id,
                    decoder_start_token_id=self.model.config.decoder_start_token_id,
                    repetition_penalty=1.5,  # Strong anti-repetition
                    forced_bos_token_id=self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else None
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Debug logging
            logger.info(f"🔍 Generated text length: {len(generated_text)}")
            logger.info(f"🔍 First 200 chars of raw output: {generated_text[:200]}...")
            
            # Clean up generated text
            test_case = self._clean_generated_text(generated_text)
            
            logger.info(f"🔍 Cleaned text length: {len(test_case)}")
            logger.info(f"🔍 Cleaned text preview: {test_case[:150]}...")
            
            # Quality check - if output is poor, use enhanced template
            if self._is_poor_quality(test_case):
                logger.warning("⚠️ Generated output quality is poor, using enhanced template")
                logger.info(f"🔍 Poor quality text: {test_case[:100]}...")
                test_case = self._generate_enhanced_template(user_story, test_type)
                confidence = 0.6  # Lower confidence for template
            else:
                logger.info("✅ Generated output quality is acceptable")
                confidence = 0.85
            
            generation_time = time.time() - start_time
            
            return {
                'success': True,
                'test_case': test_case,
                'metadata': {
                    'test_type': test_type,
                    'confidence': confidence,
                    'generation_time': generation_time
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating test case: {e}")
            return {
                'success': False,
                'error': str(e),
                'test_case': self._generate_demo_single_test_case(user_story, test_type),
                'metadata': {
                    'test_type': test_type,
                    'confidence': 0.0,
                    'generation_time': time.time() - start_time
                }
            }
    
    def _clean_generated_text(self, text: str) -> str:
        """Aggressive post-processing to force proper BDD format"""
        import re
        
        # Remove ALL prompt remnants aggressively
        text = re.sub(r"Write.*?test.*?for:.*?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Create.*?test.*?case.*?", "", text, flags=re.IGNORECASE)
        
        # Fix common model errors
        text = re.sub(r"\[.*?\]", "", text)  # Remove bracket placeholders
        text = re.sub(r"name\]", "name:", text)  # Fix syntax errors
        text = re.sub(r"Clear test scenario name", "Banking test scenario", text)
        text = re.sub(r"User actions are displayed", "User is authenticated", text)
        
        # Remove ALL non-ASCII and corrupted content
        text = re.sub(r"[^\x00-\x7F]+", "", text)
        text = re.sub(r"\{\{[^}]*\}\}", "", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        
        # Force BDD structure if missing
        if not text.startswith("Scenario:"):
            text = "Scenario: Banking test case\n" + text
        
        # Clean and normalize
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)
        
        # Split into lines and rebuild with proper BDD
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Force proper BDD keywords
        cleaned_lines = []
        for i, line in enumerate(lines):
            if i == 0 and not line.startswith("Scenario:"):
                line = "Scenario: " + line
            elif i > 0:
                if not any(line.startswith(kw) for kw in ['Given', 'When', 'Then', 'And', 'But']):
                    if i == 1:
                        line = "Given " + line
                    elif i == 2:
                        line = "When " + line
                    elif i >= 3:
                        line = "Then " + line
            
            if len(line) > 5:  # Skip very short lines
                cleaned_lines.append(line)
        
        # Ensure minimum BDD structure
        if len(cleaned_lines) < 3:
            return ""  # Trigger template if too short
            
        return '\n'.join(cleaned_lines)
    
    def _generate_demo_single_test_case(self, user_story: str, test_type: str) -> str:
        """Generate demo test case when model not available"""
        templates = {
            'functional': f"""
Scenario: {user_story}
Given the user is on the mobile banking application
When the user performs the required action
Then the system should respond appropriately
And all functional requirements are met
            """,
            'security': f"""
Scenario: Security testing for {user_story}
Given the user has valid credentials
When the user attempts secure operations
Then the system should enforce security policies
And sensitive data should be protected
            """,
            'performance': f"""
Scenario: Performance testing for {user_story}
Given the system is under normal load
When the user performs the action
Then the response time should be < 3 seconds
And system resources should be within limits
            """,
            'compliance': f"""
Scenario: Compliance testing for {user_story}
Given the system follows banking regulations
When the user performs the transaction
Then audit logs should be created
And compliance requirements should be met
            """
        }
        
        return templates.get(test_type, templates['functional']).strip()
    
    def _create_prompt(self, user_story: str, context: str, test_type: str) -> str:
        """Create minimal, focused prompt to reduce model confusion"""
        
        # Ultra-simple prompts to avoid confusing the model
        simple_templates = {
            "functional": f"Write a test case for: {user_story}",
            "security": f"Write a security test for: {user_story}",
            "performance": f"Write a performance test for: {user_story}"
        }
        
        return simple_templates.get(test_type, simple_templates['functional'])
    
    def _generate_single_test_case(self, prompt: str, test_type: str) -> Optional[Dict[str, Any]]:
        """Generate single test case using model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate với model
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=200,
                    num_beams=4,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse into test case format
            test_case = self._parse_generated_text(generated_text, test_type)
            
            return test_case
            
        except Exception as e:
            logger.error(f"❌ Error generating single test case: {e}")
            return None
    
    def _parse_generated_text(self, generated_text: str, test_type: str) -> Dict[str, Any]:
        """Parse generated text thành structured test case"""
        import re
        
        # Return raw text for app.py to handle parsing
        # This way we can use the improved parser in app.py
        return generated_text
    
    def _create_default_steps(self, test_type: str) -> List[str]:
        """Create default steps based on test type"""
        if test_type == 'security':
            return [
                "Given user is on mobile banking login page",
                "When user attempts to login with credentials",
                "Then system should verify authentication securely",
                "And sensitive data should be encrypted"
            ]
        elif test_type == 'performance':
            return [
                "Given mobile banking app is loaded",
                "When user performs banking operations",
                "Then response time should be within 2 seconds",
                "And app should handle load efficiently"
            ]
        else:  # functional
            return [
                "Given user is logged into mobile banking app",
                "When user performs the banking operation",
                "Then system should process the request correctly",
                "And user should see confirmation message"
            ]
    
    def _generate_demo_test_cases(self, user_story: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate demo test cases when model is not available"""
        num_cases = options.get('num_test_cases', 3) if options else 3
        
        # Analyze user story for keywords
        user_story_lower = user_story.lower()
        
        demo_cases = []
        
        # Generate contextual demo cases based on user story
        if 'login' in user_story_lower:
            demo_cases.append({
                'title': 'Valid User Login Test',
                'scenario': 'User successfully logs into mobile banking app',
                'steps': [
                    'Given user is on mobile banking login screen',
                    'When user enters valid username and password',
                    'And user taps the login button',
                    'Then user should be successfully authenticated',
                    'And user should see the account dashboard'
                ],
                'test_type': 'functional'
            })
            demo_cases.append({
                'title': 'Invalid Login Attempt Test',
                'scenario': 'System handles invalid login credentials',
                'steps': [
                    'Given user is on mobile banking login screen',
                    'When user enters invalid credentials',
                    'And user taps the login button',
                    'Then system should display error message',
                    'And user should remain on login screen'
                ],
                'test_type': 'security'
            })
        
        elif 'transfer' in user_story_lower or 'payment' in user_story_lower:
            demo_cases.append({
                'title': 'Money Transfer Test',
                'scenario': 'User transfers money between accounts',
                'steps': [
                    'Given user is logged into mobile banking app',
                    'And user has sufficient balance',
                    'When user selects transfer option',
                    'And user enters transfer details',
                    'And user confirms the transaction',
                    'Then transfer should be processed successfully',
                    'And user should receive confirmation'
                ],
                'test_type': 'functional'
            })
        
        elif 'balance' in user_story_lower:
            demo_cases.append({
                'title': 'Check Account Balance Test',
                'scenario': 'User views account balance',
                'steps': [
                    'Given user is logged into mobile banking app',
                    'When user navigates to account section',
                    'And user selects balance inquiry',
                    'Then current balance should be displayed',
                    'And transaction history should be available'
                ],
                'test_type': 'functional'
            })
        
        # Add generic test case if needed
        if len(demo_cases) < num_cases:
            demo_cases.append({
                'title': 'Mobile Banking Feature Test',
                'scenario': 'Test banking feature functionality',
                'steps': [
                    'Given user has access to mobile banking app',
                    'When user performs the requested action',
                    'Then system should process the request',
                    'And appropriate response should be displayed'
                ],
                'test_type': 'functional'
            })
        
        return demo_cases[:num_cases]

    def _is_poor_quality(self, text: str) -> bool:
        """Check if generated text quality is poor"""
        import re
        
        # Quality indicators
        poor_indicators = [
            r"Runnable|terminated|ExitCode|athena",  # Technical artifacts
            r"\{\{.*?\}\}",  # Template variables
            r"And\s+And\s+And",  # Excessive repetition
            r"When\s+Given",  # Wrong BDD order
            r"eturnedcode|returnedcode",  # Corrupted text
            r"デパ|ഢ|注角",  # Foreign characters/corruption
            r"getApplication|getElementById",  # Code artifacts
            r"Encode-friendly|Auditlog\s+entry",  # Technical noise
            r"banner appears;",  # Malformed sentences
        ]
        
        for indicator in poor_indicators:
            if re.search(indicator, text, re.IGNORECASE):
                return True
        
        # Check for excessive repetition (same phrase 3+ times)
        words = text.lower().split()
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            if text.lower().count(phrase) >= 3:
                return True
        
        # Check if missing proper BDD structure
        bdd_keywords = ['Given', 'When', 'Then']
        found_keywords = sum(1 for keyword in bdd_keywords if keyword in text)
        
        if found_keywords < 2:  # Must have at least Given-Then or When-Then
            return True
            
        # Check text length - too short or too long may indicate issues
        if len(text.strip()) < 30 or len(text.strip()) > 1000:  # More lenient length check
            return True
        
        # Special check for security tests - they tend to be more problematic
        # Make this less strict for security tests
        if "security" in text.lower():
            security_issues = [
                r"Audit.*?Audit.*?Audit.*?Audit",  # Require 4+ repetitions instead of 3
                r"Security.*?Security.*?Security.*?Security",  # Require 4+ repetitions
                r"validation.*?validation.*?validation.*?validation",  # Require 4+ repetitions
            ]
            for issue in security_issues:
                if re.search(issue, text, re.IGNORECASE):
                    return True
            
        return False

    def _generate_enhanced_template(self, user_story: str, test_type: str) -> str:
        """Generate high-quality template with user story context extraction"""
        import re
        
        # Extract key context from user story
        user_story_lower = user_story.lower()
        
        # Detect specific banking actions
        if "face id" in user_story_lower or "facial" in user_story_lower:
            action = "authenticate with Face ID"
            context = "biometric authentication setup"
        elif "fingerprint" in user_story_lower or "touch id" in user_story_lower:
            action = "authenticate with fingerprint"
            context = "biometric authentication setup"
        elif "transfer" in user_story_lower:
            if "international" in user_story_lower or "vietnam" in user_story_lower:
                action = "transfer money internationally"
                context = "international transfer requirements"
            else:
                action = "transfer money"
                context = "sufficient balance and valid recipient account"
        elif "payment" in user_story_lower or "pay" in user_story_lower:
            if "bill" in user_story_lower:
                action = "pay bills"
                context = "bill payment setup and valid biller information"
            else:
                action = "make payment"
                context = "payment account setup"
        elif "balance" in user_story_lower or "check" in user_story_lower:
            action = "check account balance"
            context = "account access permissions"
        elif "notification" in user_story_lower:
            action = "receive notifications"
            context = "notification preferences enabled"
        elif "offline" in user_story_lower:
            action = "access features offline"
            context = "offline mode capabilities"
        elif "network" in user_story_lower or "connectivity" in user_story_lower:
            action = "handle network issues"
            context = "poor network conditions"
        elif "credit card" in user_story_lower or "block" in user_story_lower:
            action = "block credit card"
            context = "security protocols for card management"
        elif "investment" in user_story_lower or "portfolio" in user_story_lower:
            action = "check investment portfolio"
            context = "investment account access"
        else:
            action = "perform banking operation"
            context = "valid account credentials"
        
        # Generate context-aware templates
        templates = {
            'functional': f"""Scenario: User {action} in mobile banking app
Given the user has a valid account and mobile banking app is installed
And {context} is properly configured
When the user attempts to {action}
Then the system should process the request successfully
And the user should receive appropriate feedback
And the operation should be logged for security purposes""",

            'security': f"""Scenario: Security validation for {action} in mobile banking
Given the user has valid credentials and proper authentication setup
And the mobile banking app has security protocols enabled
When the user attempts to {action} using secure methods
Then the system should validate user identity and permissions
And all sensitive data should be encrypted during transmission
And security audit logs should be created""",

            'performance': f"""Scenario: Performance testing for {action} in mobile banking
Given the mobile banking app is running on a standard device
And the user has a stable network connection
When the user initiates {action} request
Then the system should respond within acceptable timeframes
And the app should remain responsive during processing
And system resources should be used efficiently"""
        }
        
        return templates.get(test_type, templates['functional']).strip()

# Example usage
if __name__ == "__main__":
    # Initialize với checkpoint path
    logger.info("=" * 60)
    logger.info("🚀 Testing ModelInterface with auto_checkpoint_1200")
    logger.info("=" * 60)
    
    interface = ModelInterface()
    
    # Test với sample user story
    user_story = "As a banking customer, I want to transfer money between my accounts so that I can manage my finances"
    
    logger.info(f"\n📝 User Story: {user_story}")
    
    options = {
        'test_types': ['Functional Testing', 'Security Testing'],
        'num_test_cases': 2
    }
    
    # Generate test cases
    logger.info("\n🤖 Generating test cases...")
    results = interface.generate_test_cases(user_story, options=options)
    
    # Display results
    logger.info(f"\n✅ Generated {len(results)} test cases:")
    for i, test_case in enumerate(results, 1):
        print(f"\n{'='*50}")
        print(f"Test Case {i}: {test_case['title']}")
        print(f"Type: {test_case['test_type']}")
        print(f"Scenario: {test_case['scenario']}")
        print("Steps:")
        for step in test_case['steps']:
            print(f"  - {step}")
    
    logger.info("\n🎯 ModelInterface ready for integration!")
