# 📚 HƯỚNG DẪN DEMO CHO GIÁO SƯ HƯỚNG DẪN

## 🎯 BANKING TEST CASE GENERATOR
**Luận văn**: Generative AI for Testing of Banking System in Mobile Applications  
**Sinh viên**: Vũ Tuấn Chiến  
**Model**: Flan-T5 Base (Fine-tuned với 1200 epochs trên Banking Test Cases Dataset)

---

## 🚀 QUICK START

### Option 1: Chạy script tự động (Recommended)
```bash
# Trong terminal, chạy:
bash start_demo.sh

# Script sẽ:
# 1. Kiểm tra Python và packages
# 2. Verify model checkpoint
# 3. Cho phép chọn demo mode (Terminal hoặc Web)
```

### Option 2: Chạy trực tiếp

#### Terminal Demo (Đơn giản, nhanh)
```bash
python3 demo_test_generator.py
```

#### Web Interface (Đầy đủ tính năng)
```bash
streamlit run 06_deployment/ui/app.py
```

---

## 📋 DEMO SCENARIOS

### 1. **Login Feature Test**
- **User Story**: "As a banking customer, I want to login to mobile banking app securely"
- **Expected**: Generate test cases cho authentication, security, biometric login

### 2. **Money Transfer Test**
- **User Story**: "As a user, I want to transfer money between my accounts"
- **Expected**: Generate test cases cho internal transfer, validation, confirmation

### 3. **Balance Check Test**
- **User Story**: "As a customer, I want to check my account balance"
- **Expected**: Generate test cases cho real-time balance, transaction history

### 4. **Bill Payment Test**
- **User Story**: "As a user, I want to pay my bills through mobile banking"
- **Expected**: Generate test cases cho payment flow, scheduling, confirmation

---

## 🔍 CÁC ĐIỂM DEMO QUAN TRỌNG

### 1. Model Loading
```
✅ Model checkpoint: auto_checkpoint_1200
✅ Model size: ~990MB (model.safetensors)
✅ Parameters: 247.6M
✅ Architecture: T5ForConditionalGeneration (12 layers, 768 hidden dims)
```

### 2. Test Types Supported
- ✅ **Functional Testing**: Core banking features
- ✅ **Security Testing**: Authentication, encryption, access control
- ✅ **Performance Testing**: Response time, load handling
- ✅ **Compliance Testing**: PCI DSS, regulatory requirements

### 3. Output Format
- **Gherkin Format** (Given-When-Then)
- **Banking Domain Specific** (mobile app context)
- **Export to .feature file** (cho Cucumber/BDD)

---

## 💻 TERMINAL DEMO FLOW

1. **Start Demo**
   ```bash
   python3 demo_test_generator.py
   ```

2. **Choose Scenario**
   - Chọn 1-4 cho predefined scenarios
   - Chọn 5 để nhập custom User Story

3. **Select Test Types**
   - 1: Functional Testing only
   - 2: Security Testing only
   - 3: Performance Testing only
   - 4: All types

4. **Specify Number of Test Cases**
   - Default: 3 test cases
   - Range: 1-5 test cases

5. **View Generated Test Cases**
   - Formatted output với emoji indicators
   - Clear Gherkin structure
   - Banking-specific steps

6. **Export Option**
   - Save to .feature file
   - Timestamped filename

---

## 🌐 WEB INTERFACE FEATURES

### Main Features:
1. **User Story Input Form**
   - Text area for requirements
   - Context field for additional info

2. **Configuration Panel**
   - Test types selection (multi-select)
   - Number of test cases (slider)
   - Banking standards (PCI DSS, ISO 27001, etc.)

3. **Results Display**
   - Gherkin formatted output
   - Quality metrics
   - Copy/Export buttons

4. **Session Management**
   - Persistent state during session
   - History of generated tests

---

## 📊 METRICS & EVALUATION

### Model Performance:
- **Loading Time**: ~5-10 seconds (CPU)
- **Generation Speed**: 1-3 seconds per test case
- **Memory Usage**: ~2-3GB RAM

### Quality Indicators:
- **Completeness Score**: Kiểm tra Given-When-Then structure
- **Banking Relevance**: Domain-specific terminology
- **Gherkin Compliance**: Format validation
- **Testability**: Actionable steps

---

## 🔧 TROUBLESHOOTING

### Issue 1: Model không load được
```bash
# Kiểm tra checkpoint path
ls -la auto_checkpoint_1200/

# Phải có các files:
# - config.json
# - model.safetensors (944MB)
# - tokenizer.json
# - tokenizer_config.json
# - spiece.model
```

### Issue 2: Out of memory
```bash
# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""
python3 demo_test_generator.py
```

### Issue 3: Missing packages
```bash
pip install torch transformers streamlit
```

---

## 📝 SAMPLE OUTPUT

### Input:
```
User Story: As a banking customer, I want to transfer money between my accounts
Context: Mobile banking app with OTP verification
```

### Generated Test Case:
```gherkin
Feature: Money Transfer Test

Scenario: Valid money transfer between accounts
  Given user is logged into mobile banking app
  And user has sufficient balance in source account
  When user selects transfer option
  And user enters destination account details
  And user enters transfer amount
  And user confirms with OTP
  Then transfer should be processed successfully
  And user should receive confirmation message
  And account balances should be updated
```

---

## 🎓 ACADEMIC POINTS TO HIGHLIGHT

1. **Fine-tuned Model**: Trained specifically on banking test cases dataset
2. **Domain Adaptation**: Banking-specific vocabulary and patterns
3. **Gherkin Compliance**: Industry-standard BDD format
4. **Mobile Focus**: Specialized for mobile banking applications
5. **Security Awareness**: Generates security-focused test cases
6. **Compliance Ready**: Considers banking regulations

---

## 📞 SUPPORT DURING DEMO

### Quick Commands:
```bash
# Test model loading only
python3 -c "from model_interface_updated import ModelInterface; m = ModelInterface()"

# Generate single test case
python3 -c "
from model_interface_updated import ModelInterface
m = ModelInterface()
result = m.generate_test_cases('User login test', options={'num_test_cases': 1})
print(result)
"

# Check GPU availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Demo Tips:
1. **Start with Terminal Demo** - Faster, simpler, focused
2. **Use Predefined Scenarios** - Tested and reliable
3. **Show Export Feature** - Practical application
4. **Emphasize Banking Domain** - Specialized vocabulary
5. **Highlight Model Info** - 247M parameters, fine-tuned

---

## ✅ CHECKLIST TRƯỚC KHI DEMO

- [ ] Checkpoint folder `auto_checkpoint_1200` exists
- [ ] Python 3.8+ installed
- [ ] Required packages installed (torch, transformers)
- [ ] Terminal cleared and ready
- [ ] Network stable (for package downloads if needed)
- [ ] Sample user stories prepared
- [ ] Export folder writable

---

## 🎯 EXPECTED DEMO DURATION

- **Setup & Introduction**: 2-3 minutes
- **Model Loading Demo**: 1-2 minutes
- **Generate 3-4 Test Cases**: 3-5 minutes
- **Show Different Test Types**: 2-3 minutes
- **Export & Discussion**: 2-3 minutes

**Total**: ~15 minutes

---

**Good luck with your demo! 🚀**
