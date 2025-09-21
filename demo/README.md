# 🏛️ Banking AI Test Generator - Demo

## 🌟 Overview
**Web demo cho hệ thống AI tự động sinh test case cho ứng dụng ngân hàng di động**

- **Model**: CodeT5 fine-tuned (222.9M parameters)
- **Training**: 8 epochs trên Kaggle T4 GPU với 8000 samples
- **Test Loss**: 0.1228 (Xuất sắc!)
- **Location**: `../model_trained/` (850MB)

## 🚀 Quick Start

```bash
# Chạy demo
cd demo
python run_demo.py

# Hoặc chạy trực tiếp Streamlit
streamlit run app_simple.py
```

**URL**: http://localhost:8501

## 📁 File Structure

```
demo/
├── app_simple.py              # Streamlit web interface
├── model_interface_real.py    # Interface cho trained model
├── simple_model_interface.py  # Wrapper interface
├── run_demo.py                # Demo launcher script
└── README_DEMO.md             # This file
```

## ✨ Features

### 1️⃣ **Test Generation**
- Nhập user story bằng tiếng Anh
- Generate test case theo format Gherkin
- Thời gian generate: ~10 giây trên CPU

### 2️⃣ **Test Types**
- 🎯 **Functional**: Test chức năng
- 🔒 **Security**: Test bảo mật
- ⚡ **Performance**: Test hiệu năng
- 📝 **Compliance**: Test tuân thủ

### 3️⃣ **Sample User Stories**
- Login with biometric
- Money transfer
- Balance inquiry
- Bill payment
- Notifications
- Card management

## 🔧 Technical Details

### Model Specs
- **Base**: Salesforce/CodeT5-base
- **Fine-tuned**: 8 epochs trên banking test cases
- **Parameters**: 222.9M
- **Format**: Safetensors (fast loading)
- **Device**: CPU/GPU compatible

### Generation Parameters
- Max length: 150-200 tokens
- Beam search: 4 beams
- Temperature: 0.7
- No repeat n-gram: 3

## 📦 Dependencies

```bash
pip install streamlit transformers torch pandas
```

## 🎯 Usage Guide

1. **Chạy demo**: `python run_demo.py`
2. **Mở browser**: http://localhost:8501
3. **Chọn sample** hoặc nhập user story mới
4. **Chọn loại test**: Functional/Security/Performance/Compliance
5. **Click Generate** và chờ ~10 giây
6. **Download** test case đã generate

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Training Loss | 0.6448 |
| Test Loss | 0.1228 |
| Eval Loss | 0.1201 |
| Generation Time | ~10s (CPU) |
| Model Size | 850MB |
| Confidence Score | 99.9% |

## 🔍 Example Output

**Input**: "As a user, I want to login to mobile banking using fingerprint"

**Output**:
```gherkin
Scenario: Verify Login Security Controls
Given the user has authenticated successfully
And user has sufficient privileges
When the user attempts action and security challenge is presented
Then attempt is logged for review
```

## 👨‍🎓 Author
**Vũ Tuấn Chiến**
- Luận văn Thạc sĩ CNTT
- Generative AI for Testing of Banking System in Mobile Applications

## 📄 License
For academic purposes only

---
*Demo này sử dụng model đã được train trên Kaggle với 8000+ test cases cho banking domain*
