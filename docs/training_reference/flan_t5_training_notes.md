# Flan-T5 Model Training Notes

## Overview
This document contains key insights and lessons learned from training Flan-T5 model for mobile banking test case generation.

## Model Selection
- **Model**: google/flan-t5-base (250M parameters)
- **Reason**: Balance between performance and resource requirements
- **Alternative considered**: flan-t5-small (faster but less capable)

## Training Configuration

### Hyperparameters
```python
training_args = TrainingArguments(
    output_dir="./flan-t5-mobile-banking",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
)
```

### Data Preprocessing
- Input format: Natural language test specifications
- Output format: Structured test cases (Gherkin-style)
- Max input length: 512 tokens
- Max output length: 256 tokens

## Training Dataset Requirements
- Minimum samples: 7,000 for training
- Validation set: 1,500 samples
- Test set: 1,500 samples
- Total: 10,000 samples recommended

## Key Learnings

### 1. Data Quality > Quantity
- Clean, well-structured data produces better results than large noisy datasets
- Consistent formatting is crucial for model learning

### 2. Feature Coverage
Essential features for mobile banking:
- Login/Authentication
- Fund Transfer
- Bill Payment
- Card Management
- Account Overview
- Transaction History
- Security Features

### 3. Prompt Engineering
Effective prompts for test generation:
```
"Generate a {scenario_type} test case for {feature}"
"Create security test for {feature} validating {security_aspect}"
```

### 4. Security Integration
- Include security scenarios (25% of dataset)
- Cover compliance standards (GDPR, PCI-DSS)
- Add risk levels and security validations

## Performance Metrics

### Training Results
- Training loss: ~0.3 after 3 epochs
- Validation loss: ~0.35
- BLEU score: ~0.65
- ROUGE-L: ~0.70

### Inference Time
- Average generation time: 0.5-1.0 seconds per test case
- Batch processing: 20-30 test cases per second

## Common Issues & Solutions

### Issue 1: Generic Test Cases
**Solution**: Use more specific prompts and feature-specific training data

### Issue 2: Inconsistent Format
**Solution**: Enforce strict output schema during training

### Issue 3: Missing Security Aspects
**Solution**: Dedicate 25% of training data to security scenarios

## Deployment Considerations

### Resource Requirements
- GPU: Recommended for training (T4 or better)
- RAM: Minimum 16GB for training
- Storage: ~5GB for model and checkpoints

### Optimization Techniques
1. Quantization for deployment
2. Model pruning for edge devices
3. Caching for frequently generated patterns

## Future Improvements
1. Fine-tune on domain-specific terminology
2. Add multi-language support
3. Implement reinforcement learning from user feedback
4. Expand to API testing scenarios

## Code Snippets

### Model Loading
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
```

### Inference Example
```python
def generate_test_case(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,
        temperature=0.7,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## References
- Original notebook: mobile-banking-ai-flan-t5-training-v4.ipynb
- Model card: https://huggingface.co/google/flan-t5-base
- Training date: August-September 2024
