# 🔒 Security-Aware Test Case Generation System

## Mobile Banking Test Automation with AI

A comprehensive system for generating security-aware test cases for mobile banking applications using AI, with special focus on compliance standards (ISO-27001, GDPR, PCI-DSS, MASVS).

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.8
pip install -r requirements.txt
```

### Generate Dataset
```bash
cd src/generators
python generate_security_aware_dataset.py
```

### Analyze Results
```bash
cd src/analysis
python analyze_security_dataset.py
```

### Run Demo
```bash
cd demo
bash start_demo.sh
# Or directly: streamlit run app.py
```

## 📁 Project Structure

```
clean_project/
├── schemas/                    # Data schemas and configurations
│   ├── ai_optimized_schema_v2_with_security.yaml
│   └── ai_optimized_schema.yaml
│
├── src/                        # Source code
│   ├── generators/            # Dataset generation
│   │   ├── generate_security_aware_dataset.py
│   │   └── perfect_dataset_pipeline.py
│   ├── analysis/              # Analysis tools
│   │   └── analyze_security_dataset.py
│   └── utils/                 # Utilities
│       └── model_interface.py
│
├── demo/                      # Demo application
│   ├── app.py                # Streamlit UI
│   ├── test_model_detailed.py
│   └── start_demo.sh
│
├── datasets/                  # Generated datasets
│   ├── current/              # Latest security-aware dataset
│   └── reference/            # Reference datasets
│       └── perfect_dataset_v1/
│
├── reports/                   # Analysis reports
│   ├── *.png                 # Visualizations
│   ├── *.txt                 # Text reports
│   └── *.json                # Statistics
│
└── docs/                      # Documentation
    ├── guides/               # User guides
    └── training_reference/   # Training insights
```

## ✨ Key Features

### 1. **Smart Security Integration**
- 25% of dataset dedicated to security scenarios (2,500/10,000 cases)
- 4 security subtypes aligned with industry standards
- Risk assessment and compliance tracking

### 2. **Comprehensive Coverage**
- 10 mobile banking features
- 4 scenario types: positive, negative, security, edge
- Quality scoring for each test case

### 3. **Standards Alignment**
- ISO-27001 (Information Security)
- GDPR (Data Privacy)
- PCI-DSS (Payment Security)
- MASVS (Mobile Security)

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Test Cases | 10,000 |
| Security Coverage | 25% |
| Features Covered | 10 |
| Compliance Standards | 4 |
| Average Quality Score | 0.68 |

### Distribution
- **Positive scenarios**: 35% (3,500 cases)
- **Negative scenarios**: 20% (2,000 cases)
- **Security scenarios**: 25% (2,500 cases)
- **Edge cases**: 20% (2,000 cases)

## 🔐 Security Features

### Authentication Security
- Session timeout testing
- Account lockout validation
- 2FA/MFA implementation checks

### Data Privacy (GDPR)
- PII encryption verification
- Data masking validation
- Consent management testing

### Payment Security (PCI-DSS)
- Card data tokenization
- SSL/TLS validation
- Transaction signing

### Mobile Security (MASVS)
- Jailbreak/root detection
- Certificate pinning
- Anti-tampering measures

## 🛠️ Usage Examples

### Generate Custom Dataset
```python
from src.generators.generate_security_aware_dataset import SecurityAwareTestGenerator

generator = SecurityAwareTestGenerator()
dataset = generator.generate_dataset(total_samples=5000)
generator.save_dataset(dataset)
```

### Analyze Dataset
```python
from src.analysis.analyze_security_dataset import DatasetAnalyzer

analyzer = DatasetAnalyzer('datasets/current/security_aware_dataset.json')
report = analyzer.generate_report()
```

### Generate Specific Test Case
```python
test_case = generator.generate_single_test_case(
    feature='fund_transfer',
    scenario_type='security'
)
```

## 📈 Visualizations

The analysis tool generates:
- Scenario distribution charts
- Security coverage heatmaps
- Feature coverage analysis
- Quality metrics dashboards

## 🚀 Model Training

For training AI models with this dataset:

1. Use JSONL format for direct training input
2. Split: 70% train, 15% validation, 15% test
3. Recommended models: Flan-T5, GPT-3.5, Llama-2

See `docs/training_reference/flan_t5_training_notes.md` for detailed training insights.

## 📝 Documentation

- [Security Dataset Guide](docs/guides/README_SECURITY_DATASET.md)
- [Demo Instructions](docs/guides/DEMO_INSTRUCTIONS.md)
- [Training Notes](docs/training_reference/flan_t5_training_notes.md)
- [Schema Analysis](docs/schema_integration_analysis.md)
- [Security Integration](docs/security_compliance_integration_analysis.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

This project is part of a Master's Thesis at HUIT focusing on Generative AI for Software Testing.

## 🙏 Acknowledgments

- HUIT - Ho Chi Minh City University of Industry and Trade
- Thesis Advisor and Committee
- Open source community

---

**Version**: 2.0  
**Last Updated**: September 2024  
**Status**: Active Development
