# 📂 Dataset Structure

## ✅ Active Datasets (For Training)

### `/augmented/` - **PRIMARY DATASET**
- **Purpose**: Final augmented dataset for model training
- **Quality**: 96.12% average quality score
- **Size**: 10,000 samples (8k train, 1k val, 1k test)
- **Status**: **PRODUCTION READY**
- **Use this for**: Model training, evaluation, deployment

### `/cleaned/` - **BACKUP DATASET**  
- **Purpose**: Cleaned dataset before augmentation
- **Quality**: 97.14% average quality score
- **Size**: 7,000 samples
- **Status**: Reference/Backup
- **Use this for**: Fallback, analysis, comparison

## 📁 Archived Data

### `/archived_*/` 
- Contains compressed archives of original data
- For reference and reproducibility only

## 📊 Dataset Statistics

| Dataset | Train | Val | Test | Total | Quality |
|---------|-------|-----|------|-------|---------|
| **Augmented** | 8,000 | 1,000 | 1,000 | 10,000 | 96.12% |
| **Cleaned** | 5,600 | 700 | 700 | 7,000 | 97.14% |

## 🎯 Recommended Usage

```python
# For training
train_data = "datasets/augmented/train.json"
val_data = "datasets/augmented/val.json"
test_data = "datasets/augmented/test.json"
```

---
*Last Updated: $(date)*
