# Real-Time Customer Support Classification

> Multi-task BERT classifier for intelligent customer message routing with noise robustness

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/navyaravuri/customer-support-classification/blob/main/BERT_Customer_Support_Classification.ipynb)

---

## 🎯 Overview

Automated customer message classification system achieving **98.1% accuracy** on noisy real-world messages. Simultaneously predicts intent, urgency, sentiment, and routing action in **15ms** per message.

**Key Results:**
- 🎯 98.1% overall accuracy (100% intent, 98.7% urgency, 97.3% sentiment, 96.2% action)
- ⚡ 15ms inference latency
- 🚀 5.6× batch processing speedup

---

## 🏗️ Tech Stack

**Core:**
- Python 3.10+ | PyTorch 2.9.0 | Transformers 5.1.0
- DistilBERT (66M params) | Multi-task classification

**Data:**
- CLINC150 dataset | Custom noise augmentation
- 2,100 train / 450 val / 450 test samples

**Deployment:**
- Gradio 4.0+ | Google Colab L4 GPU
- 15-20 min training time

---

## 🚀 Quick Start

### **Run in Colab** (Recommended)

1. Click the "Open in Colab" badge above
2. Enable GPU: `Runtime` → `Change runtime type` → `GPU`
3. Run all cells sequentially (~25-30 min total)
4. Launch Gradio demo at the end

### **Local Setup**
```bash
# Clone repository
git clone https://github.com/navyaravuri/Customer-Support-Classification.git
cd Customer-Support-Classification

# Install dependencies
pip install -r requirements.txt

# Open notebook
jupyter notebook BERT_Customer_Support_Classification.ipynb
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | 98.1% |
| Intent Accuracy | 100% |
| Inference Speed | 15-20ms |
| Batch Throughput | 1,130 msg/sec |
| Training Time | ~17 min (4 epochs) |

**Noise Robustness:** Noisy test (98.1%) > Clean test (97.3%)

---

## 💡 Key Features

- **Multi-task learning** - 4 classification heads sharing one BERT encoder
- **Noise robust** - Handles typos, slang, ALL CAPS, emotional expressions
- **Production optimized** - Parallel batch processing, confidence scores
- **Business metrics** - Custom cost analysis beyond accuracy

---

## 🎓 Methodology

1. **Data**: CLINC150 banking queries + noise augmentation (typos, casing, slang)
2. **Model**: DistilBERT + 4 task-specific classification heads
3. **Training**: 4 epochs, AdamW optimizer, multi-task loss
4. **Evaluation**: Dual test sets (noisy/clean), business cost metric

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

**Built with PyTorch & Transformers**
