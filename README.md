# Fine-Tuning RF-DETR on Custom Dataset

This repository contains a comprehensive guide and Jupyter notebook for fine-tuning RF-DETR (Real-time Detection Transformer) on custom datasets. RF-DETR is a state-of-the-art object detection model that combines the speed of YOLO with the accuracy of DETR transformers.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Training Configuration](#training-configuration)
- [Results and Evaluation](#results-and-evaluation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

RF-DETR is a cutting-edge object detection model that offers:
- **Real-time performance**: Optimized for speed without sacrificing accuracy
- **Transformer architecture**: Leverages the power of attention mechanisms
- **End-to-end training**: No need for complex post-processing
- **Custom dataset support**: Easy fine-tuning on your specific use case

This tutorial demonstrates how to fine-tune RF-DETR Nano on your custom dataset using Google Colab or local GPU environments.

## ✨ Features

- **Easy Setup**: One-click installation of required dependencies
- **GPU Memory Management**: Automatic CUDA memory optimization
- **Roboflow Integration**: Seamless dataset loading from Roboflow
- **Flexible Training**: Customizable training parameters
- **Real-time Monitoring**: Built-in training progress tracking

## 🔧 Requirements

### Hardware Requirements
- **GPU**: CUDA-compatible GPU with at least 8GB VRAM (recommended)
- **RAM**: Minimum 16GB system RAM
- **Storage**: At least 10GB free space for models and datasets

### Software Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- PyTorch 1.10+

## 🚀 Installation

The notebook handles all installations automatically. The key dependencies include:

```bash
pip install rfdetr==1.2.1 supervision==0.26.1 roboflow
```

### Manual Installation (Optional)
If running locally, you can install dependencies manually:

```bash
# Create virtual environment
python -m venv rfdetr_env
source rfdetr_env/bin/activate  # On Windows: rfdetr_env\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install rfdetr==1.2.1
pip install supervision==0.26.1
pip install roboflow
```

## 📊 Dataset Preparation

### Using Roboflow (Recommended)

1. **Create a Roboflow Account**: Sign up at [roboflow.com](https://roboflow.com)
2. **Upload Your Dataset**: Use Roboflow's annotation tools or upload pre-annotated data
3. **Export Dataset**: Choose YOLO format for compatibility
4. **Get API Key**: Obtain your API key from account settings

### Supported Formats
- **YOLO format** (recommended)
- **COCO format**
- **Pascal VOC format**

### Dataset Structure
```
dataset/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── labels/
│       ├── img1.txt
│       └── img2.txt
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

## 📖 Usage

### Running the Notebook

1. **Open in Google Colab**: Click the "Open in Colab" button or upload to your Colab environment
2. **Set Runtime**: Change runtime type to GPU (Runtime > Change runtime type > GPU)
3. **Run Cells**: Execute cells sequentially

### Key Steps Explained

#### 1. Environment Setup
```python
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```
Sets up CUDA for better error reporting and debugging.

#### 2. Install Dependencies
```python
!pip install -q rfdetr==1.2.1 supervision==0.26.1 roboflow
```
Installs the RF-DETR framework and supporting libraries.

#### 3. Configure API Access
```python
os.environ["ROBOFLOW_API_KEY"] = "your_api_key_here"
```
Sets up Roboflow API for dataset access.

#### 4. Memory Management
```python
import torch, gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
```
Clears GPU memory to prevent out-of-memory errors.

#### 5. Model Training
```python
from rfdetr import RFDETRNano

model = RFDETRNano()
model.train(
    dataset_dir="./",
    epochs=25,
    batch_size=8,
    grad_accum_steps=2,
    output_dir="./"
)
```

## ⚙️ Training Configuration

### Default Parameters
- **Model**: RF-DETR Nano (lightweight version)
- **Epochs**: 25
- **Batch Size**: 8
- **Gradient Accumulation Steps**: 2
- **Learning Rate**: Auto-configured
- **Image Size**: 640x640 (auto-resized)

### Customization Options

```python
model.train(
    dataset_dir="path/to/dataset",
    epochs=50,                    # Increase for better convergence
    batch_size=16,               # Adjust based on GPU memory
    grad_accum_steps=1,          # Reduce if using larger batch size
    learning_rate=1e-4,          # Custom learning rate
    warmup_epochs=3,             # Warmup period
    save_period=5,               # Save checkpoint every N epochs
    output_dir="./results",      # Output directory
    device="cuda:0",             # Specific GPU device
    workers=4,                   # DataLoader workers
    patience=10,                 # Early stopping patience
    resume=False,                # Resume from checkpoint
)
```

## 📊 Results and Evaluation

### Training Metrics
The training process provides:
- **Loss curves**: Training and validation loss over epochs
- **mAP scores**: Mean Average Precision at different IoU thresholds
- **Precision/Recall**: Per-class and overall metrics
- **Inference speed**: FPS measurements

### Model Outputs
After training, you'll find:
- `best.pt`: Best performing model weights
- `last.pt`: Final epoch model weights
- `results.csv`: Training metrics log
- `confusion_matrix.png`: Classification confusion matrix
- `results.png`: Training curves visualization

### Evaluation
```python
# Load trained model
model = RFDETRNano("path/to/best.pt")

# Evaluate on test set
results = model.val(data="path/to/test/dataset")

# Inference on single image
predictions = model.predict("path/to/image.jpg")
```

## 🔧 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size
batch_size = 4

# Increase gradient accumulation
grad_accum_steps = 4

# Clear memory before training
torch.cuda.empty_cache()
```

#### Slow Training
- Ensure GPU is being used: Check `nvidia-smi` output
- Reduce image resolution if dataset allows
- Use mixed precision training (automatically enabled)

#### Dataset Loading Errors
- Verify dataset structure matches expected format
- Check file paths and permissions
- Ensure `data.yaml` is correctly formatted

#### API Key Issues
- Verify Roboflow API key is correct
- Check internet connection
- Ensure sufficient API quota

### Performance Tips

1. **Optimal Batch Size**: Start with batch_size=8, adjust based on GPU memory
2. **Learning Rate**: Use default auto-configuration for best results
3. **Data Augmentation**: Enabled by default, helps with generalization
4. **Mixed Precision**: Automatically enabled for faster training
5. **Gradient Clipping**: Prevents gradient explosion during training

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest improvements
- Submit pull requests
- Share your training results

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **RF-DETR Team**: For developing this excellent detection framework
- **Roboflow**: For providing easy dataset management tools
- **PyTorch Team**: For the underlying deep learning framework
- **Community Contributors**: For bug reports and improvements

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the [RF-DETR documentation](https://github.com/lyuwenyu/RT-DETR)
- Join the community discussions

---

**Happy Training! 🚀**

*Remember to ⭐ star this repository if you find it helpful!*
