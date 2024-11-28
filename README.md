# FARC

## Overview
This project implements a machine learning model to classify fake reviews using deep learning techniques. The classifier utilizes SentenceTransformer embeddings and a bidirectional LSTM neural network to distinguish between genuine and potentially fake reviews.

## Features
- Uses pre-trained SentenceTransformer model for text embeddings
- Bidirectional LSTM neural network architecture
- Supports both CUDA and CPU inference
- Includes train-test split for model evaluation
- Calculates and reports training and testing accuracy

## Requirements
- Python 3.8+
- PyTorch
- pandas
- scikit-learn
- sentence-transformers
- CUDA (optional, for GPU acceleration)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/narensen/FARC.git
cd FARC
```

2. Install required dependencies:
```bash
pip install torch pandas scikit-learn sentence-transformers
```

## Dataset
The model uses a custom CSV dataset with the following columns:
- `text_`: Review text
- `rating`: Review rating
- `label`: Review classification (OR/CG)

## Model Architecture
- **Embedding Layer**: SentenceTransformer 'all-MiniLM-L6-v2'
- **Neural Network**: Bidirectional LSTM with fully connected output layer
- **Hyperparameters**:
  - Hidden size: 128
  - Number of layers: 2
  - Dropout: 0.3

## Training
The model trains for 10 epochs, using:
- Adam optimizer
- Learning rate: 1e-4
- Cross-entropy loss function

## Usage
Modify the `path` variable in the script to point to your dataset, then run the script to train and evaluate the model.

## Performance Metrics
The script outputs epoch-wise:
- Training loss
- Training accuracy
- Test loss
- Test accuracy

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Your Name - scartegy@gmail.com

Project Link: [https://github.com/narensen/FARC](https://github.com/narensen/FARC)
