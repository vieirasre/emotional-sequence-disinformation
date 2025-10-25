# Detecting Disinformation through Emotional Sequence Patterns

Reproducible code and assets for the thesis project **“Detecting Disinformation through Emotional Sequence Patterns in Social Media Messages”**.

This repository contains all the notebooks, functions, and dependencies used throughout the research pipeline — from data preprocessing and sentiment analysis to emotion recognition, sequential pattern mining, association rules, and deep learning modeling.

---

## Repository Structure

```
emotional-sequence-disinformation/
│
├── data/
├── notebooks/              # Jupyter notebooks for each stage of the pipeline
├── requirements.txt        # List of dependencies
├── LICENSE                 # Apache-2.0 License
└── README.md               # Main documentation
```

---

## Setup

```bash
python -m venv .venv
# Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Pipeline Overview

The project is organized as a sequential pipeline:

1. Preprocessing, Sentiment Analysis and Visualizations  
2. Sentiment Analysis using Transformers  
3. Emotion Recognition and Sequential Encoding  
4. Frequent Sequence Mining (PrefixSpan + Association Rules)  
5. LSTM, BiLSTM, CNN+LSTM, and Transformer Modeling  
6. Evaluation and Validation

All code is designed to ensure reproducibility and traceability of every step in the emotional-sequence-based disinformation detection process.

---

## Data Files

| File | Description |
|------|--------------|
| df_all_without_cleaning.csv | Combined dataset (fake & true news) before cleaning. |
| df_all_sentiment_tensor.csv | Dataset with sentiment scores (VADER + TextBlob + Transformer). |
| df_all_emoticon_seq.csv | Emotion sequences generated after emotion recognition. |
| df_all_emoticon_seq_list_filtrado1_com_virgula | Filtered emotion sequences prepared for training. |
| dados_emocoes_divididos.npz | Encoded and padded sequences used as input for deep learning models. |

> Note: Only the first dataset will be available in the data/ folder. All subsequent datasets are automatically generated through the notebooks provided in the notebooks/ directory.

---

## Model Training and Evaluation

- Models: LSTM, BiLSTM, CNN+LSTM, Transformer Encoder  
- Metrics: Accuracy, Precision, Recall, F1-Score, AUC  
- Visualization: ROC and Precision-Recall curves  

Each notebook exports trained models (.keras) and plots in the `outputs/` directory.

---

## Citation

If you use this repository, please cite:

```
Vieira, R., Figueira, A. (2025). Detecting Disinformation through Emotional Sequence Patterns in Social Media Messages. 
Code and data repository. https://github.com/vieirasre/emotional-sequence-disinformation
```

---

## License

Licensed under the Apache-2.0 license. See the `LICENSE` file for details.
