# Pipeline Notebooks

This folder contains all Jupyter notebooks used in the project.  
They are organized sequentially, following the full pipeline from preprocessing to model evaluation.

---

## 00_preprocessing_sentiment_analysis_and_visualizations.ipynb  
**Stage:** Pre-processing, Sentiment Analysis, and Visualizations  
**Input:** df_all_without_cleaning.csv  
**Output:** Cleaned text datasets, sentiment analysis, visualizations  

**Description:**  
- Cleans text data (removes numbers, links, punctuation, emojis).  
- Applies sentiment analysis using VADER and TextBlob.  
- Prepares data for visualization and exploration (word clouds, polarity distributions, etc.).

---

## 01_sentimental_analysis_transformers.ipynb  
**Stage:** Sentiment Analysis with Transformers  
**Input:** Cleaned dataset from Stage 00  
**Output:** df_all_sentiment_tensor.csv  

**Description:**  
- Runs transformer-based sentiment classification using `cardiffnlp/twitter-roberta-base-sentiment-latest`.  
- Compares results with VADER/TextBlob.  
- Exports aggregated sentiment scores for further emotion recognition.

---

## 02_emotion_recognition.ipynb  
**Stage:** Emotion Recognition  
**Input:** df_all_sentiment_tensor.csv  
**Output:** Emotion-labeled dataset (per subsentence)  

**Description:**  
- Performs semantic chunking of messages into subsentences.  
- Uses `j-hartmann/emotion-english-distilroberta-base` to label each chunk with one of seven emotions.  
- Generates sequential emotion patterns for each post.  

---

## 03_encode_list_of_emoticons.ipynb  
**Stage:** Encoding Emotion Sequences, Prefixpan and Association Rules  
**Input:** df_all_emoticon_seq.csv  
**Output:** Encoded datasets ready for modeling and rule extraction  

**Description:**  
- Converts emotional labels into numerical IDs.  
- Applies PrefixSpan for frequent sequential pattern discovery.  
- Uses Apriori to extract association rules linked to fake and reliable news.  
- Exports filtered and encoded sequences for deep learning models.

---

## 04_training_LSTM_BILSTM_and_validation.ipynb  
**Stage:** Model Training (LSTM/BiLSTM)  
**Inputs:** df_all_emoticon_seq_list_filtrado1_com_virgula, dados_emocoes_divididos.npz  
**Outputs:** .keras models, performance metrics, ROC/PR curves  

**Description:**  
- Trains and validates LSTM and BiLSTM architectures.  
- Tests multiple hyperparameter configurations.  
- Evaluates models using classification metrics and visual curves.  

---

## 05_training_LSTM_and_CNN+LSTM.ipynb  
**Stage:** Model Training (CNN+LSTM)  
**Inputs:** df_all_emoticon_seq_list_filtrado1_com_virgula, dados_emocoes_divididos.npz  
**Outputs:** .keras models, metrics, and plots  

**Description:**  
- Combines convolutional and recurrent layers (CNN+LSTM).  
- Compares with previous architectures.  
- Exports best-performing models and results.

---

## 06_training_Transformers_and_evaluation.ipynb  
**Stage:** Transformer Modeling and Evaluation  
**Inputs:** df_all_emoticon_seq_list_filtrado1_com_virgula, dados_emocoes_divididos.npz  
**Outputs:** .keras Transformer model, metrics, comparative results  

**Description:**  
- Implements Transformer Encoder architecture for classification.  
- Evaluates models using Accuracy, F1, Precision, Recall, and AUC.  
- Generates comparative ROC and PR curves across architectures.

---

## functions.py  
**Purpose:** Shared helper functions across notebooks.  
Includes:
- Data preprocessing utilities  
- Sequence encoding helpers   

---

### Notes
- Paths are defined at the top of each notebook.  
- All figures and tables are saved under `/outputs/`.  
- Random seeds are fixed for reproducibility.
