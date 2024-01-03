# Sentiment Analysis using Convolutional Neural Networks

This repository contains code for performing sentiment analysis on hotel reviews using Convolutional Neural Networks (CNNs). The model is built using TensorFlow and Keras and trained on the TripAdvisor Hotel Reviews dataset.

## Dataset
The dataset used in this project is sourced from [TripAdvisor Hotel Reviews dataset](https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews). It consists of reviews and corresponding ratings given by users. Sentiment labels are derived from the ratings, categorizing them as Positive, Negative, or Neutral based on a threshold.

## Requirements
- Python 3
- Libraries:
  - numpy
  - pandas
  - tensorflow
  - scikit-learn
  - keras

## Usage
1. Clone the repository:

    ```
    git clone https://github.com/yourusername/Pavagada-Forecst.git
    ```

2. Download the dataset and place it in the `Data` directory: `Sentiment_Analysis/Data/tripadvisor_hotel_reviews.csv`.

3. Install the required libraries:

    ```
    pip install -r requirements.txt
    ```

4. Run the `sentiment_analysis.ipynb` notebook or execute the `sentiment_analysis.py` script to train the CNN model and perform sentiment analysis on the reviews.

## Files
- `sentiment_analysis.ipynb`: Jupyter Notebook containing the code for data preprocessing, model building, training, and evaluation.
- `sentiment_analysis.py`: Python script with the same functionality as the notebook.
- `tokenizer.pickle`: Pickle file containing the tokenizer used for text tokenization.
- `sentiment_analysis_model.h5`: Saved trained model weights and architecture in HDF5 format.
- `Data/tripadvisor_hotel_reviews.csv`: Dataset used for sentiment analysis.

## Model Architecture
The model architecture involves an Embedding layer followed by a 1D Convolutional layer, GlobalMaxPooling1D layer, Dense layers, and Dropout for regularization.

## Results
The model achieves an accuracy of `accuracy_score` on the test dataset.

