# Emotion Recognition Using Toronto Emotional Speech Set (TESS)

## Overview

This project focuses on recognizing emotions from speech data using the Toronto Emotional Speech Set (TESS). The primary objective is to classify different emotions such as anger, disgust, fear, happiness, neutrality, pleasant surprise, and sadness from audio recordings. The project leverages deep learning techniques, specifically using a Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) networks, to achieve accurate emotion recognition.

## Dataset

The dataset used for this project is the [Toronto Emotional Speech Set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess). It contains 2,800 audio recordings from 7 different emotions, spoken by two actresses aged 26 and 64. The audio files are labeled with the emotion they represent, making it an ideal dataset for supervised learning.

## Key Steps

1. **Data Loading and Preprocessing:**
   - The audio files were loaded and labeled according to their respective emotions.
   - The dataset was explored to understand the distribution of emotions.
   - Visualizations such as waveplots and spectrograms were generated for different emotions.

2. **Feature Extraction:**
   - Mel-Frequency Cepstral Coefficients (MFCC) were extracted from the audio files as features.
   - The features were reshaped and prepared for input into the deep learning model.

3. **Model Development:**
   - A Sequential model was built using LSTM layers to capture temporal dependencies in the speech data.
   - The model was trained using categorical cross-entropy as the loss function and Adam as the optimizer.
   - Dropout layers were added to prevent overfitting.

4. **Model Training:**
   - The model was trained over 30 epochs with a batch size of 64.
   - Training and validation accuracy and loss were tracked and plotted to monitor the model's performance.

5. **Evaluation:**
   - The model's performance was evaluated using a confusion matrix and classification report.
   - A heatmap of the confusion matrix was generated to visualize the model's predictions versus actual labels.

## Results

- The model demonstrated good performance in classifying emotions from speech data.
- The confusion matrix and classification report provided insights into the model's strengths and areas for improvement.

## Visualizations

Several visualizations were created throughout the project, including:

- **Waveplots:** Visual representations of the audio signals for different emotions.
- **Spectrograms:** Time-frequency representations of the audio data.
- **Confusion Matrix:** A heatmap to visualize the accuracy of the model's predictions.

## Conclusion

This project successfully demonstrated the application of deep learning techniques to emotion recognition in speech. By extracting MFCC features and using an LSTM-based model, the project achieved a strong classification performance across multiple emotions. This work can be further enhanced by experimenting with different model architectures or combining it with other modalities like facial expressions for multi-modal emotion recognition.

## Acknowledgments

Special thanks to the creators of the Toronto Emotional Speech Set (TESS) and the open-source community for providing the tools and resources to make this project possible.
