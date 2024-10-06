This project focuses on Speech Emotion Recognition (SER) using the Toronto Emotional Speech Set (TESS) dataset, which contains emotional speech samples. The goal is to identify emotions from speech data by leveraging machine learning, specifically using LSTM (Long Short-Term Memory) neural networks and MFCC (Mel Frequency Cepstral Coefficients) for feature extraction.

=> Dataset
- Dataset: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
- Size: 2800 audio files across 7 emotion categories: sad, angry, fear, neutral, ps, disgust, happy.
  
=> Key Libraries
- librosa: For audio processing and feature extraction.
- pandas, numpy: Data manipulation and handling.
- keras: Deep learning model building.
- seaborn, matplotlib: Data visualization.
- 
=> Project Steps
1. Data Loading and Preprocessing
- The dataset was loaded from Kaggle after authenticating using the kaggle.json API key.
- Audio files were organized into two columns: file paths (speech) and their respective emotion labels (label).
- The dataset contains 2800 speech samples with equal distribution of emotions (400 samples for each emotion).

2. Feature Extraction
- MFCC (Mel Frequency Cepstral Coefficients) was used to extract features from the audio data. MFCC helps capture important features from the speech signals that are useful for emotion classification.
- The MFCC for each audio sample was extracted using librosa, with 40 coefficients for each sample. These features were stored as an array.

3.Model Building
- LSTM Model: An LSTM network was chosen due to its ability to process sequential data like audio signals.

 => Model layers:
  - LSTM layer with 256 units to process sequential audio data.
  - Dropout layers to prevent overfitting.
  - Two Dense layers for further learning with ReLU activation.
  - A final output Dense layer with 7 units (for 7 emotion classes) and softmax activation.
- OneHotEncoder was applied to transform the categorical emotion labels into one-hot encoded vectors.

4. Training the Model
- The dataset was split into training (80%) and validation (20%) sets.
- The model was trained using categorical cross-entropy as the loss function and Adam optimizer for 30 epochs, with a batch size of 64.
- Training Metrics: Accuracy and loss were tracked for both training and validation sets.

5. Evaluation
- After training, the model's performance was evaluated based on its ability to predict emotions correctly on the validation set.
- Visualizations (wave plots and spectrograms) were created for sample audio files to observe the different emotional speech patterns.

=> Files and Folders
- kaggle.json: Authentication file for downloading the dataset.
- toronto-emotional-speech-set-tess.zip: The TESS dataset containing all speech files.
 => Python files:
   - main_script.ipynb: The main Jupyter notebook containing the code for data preprocessing, feature extraction, model building, and evaluation.

=> How to Run the Project
- Clone the repository or download the project files.
- Download the Toronto Emotional Speech Set (TESS) dataset from Kaggle.
-> Run the main Jupyter notebook:
    - Install the required libraries: librosa, pandas, keras, numpy, matplotlib, seaborn.
    - Ensure kaggle.json is placed in the appropriate folder for dataset download.
    - Follow the steps in the notebook to load the dataset, extract MFCC features, build and train the LSTM model.

=> Results
- The model successfully predicts emotions from speech data with a reasonably high accuracy.
Audio feature visualization through wave plots and spectrograms highlights the distinct patterns for different emotions.

=> Future Improvements
- Implement data augmentation to improve the model's generalization.
- Experiment with CNN-LSTM hybrid models for better performance.
- Use more complex feature extraction techniques such as Chroma and Spectral contrast for additional insights.

=> Requirements
- Python 3.x
- Libraries: librosa, keras, tensorflow, matplotlib, pandas, numpy, seaborn

=> Conclusion
- This project demonstrates the effective use of LSTM for speech emotion recognition, leveraging audio processing techniques like MFCC. It showcases how deep learning models can capture emotional nuances from speech patterns, enabling further developments in emotion-driven applications.
