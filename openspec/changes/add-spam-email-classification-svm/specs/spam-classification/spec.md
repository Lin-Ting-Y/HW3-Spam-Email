## ADDED Requirements

### Requirement: Baseline SVM Spam Classifier
The system SHALL provide a reproducible baseline spam classifier trained with an SVM using the referenced SMS spam dataset. The baseline SHALL include training, evaluation, and saved model artifacts.

#### Scenario: Train baseline model
- **GIVEN** the dataset from `https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/Chapter03/datasets/sms_spam_no_header.csv` is available in `data/`
- **WHEN** the `train_baseline.py` script is executed
- **THEN** the script SHALL produce a saved model artifact and an evaluation report with accuracy, precision, recall, and F1 scores

### Requirement: Streamlit App UI (Phase 2)
The system SHALL provide a Streamlit application that enables interactive spam/ham classification in Phase 2. The app SHALL accept user text input, run inference using the trained model, and display the predicted label with a confidence score. The app SHOULD also surface key evaluation visuals (e.g., accuracy/F1 summary, confusion matrix, or a word cloud) when available.

#### Scenario: Streamlit interaction
- **GIVEN** a deployed Streamlit app with access to the trained baseline model artifacts
- **WHEN** a user enters a text message and submits
- **THEN** the app SHALL display the classification `spam` or `ham` with a confidence score and MAY render available evaluation visuals
