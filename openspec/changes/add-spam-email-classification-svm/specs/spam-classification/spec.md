## ADDED Requirements

### Requirement: Baseline SVM Spam Classifier
The system SHALL provide a reproducible baseline spam classifier trained with an SVM using the referenced SMS spam dataset. The baseline SHALL include training, evaluation, and saved model artifacts.

#### Scenario: Train baseline model
- **GIVEN** the dataset from `https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/Chapter03/datasets/sms_spam_no_header.csv` is available in `data/`
- **WHEN** the `train_baseline.py` script is executed
- **THEN** the script SHALL produce a saved model artifact and an evaluation report with accuracy, precision, recall, and F1 scores

### Requirement: Streamlit App UI (Phase 2)
The system SHALL provide a Streamlit user interface for interactive classification and model exploration. The UI SHALL load a trained model and evaluation artifacts, accept user input for on-the-fly inference, and present evaluation visuals.

#### Scenario: Interactive UI
- **GIVEN** a deployed Streamlit app and a trained model with evaluation artifacts
- **WHEN** a user enters a message
- **THEN** the app SHALL return classification `spam` or `ham` with a confidence score
- **AND** the app SHALL display evaluation metrics and a confusion matrix
- **AND** the app SHALL display Precision–Recall and ROC curves if available
- **AND** the app SHALL allow adjusting a decision threshold and recomputing Accuracy, Precision, Recall, and F1 on the test split
