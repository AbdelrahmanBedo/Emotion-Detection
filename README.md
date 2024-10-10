# Emotion Detection Project with MLOps Integration

This project implements an Emotion Detection system using Convolutional Neural Networks (CNN) to classify human emotions from grayscale facial images, now integrated with **MLOps** practices for smoother model deployment, monitoring, and version control.

## Project Overview

The model is built using Keras with TensorFlow backend, featuring a CNN to classify seven emotions: **Anger, Disgust, Fear, Happy, Sad, Surprise, and Neutral**. With MLOps integration, the project now includes automated data preprocessing, model training pipelines, and deployment to cloud services for real-time emotion detection.

### MLOps Integration

The project follows MLOps principles, including:
- **Automated Pipelines**: End-to-end automation for data ingestion, model training, testing, and deployment using tools like **GitHub Actions** or **Jenkins**.
- **Model Versioning**: Tracking different versions of models with **DVC** (Data Version Control) or **MLflow** for easier rollback and comparison of performance.
- **Continuous Integration/Continuous Deployment (CI/CD)**: Automating model retraining and deployment using CI/CD pipelines.
- **Monitoring and Logging**: Using tools like **Prometheus** or **Grafana** for real-time performance monitoring, and logging model predictions for analysis.

### Dataset

The project uses a structured dataset with **training** and **test** directories. Each directory contains subfolders named after the emotions being classified. The dataset is loaded using Keras' `flow_from_directory()` method, and is versioned using **DVC** for efficient tracking.

### Model Architecture

The CNN architecture includes:
- **Convolutional Layers**: Extract features using 32, 64, and 128 filters.
- **MaxPooling Layers**: Reduce dimensionality.
- **Dropout Layers**: Prevent overfitting with dropout rates of 25% and 50%.
- **Fully Connected Layer**: A dense layer with 1024 units and a softmax output layer for classification.

### Requirements

In addition to core libraries, MLOps tools are needed:
- `opencv-python`
- `keras`
- `tensorflow`
- `dvc`
- `mlflow`
- `prometheus-client`

Install the libraries using pip:

```bash
pip install opencv-python keras tensorflow dvc mlflow prometheus-client
```

### MLOps Workflow

1. **Data Versioning**: The dataset is versioned with DVC.
2. **Training Pipeline**: Model training is automated using a CI/CD pipeline with GitHub Actions or Jenkins.
3. **Model Registry**: Trained models are registered and tracked using MLflow, allowing for easy version control and comparison.
4. **Deployment**: The model is deployed to cloud environments, such as AWS or Azure, for real-time emotion detection.
5. **Monitoring**: Real-time monitoring is implemented using Prometheus, and logs are visualized with Grafana.

### Model Training

The model is trained over 50 epochs with a batch size of 64. Training and validation datasets are rescaled to have pixel values between 0 and 1. The model is optimized using Adam.

```python
emotion_model.compile(loss='categorical_crossentropy', 
                      optimizer=Adam(learning_rate=0.0001), 
                      metrics=['accuracy'])
emotion_model.fit(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7178 // 64)
```


### Results

The trained model achieves a solid level of accuracy across the seven emotional categories. Performance is continuously monitored, and models are automatically retrained when performance degrades.

### Future Work

- Further improve the MLOps pipeline with more robust monitoring and alerting.
- Incorporate **automated hyperparameter tuning** using tools like **Optuna**.
- Integrate **edge deployment** for real-time emotion detection on mobile or embedded devices.
- Increase the model accuracy 
---
