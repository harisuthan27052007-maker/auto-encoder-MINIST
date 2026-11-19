# Fashion MNIST Classification Using a Convolutional Autoencoder

## Project Overview

This project demonstrates how a **convolutional autoencoder** can be used for image classification with the Fashion-MNIST dataset. The autoencoder is first trained to reconstruct input images. Its encoder is then reused as a feature extractor, with fully connected layers added to classify the images into 10 classes.

## Dataset

Fashion-MNIST contains 70,000 grayscale images of fashion items:

- **Training set:** 60,000 images
- **Test set:** 10,000 images
- **Image size:** 28 × 28 pixels
- **Channels:** 1 (grayscale)
- **Classes:** 10

The notebook reads the dataset from the following compressed IDX files:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

Place these files in the notebook's working directory before running the data-loading cells.

## Technologies and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Keras
- TensorFlow backend

## Project Workflow

1. **Import libraries** – Load the required Python, data-processing, visualization, and Keras modules.
2. **Load the dataset** – Read the compressed image and label files using Python's `gzip` module and NumPy.
3. **Explore the data** – Inspect image dimensions and display sample images.
4. **Preprocess images** – Reshape images to `28 × 28 × 1` and scale pixel values from 0–255 to 0–1.
5. **Split the data** – Use 80% of the training images for training and 20% for validation (`random_state=12345`).
6. **Build the convolutional autoencoder** – Create an encoder and decoder using convolution, batch normalization, max-pooling, and upsampling layers.
7. **Train the autoencoder** – Train it to reconstruct the input images using mean squared error and the RMSprop optimizer.
8. **Build the classifier** – Reuse the encoder and add a flatten layer, a 128-unit dense layer, and a 10-unit softmax output layer.
9. **Train the classifier** – Convert labels to one-hot vectors and train the classification model using categorical cross-entropy and Adam.
10. **Evaluate the model** – Predict test labels and examine accuracy, a classification report, and a confusion matrix.

## Model Architecture

### Encoder

The encoder extracts image features through convolutional layers:

- Two convolutional layers with 32 filters
- Max-pooling
- Two convolutional layers with 64 filters
- Max-pooling
- Two convolutional layers with 128 filters
- Two convolutional layers with 256 filters

Batch normalization is applied after the convolutional layers.

### Decoder

The decoder reconstructs the image from the encoded features using convolutional layers with 128, 64, and 32 filters, two upsampling operations, and a final one-channel convolution with sigmoid activation.

### Classification Head

The encoder is reused for classification. Its output is passed through:

- Flatten
- Dense layer with 128 units and ReLU activation
- Dense layer with 10 units and softmax activation

## Training Configuration

| Component | Configuration |
|---|---|
| Autoencoder batch size | 128 |
| Autoencoder epochs | 200 |
| Autoencoder optimizer | RMSprop |
| Autoencoder loss | Mean squared error |
| Classifier batch size | 64 |
| Classifier epochs | 100 per fit call |
| Classifier optimizer | Adam |
| Classifier loss | Categorical cross-entropy |
| Classifier metric | Accuracy |

The notebook contains two classifier training calls, including a retraining section. Actual results can vary depending on the environment and training execution.

## Results

The saved notebook output reports:

- **Accuracy from the later prediction and classification-report calculation:** `0.9189` (91.89%)
- **Test samples:** 10,000
- **Average precision, recall, and F1-score in the classification report:** approximately `0.92`

The notebook also contains an earlier `model.evaluate()` output showing a test accuracy value of `0.7431` (with loss `0.919`). This differs from the later `accuracy_score` result of `0.9189`. These values are reproduced as recorded in the notebook; rerun the notebook from start to finish to verify the final result and ensure the evaluation outputs correspond to the same model state.

## How to Run

1. Clone or download this project.
2. Install the required packages.
3. Download the four Fashion-MNIST compressed IDX files listed in the Dataset section and place them in the notebook's working directory.
4. Open `Autoencoder_Fashion_MNIST (1).ipynb` in Jupyter Notebook or JupyterLab.
5. Run the notebook cells in order.

Example installation command:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

> **Note:** The notebook uses older Keras import paths and APIs. Depending on your installed TensorFlow/Keras version, some imports may need updating before the notebook runs successfully.

## Expected Outputs

- Sample Fashion-MNIST images
- Training and validation loss curves for the autoencoder
- Autoencoder model summary
- Classifier training history
- Test-set predictions
- Classification report
- Confusion matrix

## Conclusion

This project shows how an autoencoder can learn useful image features through image reconstruction and how the trained encoder can be reused for a classification task. A dense classification head is added to the encoder to predict one of the 10 Fashion-MNIST classes.

---

*Project documentation prepared from the accompanying `Autoencoder_Fashion_MNIST` notebook.*
