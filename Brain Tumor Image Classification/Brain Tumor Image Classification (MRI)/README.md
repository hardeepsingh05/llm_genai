# Brain Tumor MRI Image Classification

This project trains a convolutional neural network to classify brain MRI images into two categories: `Healthy` and `Tumor`. The notebook covers dataset inspection, image preprocessing, CNN construction, supervised training, evaluation, checkpoint saving, and prediction on individual MRI images.

This is an educational computer-vision experiment and is not a medical diagnostic system. Predictions must not be used as a substitute for review by qualified medical professionals.

## Workflow

1. Import numerical, image-processing, plotting, and PyTorch dependencies.
2. Select CPU or CUDA as the processing device.
3. Download the brain-tumor MRI dataset through KaggleHub.
4. Inspect sample images, image dimensions, pixel statistics, and class distribution.
5. Resize images to `224 x 224`, convert them to tensors, and normalize pixel values using dataset statistics.
6. Load the directory-structured dataset with `ImageFolder` and split it into training and test subsets.
7. Create PyTorch DataLoaders for batched training and evaluation.
8. Define a CNN with convolution, ReLU, max-pooling, flattening, dropout, and fully connected layers.
9. Train the model with cross-entropy loss and the Adam optimizer.
10. Evaluate predictions, collect labels, and display a confusion matrix.
11. Save and reload the trained model checkpoint.
12. Visualize predictions on test batches and classify individual MRI image files.

## Project Files

- `Trained CV Model.ipynb` - Main notebook containing the complete experiment.
- `MRI_Pred_Model_Accu_97.pth` - Saved trained PyTorch model checkpoint.
- `requirements.txt` - Python dependencies for the notebook.

The MRI dataset is downloaded separately through KaggleHub and is not duplicated in this folder.

## Dataset Structure

The notebook expects the downloaded image directory to contain one folder per class:

```text
Brain Tumor MRI images/
├── Healthy/
└── Tumor/
```

The class names are defined in the notebook as `['Healthy', 'Tumor']`. `ImageFolder` assigns integer class indices based on these directory names.

## Image Preprocessing

The preprocessing pipeline performs the following operations:

- Resizes each image to `224 x 224` pixels.
- Converts images to PyTorch tensors.
- Normalizes the image using the calculated dataset mean and standard deviation.
- Leaves random horizontal flipping disabled in the shown configuration, though it is available as an optional augmentation.

The notebook assumes three input channels when creating the CNN. If the source images are grayscale, the model and normalization configuration may need to be adjusted to use one channel.

## CNN Architecture

The classifier uses a compact CNN with:

- A convolutional layer producing 32 feature maps.
- ReLU activation and max pooling.
- A second convolutional layer producing 64 feature maps.
- ReLU activation and max pooling.
- Flattening followed by a fully connected layer with 128 units.
- ReLU activation and dropout.
- A final two-unit output layer for `Healthy` and `Tumor`.

For `224 x 224` inputs, the convolution and pooling configuration produces a flattened feature size of `64 * 56 * 56` before the first fully connected layer.

## Training and Evaluation

The notebook trains for 10 epochs using:

- Cross-entropy loss.
- Adam optimization.
- Learning rate of `0.001`.
- Batch size of `32`.

During each epoch, the notebook reports training loss and accuracy on the held-out test split. It also collects predictions for a confusion matrix showing correct and incorrect classifications for both classes.

The saved checkpoint can be loaded and moved to the selected device for additional predictions. The notebook includes examples for displaying a batch of predictions and classifying a standalone MRI image path.

## Setup

Install the dependencies listed in `requirements.txt` in a Python environment with Jupyter support. A CUDA-enabled PyTorch installation is recommended for faster training but is optional.

The notebook contains machine-specific Windows paths for the downloaded dataset and example images. Update those paths before running the experiment on another computer.

## Running the Notebook

1. Install the dependencies from `requirements.txt`.
2. Open `Trained CV Model.ipynb` in VS Code or Jupyter.
3. Confirm the KaggleHub dataset path and the class-folder structure.
4. Run the cells from top to bottom so that preprocessing, DataLoaders, the model, and training state are initialized in order.
5. Review the sample images, pixel statistics, class distribution, loss, accuracy, and confusion matrix.
6. Restore `MRI_Pred_Model_Accu_97.pth` for inference after training.
7. Pass a new MRI image path to the prediction helper to view the predicted class.

## Limitations

- The train/test split uses a random split without a separate validation set.
- The displayed accuracy depends on the dataset split, seed, preprocessing, and training settings.
- The dataset may not represent all scanners, acquisition protocols, patient populations, or tumor types.
- A confusion matrix and accuracy are useful summary metrics but do not establish clinical reliability.
- The model should be treated as an experimental classifier, not a tool for diagnosis or treatment decisions.

## Purpose

The goal of this project is to demonstrate a complete image-classification workflow with PyTorch: exploring MRI data, building a CNN from basic components, training it on labeled images, evaluating its predictions, saving the model, and using it for inference.
