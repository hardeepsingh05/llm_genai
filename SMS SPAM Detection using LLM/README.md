# SMS Spam Detection Using a Fine-Tuned GPT-2 Model

This project experiments with adapting the pretrained GPT-2 124M language model for binary SMS spam classification. It uses the SMS Spam Collection dataset, prepares balanced ham and spam examples, tokenizes messages with the GPT-2 tokenizer, and fine-tunes a classification head on top of the transformer.

## Workflow

1. Load and inspect the SMS Spam Collection dataset.
2. Balance the ham and spam classes by downsampling ham messages.
3. Encode `ham` as `0` and `spam` as `1`.
4. Shuffle and split the data into training, validation, and test sets.
5. Save the generated CSV splits.
6. Tokenize and pad SMS messages with the GPT-2 tokenizer.
7. Create PyTorch datasets and DataLoaders.
8. Recreate GPT-2 124M and load its pretrained checkpoint weights.
9. Replace the language-model output layer with a two-class classification head.
10. Freeze the base model and fine-tune the final transformer block, final normalization layer, and classification head.
11. Track loss and accuracy, plot metrics, classify sample messages, and save the trained checkpoint.

## Project Files

- `SMS Spam detection using LLM.ipynb` - Main experiment notebook.
- `sms+spam+collection/SMSSpamCollection` - Original tab-separated SMS dataset.
- `train.csv`, `validation.csv`, `test.csv` - Generated dataset splits.
- `loss-plot.pdf`, `accuracy-plot.pdf` - Generated metric plots.
- `Pretrained_Finetuned_LLM_model_and_optimizer.pth` - Generated model checkpoint.
- `requirements.txt` - Python dependencies.

## Dataset and Labels

The source dataset contains two categories:

- `ham` - A normal, non-spam SMS.
- `spam` - An unsolicited or promotional SMS.

The notebook balances the dataset by sampling the ham class to match the spam count. It then maps `ham` to `0` and `spam` to `1`, shuffles the records with a fixed seed, and creates approximately 70% training, 10% validation, and 20% test splits.

## Model Configuration

The notebook uses GPT-2 124M settings:

- Vocabulary size: 50,257
- Context length: 1,024 tokens
- Embedding size: 768
- Attention heads: 12
- Transformer layers: 12
- Dropout rate: 0.0 in the shown configuration
- Query, key, and value biases: enabled

The original GPT-2 output layer is replaced with a linear layer that produces two logits. The class with the highest final-token logit is returned as `spam` or `not spam`.

## Setup

Install the packages listed in `requirements.txt` in a Python environment that supports Jupyter notebooks. CUDA is optional, but a GPU is recommended because fine-tuning GPT-2 with 1,024-token sequences can require substantial memory.

The notebook includes local, machine-specific dataset and checkpoint paths. Update those paths before running it on another computer.

## GPT-2 Checkpoint

The model-loading section expects the GPT-2 checkpoint files, including checkpoint metadata, model weights, `hparams.json`, `encoder.json`, and `vocab.bpe`. The checkpoint size must match the `NEW_CONFIG` architecture; the main experiment is configured for GPT-2 124M.

## Running the Experiment

1. Install dependencies from `requirements.txt`.
2. Open `SMS Spam detection using LLM.ipynb` in VS Code or Jupyter.
3. Confirm the dataset and GPT-2 checkpoint paths.
4. Run the notebook cells from top to bottom.
5. Review the train, validation, and test metrics and generated plots.
6. Test new SMS messages with the classification helper.
7. Save or restore the generated model and optimizer checkpoint as needed.

## Limitations

- Downsampling removes many ham examples from the original dataset.
- Padding all messages to a common maximum length can increase memory usage.
- Accuracy alone does not fully describe spam-filter quality; precision, recall, F1 score, and a confusion matrix would provide additional insight.
- Results depend on the random seed, optimizer settings, training duration, and available hardware.
- Absolute paths in the notebook may require adjustment in a different environment.

## Purpose

This experiment demonstrates how a pretrained causal language model can be adapted to a practical binary classification task through GPT-2 tokenization, transformer weight loading, selective fine-tuning, evaluation, prediction, and checkpoint management.
