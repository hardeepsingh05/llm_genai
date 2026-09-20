# Instruction-Based Fine-Tuning of GPT-2

This project experiments with supervised instruction fine-tuning for a pretrained GPT-2 language model. The notebook uses instruction, optional input, and response examples to teach the model how to generate task-specific answers in an Alpaca-style prompt format.

The project is primarily educational: it demonstrates the full path from structured instruction data to tokenized causal-language-model training, checkpoint restoration, generated responses, and automated response scoring.

## Project Workflow

1. Load the GPT-2 tokenizer and select CPU or CUDA processing.
2. Read the instruction dataset containing instruction, input, and output fields.
3. Format each example into a prompt with `### Instruction`, optional `### Input`, and `### Response` sections.
4. Split the examples into training, test, and validation subsets.
5. Tokenize complete prompt-response sequences with the GPT-2 tokenizer.
6. Build custom collation functions that pad sequences, shift targets by one token, and mask extra padding tokens with `-100` so they are ignored by cross-entropy loss.
7. Create PyTorch datasets and DataLoaders for each split.
8. Recreate the GPT-2 architecture and load pretrained GPT-2 checkpoint parameters.
9. Measure the initial loss, fine-tune the model with AdamW, and periodically evaluate training and validation loss.
10. Generate sample responses after training and visualize the loss history.
11. Save the trained model and optimizer state for later restoration.
12. Generate responses for test examples and store them with the original instructions and expected outputs.
13. Optionally use a locally running Ollama model to score generated responses and calculate an average test score.

## Main Files

- `Instruction-Based Fine-Tuning.ipynb` - Main experiment notebook.
- `instruction-data.json` - Source instruction, input, and expected-response examples.
- `intruction-data-with-response.json` - Test examples augmented with generated responses.
- `trained_model_and_optimizer.pth` - Saved model and optimizer state after fine-tuning.
- `requirements.txt` - Python packages used by the notebook.

Generated files may be saved to a different location when the notebook is run in a hosted environment or from a different working directory.

## Prompt Format

Each training example is converted into a structured prompt similar to:

```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
<instruction text>

### Input:
<optional input text>

### Response:
<expected response>
```

The input section is omitted when an example does not provide additional context. During training, the model learns to continue the formatted prompt with the expected response.

## Model and Training Details

The notebook reconstructs GPT-2 from its published checkpoint parameters. The configuration shown in the experiment uses GPT-2 355M settings:

- Vocabulary size: 50,257 tokens
- Context length: 1,024 tokens
- Embedding size: 1,024
- Attention heads: 16
- Transformer layers: 24
- Query, key, and value biases: enabled

Training uses teacher forcing. Input token IDs and target token IDs are offset by one position, allowing the model to learn the next token at every position. Padding positions after the meaningful sequence are replaced with `-100`, which PyTorch's cross-entropy loss ignores.

## Requirements

Install the packages listed in `requirements.txt` in a Python environment suitable for the notebook. The core dependencies are PyTorch, tiktoken, NumPy, TensorFlow, requests, tqdm, matplotlib, ipykernel, and psutil.

GPU acceleration is optional but strongly recommended for fine-tuning a GPT-2 355M model. The notebook selects CUDA when it is available and otherwise uses the CPU.

## GPT-2 Checkpoint Files

The model-loading utilities expect the GPT-2 checkpoint files, including the model checkpoint shards, `hparams.json`, `encoder.json`, and `vocab.bpe`. The notebook can download and load the requested GPT-2 model size when network access is available, or it can use checkpoint files already stored locally.

The model size used in the main experiment must match the architecture configuration and the downloaded checkpoint. For example, GPT-2 355M requires the 355M embedding, attention-head, and transformer-layer settings shown above.

## Running the Notebook

1. Install the dependencies from `requirements.txt`.
2. Open the notebook in VS Code or Jupyter.
3. Set the instruction-data path and GPT-2 checkpoint path for the current machine.
4. Run cells from top to bottom so that tokenizers, datasets, model parameters, and training functions are initialized in order.
5. Review the training and validation loss plots and generated sample responses.
6. Restore the saved checkpoint when continuing training or generating additional responses.

The notebook contains paths from both hosted and local environments. These paths may need to be updated before running the experiment on another machine.

## Optional Ollama Evaluation

The final evaluation section can send scoring prompts to a local Ollama HTTP endpoint. This requires:

- Ollama to be installed and running locally.
- A compatible evaluation model, such as `llama3`, to be available to Ollama.
- The local API endpoint to be reachable at `http://localhost:11434/api/chat`.

This evaluator provides a qualitative reference score for generated answers. It should be treated as an additional experiment rather than a replacement for human evaluation or task-specific metrics.

## Limitations

- Fine-tuning GPT-2 355M can require considerable GPU memory and runtime.
- The source data split is positional after calculating split sizes, so shuffling or a stratified split may be useful for other datasets.
- Response quality depends heavily on the instruction-data quality and coverage.
- Absolute paths in the notebook are environment-specific.
- Automatic scoring by another language model can be inconsistent and should be interpreted alongside manual inspection.

## Purpose of the Experiment

The project shows how a causal language model can be adapted from general next-token prediction to instruction following. It focuses on understanding prompt templates, dynamic batching, target masking, checkpoint management, generation, and practical evaluation of a fine-tuned language model.
