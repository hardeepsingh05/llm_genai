# Experimental RAG Chatbot with Input Documents

This project is an experiment in building a retrieval-augmented generation (RAG) chatbot over a technical PDF document. The notebook uses a Volvo truck service manual as the input source, extracts structured text and images, creates semantic embeddings, indexes the content with FAISS, and uses a local open-source language model to answer questions from the retrieved context.

The goal is to understand the main building blocks of a RAG system rather than to provide a production-ready chatbot.

## RAG Pipeline

The notebook follows this flow:

1. Install and load the libraries needed for parsing, embeddings, vector search, and local inference.
2. Download and initialize the Phi-3 Mini GGUF generation model.
3. Parse the PDF and identify headings, sections, subsections, content, cautions, and extracted images.
4. Flatten the structured document into text chunks that retain their hierarchy path.
5. Generate embeddings for each chunk with `all-MiniLM-L6-v2`.
6. Store the chunks and embeddings in a FAISS vector database.
7. Configure a LangChain `RetrievalQA` chain with a grounded prompt.
8. Ask questions about the indexed service manual and inspect the generated answers.

## Files

- `RAG with Input Documents.ipynb` - Main RAG experiment notebook.
- `volvo-trucks-basic-service-manual-pages.pdf` - Input technical document.
- `requirements.txt` - Python dependencies used by the notebook.

The notebook may create an `output_images/` directory when images are extracted from the PDF.

## Document Parsing

The PDF parser uses PyMuPDF to inspect text spans and image blocks. It applies formatting heuristics based on font size, character flags, and text coordinates to classify content into a nested structure:

- Delivery type
- Section
- Subsection
- Main content
- Caution or danger notes
- Associated images

This structure is useful because retrieval can preserve where a passage came from instead of storing only unlabelled text.

The parser is intentionally heuristic. Changes in PDF layout, fonts, or page structure may require adjustments to the thresholds used by the extraction function.

## Embeddings and FAISS Retrieval

The flattened document chunks are converted into vector embeddings using the `all-MiniLM-L6-v2` sentence-transformer model. Similar questions and passages should have nearby vector representations. FAISS provides efficient nearest-neighbor search over those vectors and supplies relevant chunks to the generation chain.

This notebook uses a simple vector-retrieval setup. Retrieval quality can be improved later with chunk-size experiments, metadata filters, reranking, hybrid lexical and semantic search, or a persistent vector store.

## Generation Model

The generation component is a Phi-3 Mini instruction model in GGUF format, loaded through `llama-cpp-python` and LangChain's `LlamaCpp` wrapper. The notebook configures GPU layers, a context window, a token limit, and a fixed seed for repeatable experimentation where the runtime supports those settings.

The GGUF model file is downloaded separately from Hugging Face in the notebook. It is not included in this folder because model files are large.

## RAG Prompt Behavior

The prompt supplies the retrieved passages under a `Context` heading and asks the model to answer the question using that context. It also instructs the model to respond with `Sorry, I don't know` when the answer is not supported by the retrieved information.

This is a useful starting guardrail, but it does not guarantee factuality. The quality of the final answer depends on both retrieval quality and the generation model's ability to follow the prompt.

## Example Questions

The notebook queries the system about:

- Alternative clutches.
- PTO servicing cautions.
- Recommended driveshaft grease.
- Driveshaft safety cautions.
- Air dryer cartridge maintenance.

These questions exercise both semantic retrieval and grounded answer generation over different parts of the manual.

## Requirements and Setup

Install the packages listed in `requirements.txt` in a Python environment with Jupyter support. `llama-cpp-python` may require platform-specific build tools or a compatible prebuilt wheel. GPU-enabled installation can require additional configuration and differs by operating system.

The notebook currently contains hosted-runtime paths such as `/content/...`. Update the PDF path and GGUF model path before running it locally.

A typical setup is:

```bash
pip install -r requirements.txt
```

The notebook's first cells also contain installation commands reflecting the versions used during the original experiment.

## Running the Notebook

1. Install the dependencies.
2. Make the PDF available at the configured path.
3. Download or provide the Phi-3 Mini GGUF model file.
4. Run the notebook cells from top to bottom.
5. Inspect the extracted structure and chunks before indexing.
6. Build the FAISS vector store and initialize the RAG chain.
7. Try the example questions or ask questions relevant to the source manual.

The notebook has not been changed beyond the added documentation markdown cells. Its code remains an experimental reference implementation.

## Limitations and Future Experiments

- The document parser relies on font and coordinate heuristics.
- The current chunking approach is based on flattened structured fields rather than configurable semantic chunk sizes.
- The vector index is created in memory and is not persisted by the notebook.
- Retrieval quality is not measured with a labelled evaluation set.
- The generation model and embedding model may require substantial memory.
- Extracted images are stored, but the current retrieval prompt is primarily text-based.
- Future experiments could compare embedding models, tune retriever parameters, add reranking, preserve richer metadata, persist the index, and evaluate answer faithfulness.

## Purpose

This project demonstrates the core RAG pattern: retrieve relevant information from an external knowledge source and provide it to a language model as context before generating an answer. It is a practical sandbox for learning how document ingestion, semantic search, prompt design, and local model inference fit together.
