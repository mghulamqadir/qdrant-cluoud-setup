# Qdrant Text Embedding and Indexing

This repository demonstrates how to embed text documents and index them in Qdrant Cloud using Hugging Face embeddings and LangChain tools. The project includes an example of processing a PDF document, splitting it into chunks, embedding the text, and managing a Qdrant collection.

## Features

- **Document Loading**: Load PDF documents using `PyPDFLoader`.
- **Text Splitting**: Chunk large documents into smaller, overlapping text segments for better embedding and retrieval.
- **Embeddings**: Generate embeddings with Hugging Face models (`BAAI/bge-large-en-v1.5`).
- **Vector Database**: Manage collections in Qdrant Cloud.
- **Environment Variable Management**: Use `.env.local` to securely store Qdrant API credentials.

## Prerequisites

### Tools and Libraries
- Python 3.8+
- [LangChain](https://docs.langchain.com/)
- [Qdrant Client](https://github.com/qdrant/qdrant_client)
- [PyPDFLoader](https://github.com/hwchase17/langchain)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [python-dotenv](https://github.com/theskumar/python-dotenv)

### Environment Variables
Create a `.env` file to store your Qdrant Cloud API credentials:
```env
QDRANT_ENDPOINT_URL=<Your Qdrant Cloud Endpoint URL>
QDRANT_API_KEY=<Your Qdrant API Key>
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/qdrant-text-indexing.git
   cd qdrant-text-indexing
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Add your Qdrant credentials to `.env.local`.

## Usage

### Steps

1. **Initialize Qdrant Client and Collection**
   - Checks if a specified collection exists.
   - Deletes the existing collection (if applicable).
   - Creates a new collection with the appropriate vector size and distance metric.

2. **Load and Process PDF Document**
   - Use `PyPDFLoader` to extract text from a PDF file.
   - Split the document into smaller chunks for improved retrieval performance.

3. **Generate Embeddings**
   - Configure the embedding model (`BAAI/bge-large-en-v1.5`) with device support for GPU or CPU.

4. **Index Embeddings in Qdrant**
   - Embed the text and upload the vectors to Qdrant Cloud.

### Example
Run the script to embed and index text from a PDF document:
```bash
python index_texts.py
```
Replace `risk.pdf` with your own PDF file.

## Output

- Confirms successful collection creation and indexing.
- Displays collection details such as vector size and distance metric.

## Notes

- The script automatically normalizes embeddings for consistency.
- GPU support is detected and utilized if available.

## Troubleshooting

- Ensure your `.env` file is correctly configured.
- Verify that your Qdrant Cloud endpoint URL and API key are valid.
- Install all required Python dependencies listed in `requirements.txt`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributions

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

