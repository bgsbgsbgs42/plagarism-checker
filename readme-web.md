# Document Similarity Plagiarism Checker - Web Interface

This is the web interface version of the Document Similarity Plagiarism Checker, providing an easy-to-use GUI for analyzing documents for potential plagiarism.

## Features

- **User-Friendly Interface**: Simple upload and configuration process
- **Multi-Language Support**: Analyze documents in English, French, and Spanish
- **Document Comparison**: Compare multiple documents against each other
- **Online Repository Checking**: Search the web and scholarly repositories for potential matches
- **Interactive Results**: View similarity scores, common phrases, and matched content
- **Visualization**: Generate heatmaps to visualize similarity between documents
- **Flexible File Support**: Analyze various file formats including PDF, DOCX, DOC, RTF, ODT, and TXT

## Installation

### Standard Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/plagiarism-checker.git
cd plagiarism-checker
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('snowball_data')"
```

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to:
```
http://localhost:5000
```

### Docker Installation

For an easier setup, you can use Docker:

1. Build the Docker image:
```bash
docker build -t plagiarism-checker .
```

2. Run the container:
```bash
docker run -p 5000:5000 plagiarism-checker
```

3. Access the web interface at:
```
http://localhost:5000
```

## API Usage

The application also provides a REST API for programmatic access:

### Check Text Against Online Sources

```bash
curl -X POST http://localhost:5000/api/check \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text content here...",
    "language": "en",
    "check_online": true,
    "check_scholarly": true
  }'
```

### Compare Text Against Other Documents

```bash
curl -X POST http://localhost:5000/api/check \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your primary text content here...",
    "language": "en",
    "comparison_texts": [
      "First document to compare against",
      "Second document to compare against"
    ]
  }'
```

## API Key Configuration

For enhanced online checking capabilities, create a file named `api_keys.json` in the application root directory:

```json
{
  "semanticscholar": "your-api-key-here"
}
```

## Language Support

The plagiarism checker supports analyzing documents in multiple languages:

- **English**: Full support with optimized tokenization and stopword removal
- **French**: Support for language-specific stemming, tokenization, and stopwords
- **Spanish**: Support for language-specific stemming, tokenization, and stopwords

## Usage Tips

1. **For Academic Papers**: Enable "Check against scholarly repositories" for better results.
2. **For Multiple Documents**: Upload all documents at once to compare them against each other.
3. **For Single Document Checking**: Use the online source checking feature.
4. **Adjust Threshold**: Lower the similarity threshold for stricter plagiarism detection.
5. **Phrase Detection**: Always enable "Detect common phrases" for more detailed results.

## Security and Privacy

- All uploaded files and analysis results are stored temporarily and deleted after the session expires.
- The web application runs locally by default and does not share your documents with any external service.
- When using the online checking feature, only key phrases from your documents are sent to search engines, not the full text.

## Troubleshooting

- **OCR Issues**: If text extraction from PDFs fails, ensure your PDF contains a text layer or use a dedicated OCR tool first.
- **Large Files**: For very large documents, break them into smaller sections for more efficient processing.
- **Rate Limiting**: If you encounter errors when checking against online sources, wait a few minutes as search engines may temporarily rate-limit requests.
