# Document Similarity Plagiarism Checker

A comprehensive, multi-language plagiarism detection tool that uses Natural Language Processing (NLP) techniques to identify similarities between documents and detect potential plagiarism from both uploaded files and online sources.

![Plagiarism Checker Screenshot](https://placehold.co/600x400?text=Plagiarism+Checker)

## Features

- **Advanced Text Processing**: Cleans and normalizes text with language-specific tokenization, stopword removal, and stemming
- **Multi-Language Support**: Analyzes documents in English, French, and Spanish
- **TF-IDF Vectorization**: Transforms documents into numerical vectors that capture word importance
- **Cosine Similarity Analysis**: Calculates similarity percentages between document pairs
- **Intelligent Phrase Detection**: Uses context-aware algorithms to identify meaningful matching content
- **Context Highlighting**: Shows matched phrases with surrounding context for better understanding
- **Online Repository Integration**: Checks documents against web and scholarly sources for potential matches
- **Multi-Source Searching**: Supports Google web search and academic repositories like arXiv and Semantic Scholar
- **Visualization**: Generates heatmaps to visualize similarity relationships across multiple documents
- **Multiple File Format Support**: Analyzes PDF, DOCX, DOC, RTF, ODT, and TXT files
- **Flexible Interface**: Available as both a web application and command-line tool

## Installation

### Prerequisites

- Python 3.6+
- Pip package manager

### Option 1: Standard Installation

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

### Option 2: Docker Installation

For an easier setup, you can use Docker:

1. Build the Docker image:
```bash
docker build -t plagiarism-checker .
```

2. Run the container:
```bash
docker run -p 5000:5000 plagiarism-checker
```

## Usage

### Web Interface

1. Start the web server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload documents, select language, and configure analysis options
4. Review detailed similarity results and online matches

### Command Line

The plagiarism checker can be used in several ways:

1. **Compare files in a directory**:

```bash
python plagiarism_core.py --directory path/to/documents --threshold 30 --visualize --phrases --language english
```

2. **Compare specific files**:

```bash
python plagiarism_core.py --files doc1.txt doc2.txt doc3.txt --threshold 40 --visualize
```

3. **Check documents against online sources**:

```bash
python plagiarism_core.py --files research_paper.pdf --online
```

4. **Use different language settings**:

```bash
python plagiarism_core.py --files document_fr.txt --language french --online --scholarly-only
```

### Command Line Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--directory` | `-d` | Directory containing documents to compare |
| `--files` | `-f` | Specific files to compare (space-separated) |
| `--threshold` | `-t` | Similarity threshold percentage to flag (default: 30.0) |
| `--visualize` | `-v` | Generate similarity heatmap visualization |
| `--phrases` | `-p` | Find common phrases between documents |
| `--formats` | | List all supported file formats |
| `--online` | `-o` | Check documents against online sources |
| `--api-keys` | | Path to JSON file with API keys for online services |
| `--scholarly-only` | | Only check scholarly repositories, not general web search |
| `--no-google` | | Disable Google search for online checking |
| `--language` | `-l` | Document language: english, french, or spanish (default: english) |

### API Usage

The web application also provides a REST API for programmatic access:

#### Check Text Against Online Sources

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

#### Compare Text Against Other Documents

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

## Supported File Formats

The plagiarism checker supports a variety of document formats:

- **.txt** - Plain text files
- **.pdf** - PDF documents (including scanned PDFs with text layers)
- **.docx** - Microsoft Word documents (Office Open XML)
- **.doc** - Microsoft Word legacy format
- **.rtf** - Rich Text Format
- **.odt** - OpenDocument Text format

## Language Support

The plagiarism checker supports analyzing documents in multiple languages:

- **English**: Full support with Porter Stemmer
- **French**: Uses FrenchStemmer and French stopwords
- **Spanish**: Uses SnowballStemmer('spanish') and Spanish stopwords

## How It Works

1. **Text Preprocessing**:
   - Converts text to lowercase
   - Removes special characters and numbers
   - Tokenizes into words using language-specific tokenizers
   - Removes language-specific stopwords
   - Applies appropriate stemmer based on language

2. **Vectorization**:
   - Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert documents into numerical vectors
   - Words that are rare across all documents but common in specific documents receive higher weights

3. **Similarity Calculation**:
   - Computes cosine similarity between document vectors
   - Returns a similarity score between 0 (completely different) and 1 (identical)

4. **Intelligent Phrase Detection**:
   - Uses a sliding window approach to find matching sequences of words
   - Filters out common, generic phrases that aren't meaningful indicators of plagiarism
   - Applies a sophisticated scoring algorithm based on:
     - Phrase length (longer matches are more significant)
     - Word rarity (phrases with uncommon words score higher)
     - Uniqueness (phrases with more unique words are prioritized)
   - Provides context by showing surrounding text for each match
   - Eliminates overlapping matches to prevent redundancy

5. **Online Repository Checking**:
   - Extracts key phrases from the document for targeted searching
   - Searches multiple online sources in the document's language:
     - General web search (using Google)
     - Academic repositories (arXiv, Semantic Scholar)
   - Uses intelligent chunking to break large documents into searchable segments
   - Filters and ranks results based on relevance
   - Provides URLs and context snippets for potential matches

## Use Cases

- **Academic Integrity**: Check student papers for plagiarism against both other students' work and online sources
- **Content Publishing**: Ensure originality of articles or blog posts before publication
- **Research**: Compare research papers for similarities and check for proper citation of sources
- **Legal Document Analysis**: Find overlapping content in legal texts
- **Academic Research**: Verify the originality of theses and dissertations against scholarly repositories
- **Content Marketing**: Ensure your content isn't duplicating existing online material
- **Editorial Review**: Screen submitted articles for potential plagiarism before publication
- **Multi-Language Documents**: Check documents in English, French, and Spanish for plagiarism

## Security and Privacy

- All uploaded files and analysis results are stored temporarily and deleted after the session expires
- The web application runs locally by default and does not share your documents with any external service
- When using the online checking feature, only key phrases from your documents are sent to search engines, not the full text

## Future Improvements

- Support for additional languages (German, Italian, Portuguese)
- OCR support for scanned documents without text layers
- AI-powered content paraphrase detection
- Citation analysis and verification
- Bulk processing capabilities for large document sets
- PDF report generation with highlighted matches
- Browser extension for quick content checking

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
