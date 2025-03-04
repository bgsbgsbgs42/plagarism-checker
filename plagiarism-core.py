#!/usr/bin/env python3
"""
Core functionality for the Plagiarism Checker with multi-language support.
This module contains the main classes and functions for text analysis,
similarity detection, and online repository checking.
"""

import os
import re
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, FrenchStemmer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import docx
import PyPDF2
import textract
import magic

# Download required NLTK resources for multiple languages
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


def extract_text_from_file(file_path):
    """Extract text from various file formats."""
    # Get file extension
    _, ext = os.path.splitext(file_path.lower())
    
    try:
        # Detect file type using python-magic for more accurate detection
        file_type = magic.from_file(file_path, mime=True)
        
        # Plain text files
        if ext == '.txt' or 'text/plain' in file_type:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                return file.read()
        
        # PDF files
        elif ext == '.pdf' or 'application/pdf' in file_type:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text() + "\n"
            
            # If PyPDF2 fails to extract text (e.g., from scanned PDFs), try textract
            if text.strip() == "":
                text = textract.process(file_path, method='pdfminer').decode('utf-8', errors='replace')
            
            return text
        
        # Word documents
        elif ext == '.docx' or 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in file_type:
            doc = docx.Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        
        # DOC files (older Word format)
        elif ext == '.doc' or 'application/msword' in file_type:
            return textract.process(file_path).decode('utf-8', errors='replace')
        
        # RTF files
        elif ext == '.rtf' or 'application/rtf' in file_type:
            return textract.process(file_path).decode('utf-8', errors='replace')
        
        # Other text-based formats that textract can handle
        else:
            try:
                return textract.process(file_path).decode('utf-8', errors='replace')
            except:
                print(f"Warning: Could not extract text from {file_path}. Unsupported format.")
                return ""
    
    except Exception as e:
        print(f"Error extracting text from {file_path}: {str(e)}")
        return ""


class OnlineRepositoryChecker:
    """Class to handle checking content against online repositories"""
    
    def __init__(self, api_keys=None):
        """Initialize with optional API keys for premium services"""
        self.api_keys = api_keys or {}
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
        ]
        self.session = requests.Session()
        self.results_cache = {}  # Cache results to avoid duplicate searches
    
    def _get_random_user_agent(self):
        """Return a random user agent to avoid detection"""
        return np.random.choice(self.user_agents)
    
    def _get_text_fingerprint(self, text):
        """Generate a fingerprint for text to use as cache key"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _extract_key_phrases(self, text, num_phrases=3, min_length=5, max_length=15, language='english'):
        """Extract key phrases from text for targeted searching"""
        try:
            stop_words = set(stopwords.words(language))
        except:
            stop_words = set(stopwords.words('english'))  # Fallback
            
        sentences = sent_tokenize(text)
        
        # Tokenize and clean sentences
        tokenized_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 2]
            tokenized_sentences.append(words)
        
        # Extract phrases of varying length
        phrases = []
        for sentence in tokenized_sentences:
            if len(sentence) < min_length:
                continue
                
            for start in range(len(sentence) - min_length + 1):
                for length in range(min_length, min(max_length, len(sentence) - start + 1)):
                    phrase = ' '.join(sentence[start:start+length])
                    # Ensure phrase has enough substance (not just common words)
                    if len(phrase) > 20:  # At least 20 characters
                        phrases.append(phrase)
        
        # If we couldn't get enough substantial phrases, fall back to sentences
        if len(phrases) < num_phrases:
            phrases = [' '.join(s[:min(15, len(s))]) for s in tokenized_sentences if len(s) >= 5]
        
        # Select diverse phrases by maximizing word variation
        if len(phrases) <= num_phrases:
            return phrases
            
        selected_phrases = []
        all_words = set()
        
        # Greedily select phrases that add the most new words
        phrases_with_scores = []
        for phrase in phrases:
            phrase_words = set(phrase.split())
            new_words = len(phrase_words - all_words)
            phrases_with_scores.append((phrase, new_words))
        
        # Sort by number of new words (descending)
        phrases_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top phrases
        return [p[0] for p in phrases_with_scores[:num_phrases]]
    
    def search_google(self, text, num_results=5, language='english'):
        """Search Google for potential matches using key phrases from the text"""
        # Check cache first
        fingerprint = self._get_text_fingerprint(text)
        if fingerprint in self.results_cache:
            return self.results_cache[fingerprint].get('google', [])
        
        # Extract key phrases for searching
        key_phrases = self._extract_key_phrases(text, language=language)
        all_results = []
        
        # Set up headers to avoid being blocked
        headers = {
            'User-Agent': self._get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Search for each key phrase
        for phrase in key_phrases:
            try:
                # Construct search URL
                query = quote_plus(f'"{phrase}"')  # Use quotes for exact matching
                url = f'https://www.google.com/search?q={query}&num={num_results}'
                
                # Send request
                response = self.session.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Parse results
                soup = BeautifulSoup(response.text, 'html.parser')
                search_results = []
                
                # Extract search results (this may need adjustment as Google's HTML structure changes)
                for result in soup.select('.tF2Cxc'):
                    title_element = result.select_one('h3')
                    link_element = result.select_one('a')
                    snippet_element = result.select_one('.VwiC3b')
                    
                    if title_element and link_element and snippet_element:
                        title = title_element.get_text()
                        link = link_element.get('href')
                        if link.startswith('/url?q='):
                            link = link.split('/url?q=')[1].split('&')[0]
                        snippet = snippet_element.get_text()
                        
                        search_results.append({
                            'title': title,
                            'url': link,
                            'snippet': snippet,
                            'matched_phrase': phrase
                        })
                
                all_results.extend(search_results)
                
                # Respect rate limits
                time.sleep(2)
                
            except Exception as e:
                print(f"Error searching Google for phrase '{phrase}': {str(e)}")
        
        # Remove duplicates by URL
        unique_results = []
        seen_urls = set()
        for result in all_results:
            if result['url'] not in seen_urls:
                unique_results.append(result)
                seen_urls.add(result['url'])
        
        # Store in cache
        if fingerprint not in self.results_cache:
            self.results_cache[fingerprint] = {}
        self.results_cache[fingerprint]['google'] = unique_results[:num_results]
        
        return unique_results[:num_results]
    
    def search_scholarly_repositories(self, text, repositories=None, language='english'):
        """Search academic repositories for potential matches"""
        # Default repositories to search
        default_repositories = ['arxiv', 'core.ac.uk', 'semanticscholar']
        repositories = repositories or default_repositories
        
        fingerprint = self._get_text_fingerprint(text)
        if fingerprint in self.results_cache:
            return self.results_cache[fingerprint].get('scholarly', [])
        
        # Extract key phrases that are more likely to be academic
        key_phrases = self._extract_key_phrases(text, num_phrases=2, min_length=6, max_length=20, language=language)
        all_results = []
        
        headers = {
            'User-Agent': self._get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        }
        
        # Search each repository
        for repository in repositories:
            try:
                if repository == 'arxiv':
                    # ArXiv API query
                    for phrase in key_phrases:
                        query = quote_plus(f'"{phrase}"')
                        url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5'
                        
                        response = self.session.get(url, headers=headers, timeout=15)
                        response.raise_for_status()
                        
                        # Parse ArXiv XML response
                        soup = BeautifulSoup(response.text, 'xml')
                        for entry in soup.find_all('entry'):
                            title = entry.find('title').text.strip()
                            url = entry.find('id').text.strip()
                            summary = entry.find('summary').text.strip()
                            
                            all_results.append({
                                'title': title,
                                'url': url,
                                'snippet': summary[:200] + '...' if len(summary) > 200 else summary,
                                'repository': 'arXiv',
                                'matched_phrase': phrase
                            })
                
                elif repository == 'semanticscholar':
                    # Semantic Scholar API query
                    if 'semanticscholar' in self.api_keys:
                        api_key = self.api_keys['semanticscholar']
                        headers['x-api-key'] = api_key
                    
                    for phrase in key_phrases:
                        query = quote_plus(phrase)
                        url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5&fields=title,abstract,url'
                        
                        response = self.session.get(url, headers=headers, timeout=15)
                        response.raise_for_status()
                        
                        data = json.loads(response.text)
                        for paper in data.get('data', []):
                            title = paper.get('title', 'Unknown Title')
                            paper_url = paper.get('url', '')
                            abstract = paper.get('abstract', 'No abstract available')
                            
                            all_results.append({
                                'title': title,
                                'url': paper_url,
                                'snippet': abstract[:200] + '...' if len(abstract) > 200 else abstract,
                                'repository': 'Semantic Scholar',
                                'matched_phrase': phrase
                            })
                
                # Add more repositories as needed
                
                # Respect rate limits
                time.sleep(2)
                
            except Exception as e:
                print(f"Error searching {repository} for matches: {str(e)}")
        
        # Remove duplicates and store in cache
        unique_results = []
        seen_urls = set()
        for result in all_results:
            if result['url'] not in seen_urls:
                unique_results.append(result)
                seen_urls.add(result['url'])
        
        if fingerprint not in self.results_cache:
            self.results_cache[fingerprint] = {}
        self.results_cache[fingerprint]['scholarly'] = unique_results
        
        return unique_results
    
    def check_document_against_online_sources(self, document_text, check_google=True, check_scholarly=True, language='english'):
        """
        Check a document against online sources using multiple search methods
        Returns potential matches from various sources
        """
        results = {
            'google_matches': [],
            'scholarly_matches': [],
        }
        
        # Split document into chunks for better searching
        chunks = self._split_document_into_chunks(document_text)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            
            # Submit search tasks for each chunk
            for chunk in chunks:
                if check_google:
                    futures.append(executor.submit(self.search_google, chunk, language=language))
                if check_scholarly:
                    futures.append(executor.submit(self.search_scholarly_repositories, chunk, language=language))
            
            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        # Determine which type of result based on structure
                        if any('repository' in r for r in result if isinstance(r, dict)):
                            results['scholarly_matches'].extend(result)
                        else:
                            results['google_matches'].extend(result)
                except Exception as e:
                    print(f"Error getting search results: {str(e)}")
        
        # Remove duplicates
        results['google_matches'] = self._deduplicate_results(results['google_matches'])
        results['scholarly_matches'] = self._deduplicate_results(results['scholarly_matches'])
        
        return results
    
    def _split_document_into_chunks(self, text, chunk_size=1000, overlap=200):
        """Split a document into overlapping chunks for more effective searching"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk of specified size
            end = start + chunk_size
            
            # Adjust end to not break in the middle of a sentence
            if end < len(text):
                # Look for a good breaking point (period followed by space)
                breakpoint = text.rfind('. ', start, end) + 2
                if breakpoint > start + 100:  # Ensure chunk isn't too small
                    end = breakpoint
            
            chunks.append(text[start:min(end, len(text))])
            start = end - overlap  # Create overlap for better coverage
        
        return chunks
    
    def _deduplicate_results(self, results):
        """Remove duplicate results based on URL"""
        unique_results = []
        seen_urls = set()
        
        for result in results:
            if result['url'] not in seen_urls:
                unique_results.append(result)
                seen_urls.add(result['url'])
        
        return unique_results


class PlagiarismChecker:
    """Main class for plagiarism detection with multi-language support"""
    
    def __init__(self, online_checker=None, language='english'):
        """Initialize the plagiarism checker with language support"""
        self.language = language
        
        # Set up language-specific resources
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            print(f"Warning: Stopwords not available for language '{language}', falling back to English")
            self.stop_words = set(stopwords.words('english'))
        
        # Choose appropriate stemmer based on language
        if language == 'french':
            self.stemmer = FrenchStemmer()
        elif language == 'spanish':
            self.stemmer = SnowballStemmer('spanish')
        else:
            # Default to Porter Stemmer for English and fallback for other languages
            self.stemmer = PorterStemmer()
        
        # Initialize online checker
        self.online_checker = online_checker
    
    def preprocess_text(self, text):
        """Clean and preprocess text for comparison."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize
        try:
            tokens = word_tokenize(text, language=self.language[:2])  # Use language code (first 2 chars)
        except:
            tokens = word_tokenize(text)  # Fallback to default
            
        # Remove stopwords and stem
        preprocessed_tokens = [
            self.stemmer.stem(token) for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        return ' '.join(preprocessed_tokens)
    
    def preprocess_documents(self, documents):
        """Preprocess a list of documents."""
        return [self.preprocess_text(doc) for doc in documents]
    
    def check_similarity_tfidf(self, documents, file_names=None):
        """
        Check document similarity using TF-IDF and cosine similarity.
        Returns a similarity matrix and a dataframe with pairwise scores.
        """
        # Apply preprocessing
        preprocessed_docs = self.preprocess_documents(documents)
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Create a more readable output format
        results = []
        for i in range(len(documents)):
            for j in range(i+1, len(documents)):
                doc1 = file_names[i] if file_names else f"Document {i+1}"
                doc2 = file_names[j] if file_names else f"Document {j+1}"
                similarity = similarity_matrix[i][j] * 100  # Convert to percentage
                
                # Find common phrases if similarity is notable
                common_phrases = []
                if similarity >= 20:  # Only compute phrases if similarity is significant
                    common_phrases = self.find_common_phrases(documents[i], documents[j])
                    
                results.append({
                    'Document 1': doc1,
                    'Document 2': doc2,
                    'Similarity (%)': round(similarity, 2),
                    'Common Phrases': common_phrases[:5]  # Store top 5 phrases
                })
                
        return similarity_matrix, results
    
    def find_common_phrases(self, doc1, doc2, min_length=4, max_length=25):
        """
        Find common phrases between two documents using an improved sliding window 
        approach with intelligent filtering.
        
        Args:
            doc1, doc2: The documents to compare
            min_length: Minimum length of phrases to consider (in words)
            max_length: Maximum length of phrases to check for (in words)
            
        Returns:
            List of dictionaries containing phrase details (text, positions, length)
        """
        # Ensure we're working with clean text
        doc1 = doc1.lower()
        doc2 = doc2.lower()
        
        # Tokenize at sentence level first
        try:
            sentences1 = sent_tokenize(doc1, language=self.language[:2])
            sentences2 = sent_tokenize(doc2, language=self.language[:2])
        except:
            # Fallback to default tokenizer if language-specific fails
            sentences1 = sent_tokenize(doc1)
            sentences2 = sent_tokenize(doc2)
        
        # Preprocess and get word tokens for each sentence
        processed_sentences1 = []
        for s in sentences1:
            # Remove punctuation but preserve sentence structure
            clean_s = re.sub(r'[^\w\s]', '', s)
            try:
                tokens = word_tokenize(clean_s, language=self.language[:2])
            except:
                tokens = word_tokenize(clean_s)  # Fallback
            processed_sentences1.append([t for t in tokens if len(t) > 1])
            
        processed_sentences2 = []
        for s in sentences2:
            clean_s = re.sub(r'[^\w\s]', '', s)
            try:
                tokens = word_tokenize(clean_s, language=self.language[:2])
            except:
                tokens = word_tokenize(clean_s)  # Fallback
            processed_sentences2.append([t for t in tokens if len(t) > 1])
        
        # Track found phrases with positions
        found_phrases = []
        
        # We'll create a hash-based approach for faster matching
        # First, create n-gram hashes from the second document for all valid sizes
        ngram_hashes = {}
        sentence_idx2 = 0
        
        for sentence in processed_sentences2:
            word_idx2 = 0
            for n in range(min_length, min(max_length + 1, len(sentence) + 1)):
                for i in range(len(sentence) - n + 1):
                    ngram = tuple(sentence[i:i+n])
                    
                    # Filter out n-grams that are mostly stopwords
                    stopword_count = sum(1 for word in ngram if word in self.stop_words)
                    if stopword_count / len(ngram) > 0.7:  # Skip if >70% stopwords
                        continue
                    
                    # Store all positions where this n-gram occurs
                    if ngram not in ngram_hashes:
                        ngram_hashes[ngram] = []
                    ngram_hashes[ngram].append((sentence_idx2, word_idx2 + i, n))
            
            sentence_idx2 += 1
            word_idx2 += len(sentence)
        
        # Now check document 1 against the hash table
        sentence_idx1 = 0
        for sentence in processed_sentences1:
            word_idx1 = 0
            for n in range(min_length, min(max_length + 1, len(sentence) + 1)):
                for i in range(len(sentence) - n + 1):
                    ngram = tuple(sentence[i:i+n])
                    
                    # Filter out n-grams that are mostly stopwords
                    stopword_count = sum(1 for word in ngram if word in self.stop_words)
                    if stopword_count / len(ngram) > 0.7:  # Skip if >70% stopwords
                        continue
                    
                    # If this n-gram exists in document 2
                    if ngram in ngram_hashes:
                        phrase_text = ' '.join(ngram)
                        
                        # Check if this is a generic phrase using a more sophisticated approach
                        word_count = len(ngram)
                        unique_words = len(set(ngram) - self.stop_words)
                        
                        # Skip overly generic phrases
                        if unique_words < max(2, word_count * 0.3):  # At least 30% unique non-stopwords
                            continue
                        
                        # Get position in doc1
                        pos1 = (sentence_idx1, word_idx1 + i)
                        
                        # Add all matches with their positions
                        for pos2 in ngram_hashes[ngram]:
                            found_phrases.append({
                                'text': phrase_text,
                                'length': n,
                                'pos_doc1': pos1,
                                'pos_doc2': (pos2[0], pos2[1]),
                                'sentence_doc1': sentences1[sentence_idx1],
                                'sentence_doc2': sentences2[pos2[0]]
                            })
            
            sentence_idx1 += 1
            word_idx1 += len(sentence)
        
        # Remove overlapping and redundant matches
        filtered_phrases = self._filter_overlapping_phrases(found_phrases)
        
        # Get top matches based on length and relevance
        scored_phrases = self._score_phrases(filtered_phrases)
        
        return scored_phrases[:20]  # Return top 20 phrases
        
    def _filter_overlapping_phrases(self, phrases):
        """Filter out redundant or overlapping phrase matches."""
        if not phrases:
            return []
            
        # Sort by length (descending) and then by position in doc1
        sorted_phrases = sorted(phrases, key=lambda x: (-x['length'], x['pos_doc1']))
        
        filtered = []
        covered_positions_doc1 = set()
        covered_positions_doc2 = set()
        
        for phrase in sorted_phrases:
            # Get range of positions this phrase covers
            pos1_start = phrase['pos_doc1']
            pos1_end = (phrase['pos_doc1'][0], phrase['pos_doc1'][1] + phrase['length'] - 1)
            
            pos2_start = phrase['pos_doc2']
            pos2_end = (phrase['pos_doc2'][0], phrase['pos_doc2'][1] + phrase['length'] - 1)
            
            # Check if this phrase significantly overlaps with already covered positions
            overlapping = False
            
            # Generate all position tuples for this phrase
            current_positions1 = set()
            for i in range(phrase['length']):
                current_positions1.add((pos1_start[0], pos1_start[1] + i))
                
            current_positions2 = set()
            for i in range(phrase['length']):
                current_positions2.add((pos2_start[0], pos2_start[1] + i))
            
            # Calculate overlap percentage
            overlap1 = len(current_positions1.intersection(covered_positions_doc1)) / len(current_positions1) if current_positions1 else 0
            overlap2 = len(current_positions2.intersection(covered_positions_doc2)) / len(current_positions2) if current_positions2 else 0
            
            # If overlap is too high, skip this phrase
            if overlap1 > 0.5 or overlap2 > 0.5:
                continue
                
            # Add this phrase and mark its positions as covered
            filtered.append(phrase)
            covered_positions_doc1.update(current_positions1)
            covered_positions_doc2.update(current_positions2)
            
        return filtered
        
    def _score_phrases(self, phrases):
        """Score phrases based on length, uniqueness and informativeness."""
        if not phrases:
            return []
            
        scored_phrases = []
        
        for phrase in phrases:
            # Split into words for analysis
            words = phrase['text'].split()
            
            # Base score is the phrase length (longer phrases are better)
            length_score = min(1.0, phrase['length'] / 10.0)  # Cap at 1.0
            
            # Calculate word rarity by checking what percentage are not stopwords
            non_stopwords = [w for w in words if w not in self.stop_words]
            rarity_score = len(non_stopwords) / len(words) if words else 0
            
            # Calculate uniqueness of words (using a simple metric of unique words / total words)
            unique_words = len(set(words))
            uniqueness_score = unique_words / len(words) if words else 0
            
            # Combined score (weighted factors)
            total_score = (length_score * 0.5) + (rarity_score * 0.3) + (uniqueness_score * 0.2)
            
            # Add context by including surrounding text
            scored_phrases.append({
                'phrase': phrase['text'],
                'length': phrase['length'],
                'score': total_score,
                'context_doc1': phrase['sentence_doc1'],
                'context_doc2': phrase['sentence_doc2']
            })
            
        # Sort by score (descending)
        return sorted(scored_phrases, key=lambda x: -x['score'])
    
    def check_similarity_with_online_sources(self, document, file_name=None, check_google=True, check_scholarly=True):
        """
        Check a document against online sources to find potential matches.
        Returns a list of potential online matches.
        """
        if not self.online_checker:
            raise ValueError("Online repository checker not initialized. Please provide API keys if required.")
        
        print(f"\nChecking document against online sources: {file_name if file_name else 'Unnamed document'}")
        print("This may take a few moments...")
        
        # Search online repositories
        results = self.online_checker.check_document_against_online_sources(
            document, 
            check_google=check_google, 
            check_scholarly=check_scholarly,
            language=self.language
        )
        
        return {
            'google_matches': results['google_matches'],
            'scholarly_matches': results['scholarly_matches'],
            'total_matches': len(results['google_matches']) + len(results['scholarly_matches'])
        }
    
    def visualize_similarity(self, similarity_matrix, file_names=None, output_path=None):
        """Generate a heatmap visualization of similarity matrix."""
        plt.figure(figsize=(10, 8))
        
        # Create labels
        labels = file_names if file_names else [f"Doc {i+1}" for i in range(len(similarity_matrix))]
        
        # Create heatmap
        sns.heatmap(
            similarity_matrix,
            annot=True,
            cmap='YlOrRd',
            xticklabels=labels,
            yticklabels=labels,
            vmin=0,
            vmax=1,
            fmt='.2f'
        )
        plt.title('Document Similarity Heatmap')
        plt.tight_layout()
        
        # Save the plot if path provided
        if output_path:
            plt.savefig(output_path)
            print(f"Heatmap saved as '{output_path}'")
        
        # Display the plot if no path provided
        else:
            plt.show()
        
        plt.close()  # Close the plot to free memory
        
        return output_path


def load_documents(directory=None, files=None):
    """Load documents from either a directory or specific files."""
    documents = []
    file_names = []
    
    # Supported file extensions
    supported_extensions = ['.txt', '.pdf', '.docx', '.doc', '.rtf', '.odt']
    
    if directory and os.path.isdir(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            _, ext = os.path.splitext(filename.lower())
            
            if ext in supported_extensions and os.path.isfile(file_path):
                text = extract_text_from_file(file_path)
                if text:  # Only add if text extraction was successful
                    documents.append(text)
                    file_names.append(filename)
    
    elif files:
        for file_path in files:
            if os.path.isfile(file_path):
                text = extract_text_from_file(file_path)
                if text:  # Only add if text extraction was successful
                    documents.append(text)
                    file_names.append(os.path.basename(file_path))
    
    return documents, file_names


def main():
    """Command-line interface function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check documents for plagiarism and similarity.')
    parser.add_argument('--directory', '-d', help='Directory containing documents to compare')
    parser.add_argument('--files', '-f', nargs='+', help='Specific files to compare')
    parser.add_argument('--threshold', '-t', type=float, default=30.0, 
                        help='Similarity threshold percentage to flag (default: 30.0)')
    parser.add_argument('--visualize', '-v', action='store_true', 
                        help='Generate similarity heatmap visualization')
    parser.add_argument('--phrases', '-p', action='store_true', 
                        help='Find common phrases between documents')
    parser.add_argument('--formats', action='store_true',
                        help='List all supported file formats')
    parser.add_argument('--online', '-o', action='store_true',
                        help='Check documents against online sources')
    parser.add_argument('--api-keys', type=str,
                        help='Path to JSON file with API keys for online services')
    parser.add_argument('--scholarly-only', action='store_true',
                        help='Only check scholarly repositories, not general web search')
    parser.add_argument('--no-google', action='store_true',
                        help='Disable Google search for online checking')
    parser.add_argument('--language', '-l', type=str, default='english',
                        choices=['english', 'french', 'spanish'],
                        help='Language of the documents (default: english)')
    
    args = parser.parse_args()
    
    # Display supported formats if requested
    if args.formats:
        print("\nSupported File Formats:")
        print("  - .txt  (Plain text files)")
        print("  - .pdf  (PDF documents)")
        print("  - .docx (Microsoft Word documents)")
        print("  - .doc  (Microsoft Word legacy format)")
        print("  - .rtf  (Rich Text Format)")
        print("  - .odt  (OpenDocument Text)")
        return
    
    if not args.directory and not args.files:
        parser.error("Either --directory or --files must be provided.")
    
    # Load documents
    documents, file_names = load_documents(args.directory, args.files)
    
    if len(documents) < 1:
        print("At least one document is required for analysis.")
        return
    
    # Initialize plagiarism checker with online repository checker if needed
    online_checker = None
    if args.online:
        # Load API keys if provided
        api_keys = {}
        if args.api_keys and os.path.isfile(args.api_keys):
            try:
                with open(args.api_keys, 'r') as f:
                    api_keys = json.load(f)
                print("Loaded API keys for online repository checking.")
            except Exception as e:
                print(f"Error loading API keys: {str(e)}")
                print("Proceeding with limited online checking capabilities.")
        
        online_checker = OnlineRepositoryChecker(api_keys)
    
    checker = PlagiarismChecker(online_checker, language=args.language)
    
    # Perform document-to-document comparison if multiple documents
    if len(documents) >= 2:
        similarity_matrix, results = checker.check_similarity_tfidf(documents, file_names)
        
        # Print results
        print("\n" + "="*80)
        print(f"DOCUMENT SIMILARITY ANALYSIS (Language: {args.language})")
        print("="*80)
        
        for result in results:
            similarity = result['Similarity (%)']
            print(f"{result['Document 1']} <-> {result['Document 2']}: {similarity:.2f}% similar")
            
            # Flag potentially plagiarized content
            if similarity >= args.threshold:
                print(f"⚠️  POTENTIAL PLAGIARISM DETECTED (above {args.threshold}% threshold)")
                
                # Display common phrases if requested and available
                if args.phrases and 'Common Phrases' in result:
                    phrases = result['Common Phrases']
                    
                    if phrases:
                        print("\nTop matching phrases:")
                        for i, phrase_data in enumerate(phrases, 1):
                            phrase = phrase_data['phrase']
                            context1 = phrase_data['context_doc1'].strip()
                            context2 = phrase_data['context_doc2'].strip()
                            
                            # Highlight the phrase in context
                            highlighted_context1 = context1.replace(phrase, f"**{phrase}**")
                            highlighted_context2 = context2.replace(phrase, f"**{phrase}**")
                            
                            print(f"\n{i}. \"{phrase}\" (length: {phrase_data['length']} words, score: {phrase_data['score']:.2f})")
                            print(f"   Document 1: \"{highlighted_context1}\"")
                            print(f"   Document 2: \"{highlighted_context2}\"")
                    else:
                        print("\nNo significant matching phrases found despite similarity.")
                        print("This may indicate structural similarity rather than verbatim copying.")
                
                # Find common phrases if not already calculated but phrases requested
                elif args.phrases and 'Common Phrases' not in result:
                    doc1_idx = file_names.index(result['Document 1'])
                    doc2_idx = file_names.index(result['Document 2'])
                    
                    print("\nAnalyzing for common phrases...")
                    common_phrases = checker.find_common_phrases(documents[doc1_idx], documents[doc2_idx])
                    
                    if common_phrases:
                        print("\nTop matching phrases:")
                        for i, phrase_data in enumerate(common_phrases[:5], 1):
                            phrase = phrase_data['phrase']
                            context1 = phrase_data['context_doc1'].strip()
                            context2 = phrase_data['context_doc2'].strip()
                            
                            # Highlight the phrase in context
                            highlighted_context1 = context1.replace(phrase, f"**{phrase}**")
                            highlighted_context2 = context2.replace(phrase, f"**{phrase}**")
                            
                            print(f"\n{i}. \"{phrase}\" (length: {phrase_data['length']} words, score: {phrase_data['score']:.2f})")
                            print(f"   Document 1: \"{highlighted_context1}\"")
                            print(f"   Document 2: \"{highlighted_context2}\"")
                    else:
                        print("\nNo significant matching phrases found despite similarity.")
                        print("This may indicate structural similarity rather than verbatim copying.")
                    
            print("-"*80)
        
        # Generate visualization if requested
        if args.visualize:
            checker.visualize_similarity(similarity_matrix, file_names)
    
    # Perform online source checking if requested
    if args.online and online_checker:
        for idx, document in enumerate(documents):
            check_google = not args.no_google and not args.scholarly_only
            check_scholarly = not args.scholarly_only
            
            try:
                print("\n" + "="*80)
                print(f"CHECKING DOCUMENT AGAINST ONLINE SOURCES: {file_names[idx]}")
                print("="*80)
                
                online_results = checker.check_similarity_with_online_sources(document, file_names[idx],
                                                                         check_google=check_google,
                                                                         check_scholarly=check_scholarly)
                
                # Print Google matches
                if check_google and online_results['google_matches']:
                    print("\nPotential matches found on the web:")
                    for i, match in enumerate(online_results['google_matches'], 1):
                        print(f"\n{i}. {match['title']}")
                        print(f"   URL: {match['url']}")
                        print(f"   Matched phrase: \"{match['matched_phrase']}\"")
                        print(f"   Context: {match['snippet']}")
                        print("-"*60)
                elif check_google:
                    print("\nNo significant matches found on the web.")
                
                # Print scholarly matches
                if check_scholarly and online_results['scholarly_matches']:
                    print("\nPotential matches found in scholarly repositories:")
                    for i, match in enumerate(online_results['scholarly_matches'], 1):
                        print(f"\n{i}. {match['title']}")
                        print(f"   URL: {match['url']}")
                        print(f"   Repository: {match.get('repository', 'Unknown')}")
                        print(f"   Matched phrase: \"{match['matched_phrase']}\"")
                        print(f"   Context: {match['snippet']}")
                        print("-"*60)
                elif check_scholarly:
                    print("\nNo significant matches found in scholarly repositories.")
                
                if online_results['total_matches'] == 0:
                    print("\nNo significant online matches found for this document.")
                
            except Exception as e:
                print(f"Error checking online sources: {str(e)}")
                print("Skipping online check for this document.")
    
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()