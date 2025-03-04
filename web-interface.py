#!/usr/bin/env python3
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
import uuid
import json
import time
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

# Import plagiarism checker modules
from plagiarism_core import PlagiarismChecker, OnlineRepositoryChecker, extract_text_from_file

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-for-plagiarism-checker')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx', 'doc', 'rtf', 'odt'}

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['RESULTS_FOLDER'], 'plots'), exist_ok=True)

# Initialize plagiarism checker with API keys if available
api_keys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_keys.json')
api_keys = {}
if os.path.exists(api_keys_path):
    try:
        with open(api_keys_path, 'r') as f:
            api_keys = json.load(f)
    except Exception as e:
        print(f"Error loading API keys: {str(e)}")

# Create online repository checker only when needed to avoid unnecessary resources
online_checker = None

def get_online_checker():
    global online_checker
    if online_checker is None:
        online_checker = OnlineRepositoryChecker(api_keys)
    return online_checker

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_session_id():
    return str(uuid.uuid4())

def get_language_nltk_name(language_code):
    """Convert language code to NLTK language name"""
    language_map = {
        'en': 'english',
        'fr': 'french',
        'es': 'spanish'
    }
    return language_map.get(language_code, 'english')

@app.route('/')
def index():
    # Create a new session if none exists
    if 'session_id' not in session:
        session['session_id'] = generate_session_id()
    
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'session_id' not in session:
        session['session_id'] = generate_session_id()
    
    session_id = session['session_id']
    session_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_upload_folder, exist_ok=True)
    
    # Get language selection
    language = request.form.get('language', 'en')
    
    # Get analysis options
    check_online = 'check_online' in request.form
    check_scholarly = 'check_scholarly' in request.form
    check_phrases = 'check_phrases' in request.form
    visualize = 'visualize' in request.form
    similarity_threshold = float(request.form.get('threshold', 30.0))
    
    # Save options to session
    session['options'] = {
        'language': language,
        'check_online': check_online,
        'check_scholarly': check_scholarly,
        'check_phrases': check_phrases,
        'visualize': visualize,
        'threshold': similarity_threshold
    }
    
    # Check if files were uploaded
    if 'files' not in request.files:
        flash('No files selected', 'error')
        return redirect(url_for('index'))
    
    files = request.files.getlist('files')
    
    # Check if at least one valid file was selected
    if len(files) == 0 or files[0].filename == '':
        flash('No files selected', 'error')
        return redirect(url_for('index'))
    
    # Save the uploaded files
    filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(session_upload_folder, filename)
            file.save(file_path)
            filenames.append(filename)
    
    if not filenames:
        flash('No valid files uploaded. Supported formats: txt, pdf, docx, doc, rtf, odt', 'error')
        return redirect(url_for('index'))
    
    # Store filenames in session
    session['filenames'] = filenames
    
    # Start analysis
    return redirect(url_for('analyze'))

@app.route('/analyze')
def analyze():
    if 'session_id' not in session or 'filenames' not in session or 'options' not in session:
        flash('Session expired or no files uploaded', 'error')
        return redirect(url_for('index'))
    
    session_id = session['session_id']
    filenames = session['filenames']
    options = session['options']
    
    session_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    session_results_folder = os.path.join(app.config['RESULTS_FOLDER'], session_id)
    os.makedirs(session_results_folder, exist_ok=True)
    
    # Get full file paths
    file_paths = [os.path.join(session_upload_folder, filename) for filename in filenames]
    
    # Extract text from files
    documents = []
    for file_path in file_paths:
        try:
            text = extract_text_from_file(file_path)
            if text:
                documents.append(text)
            else:
                flash(f'Could not extract text from {os.path.basename(file_path)}', 'warning')
        except Exception as e:
            flash(f'Error processing {os.path.basename(file_path)}: {str(e)}', 'error')
    
    if not documents:
        flash('No valid text content could be extracted from the uploaded files', 'error')
        return redirect(url_for('index'))
    
    # Initialize plagiarism checker with the selected language
    nltk_language = get_language_nltk_name(options['language'])
    plagiarism_checker = PlagiarismChecker(
        online_checker=get_online_checker() if options['check_online'] else None,
        language=nltk_language
    )
    
    results = {}
    
    # Document-to-document comparison
    if len(documents) >= 2:
        similarity_matrix, comparisons = plagiarism_checker.check_similarity_tfidf(documents, filenames)
        
        # If visualization is requested, generate heatmap
        visualization_path = None
        if options['visualize']:
            plot_filename = f"similarity_heatmap_{int(time.time())}.png"
            plot_path = os.path.join(session_results_folder, plot_filename)
            plagiarism_checker.visualize_similarity(similarity_matrix, filenames, output_path=plot_path)
            visualization_path = plot_filename
        
        # If phrase detection is requested, find common phrases for each high-similarity pair
        if options['check_phrases']:
            for i, comparison in enumerate(comparisons):
                if comparison['Similarity (%)'] >= options['threshold']:
                    doc1_idx = filenames.index(comparison['Document 1'])
                    doc2_idx = filenames.index(comparison['Document 2'])
                    
                    if 'Common Phrases' not in comparison:
                        phrases = plagiarism_checker.find_common_phrases(documents[doc1_idx], documents[doc2_idx])
                        comparisons[i]['Common Phrases'] = phrases[:5]  # Top 5 phrases
        
        results['document_comparisons'] = comparisons
        results['visualization_path'] = visualization_path
    
    # Online source checking
    online_results = {}
    if options['check_online']:
        for i, document in enumerate(documents):
            try:
                check_google = True
                check_scholarly = options['check_scholarly']
                
                online_matches = plagiarism_checker.check_similarity_with_online_sources(
                    document, 
                    filenames[i],
                    check_google=check_google,
                    check_scholarly=check_scholarly
                )
                
                online_results[filenames[i]] = online_matches
            except Exception as e:
                flash(f'Error checking online sources for {filenames[i]}: {str(e)}', 'error')
    
    results['online_results'] = online_results
    
    # Save results to file
    results_file = os.path.join(session_results_folder, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f)
    
    # Store results path in session
    session['results_path'] = results_file
    
    return redirect(url_for('results'))

@app.route('/results')
def results():
    if 'session_id' not in session or 'results_path' not in session:
        flash('No analysis results found', 'error')
        return redirect(url_for('index'))
    
    results_path = session['results_path']
    
    if not os.path.exists(results_path):
        flash('Results file not found', 'error')
        return redirect(url_for('index'))
    
    # Load results from file
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Get session options
    options = session.get('options', {})
    threshold = options.get('threshold', 30.0)
    
    # Prepare visualization path if exists
    visualization_path = None
    if 'visualization_path' in results and results['visualization_path']:
        session_id = session['session_id']
        visualization_path = os.path.join(session_id, results['visualization_path'])
    
    return render_template(
        'results.html',
        comparisons=results.get('document_comparisons', []),
        online_results=results.get('online_results', {}),
        visualization_path=visualization_path,
        threshold=threshold
    )

@app.route('/download_report')
def download_report():
    # Generate a downloadable report (implementation left as future enhancement)
    flash('Report download functionality will be available in a future update', 'info')
    return redirect(url_for('results'))

@app.route('/api/check', methods=['POST'])
def api_check():
    """API endpoint for programmatic plagiarism checking"""
    # Get request data
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    # Get text and options
    text = data.get('text', '')
    language = data.get('language', 'en')
    check_online = data.get('check_online', False)
    check_scholarly = data.get('check_scholarly', False)
    
    # Validate text
    if not text or len(text.strip()) < 50:
        return jsonify({'error': 'Text too short for meaningful analysis'}), 400
    
    # Initialize checker
    nltk_language = get_language_nltk_name(language)
    plagiarism_checker = PlagiarismChecker(
        online_checker=get_online_checker() if check_online else None,
        language=nltk_language
    )
    
    results = {}
    
    # Check against online sources if requested
    if check_online:
        try:
            online_results = plagiarism_checker.check_similarity_with_online_sources(
                text,
                'api_submission',
                check_google=True,
                check_scholarly=check_scholarly
            )
            results['online_results'] = online_results
        except Exception as e:
            results['online_error'] = str(e)
    
    # If comparison texts are provided, check against those
    if 'comparison_texts' in data and isinstance(data['comparison_texts'], list):
        comparison_texts = data['comparison_texts']
        
        if comparison_texts and len(comparison_texts) > 0:
            documents = [text] + comparison_texts
            labels = ['Submitted Text'] + [f'Comparison Text {i+1}' for i in range(len(comparison_texts))]
            
            try:
                _, comparisons = plagiarism_checker.check_similarity_tfidf(documents, labels)
                
                # Filter to only show comparisons with the submitted text
                relevant_comparisons = [c for c in comparisons if 'Submitted Text' in 
                                       [c['Document 1'], c['Document 2']]]
                
                results['document_comparisons'] = relevant_comparisons
            except Exception as e:
                results['comparison_error'] = str(e)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
