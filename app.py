from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import difflib
from difflib import SequenceMatcher
import json

app = Flask(__name__)

# Sample Social Security FAQ data (you can expand this significantly)
# In a real application, this would come from a database.
faqs = [
    {
        "question": "What is Social Security?",
        "answer": "Social Security is a social insurance program run by the U.S. government. It provides benefits to retirees, the disabled, and survivors of deceased workers."
    },
    {
        "question": "How do I apply for Social Security benefits?",
        "answer": "You can apply for Social Security benefits online, by phone, or in person at your local Social Security office. It's recommended to apply a few months before you want your benefits to start."
    },
    {
        "question": "When can I retire and get full benefits?",
        "answer": "Your full retirement age depends on your birth year. For most people currently, it's between 66 and 67 years old. You can start receiving benefits as early as age 62, but they will be reduced."
    },
    {
        "question": "How can I check my Social Security statement?",
        "answer": "You can check your Social Security statement online by creating a my Social Security account on the official SSA website. This statement shows your earnings history and estimated future benefits."
    },
    {
        "question": "What happens if I work while receiving Social Security benefits?",
        "answer": "If you are under full retirement age and work while receiving benefits, your benefits may be reduced if your earnings exceed certain limits. Once you reach full retirement age, your benefits are not reduced regardless of how much you earn."
    },
    {
        "question": "Can I get Social Security benefits if I'm disabled?",
        "answer": "Yes, Social Security Disability Insurance (SSDI) provides benefits to those who have worked long enough and paid Social Security taxes, and who have a medical condition that meets Social Security's definition of disability."
    },
    {
        "question": "What is Medicare?",
        "answer": "Medicare is the federal health insurance program for people who are 65 or older, certain younger people with disabilities, and people with End-Stage Renal Disease (permanent kidney failure requiring dialysis or a transplant)."
    },
    {
        "question": "How are Social Security benefits calculated?",
        "answer": "Your Social Security benefit amount is based on your average indexed monthly earnings (AIME) during your 35 highest earning years. The Social Security Administration uses a formula to calculate your Primary Insurance Amount (PIA)."
    },
    {
        "question": "What is the difference between Social Security and SSI?",
        "answer": "Social Security (SSDI) is an insurance program for those who have paid into it through taxes. Supplemental Security Income (SSI) is a needs-based program for low-income individuals who are aged, blind, or disabled, regardless of their work history."
    },
    {
        "question": "How do I report a change of address to Social Security?",
        "answer": "You can report a change of address online through your my Social Security account, by calling Social Security directly, or by visiting a local office."
    }
]

# Extract only the questions for vectorization
faq_questions = [faq["question"] for faq in faqs]

# Initialize TF-IDF Vectorizer for string-based cosine similarity
vectorizer = TfidfVectorizer().fit(faq_questions)
faq_question_vectors = vectorizer.transform(faq_questions)

# Initialize Sentence Transformer for semantic embeddings
# This model is good for semantic similarity tasks
print("Loading sentence transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings for all FAQ questions at startup
print("Computing embeddings for FAQ questions...")
faq_embeddings = embedding_model.encode(faq_questions)
print(f"Embeddings computed for {len(faq_questions)} questions.")

# Load test questions for UI testing
try:
    with open('test_questions.json', 'r') as f:
        test_data = json.load(f)
        test_questions = test_data['test_questions']
        print(f"Loaded {len(test_questions)} test questions.")
except FileNotFoundError:
    test_questions = []
    print("Test questions file not found.")

def string_cosine_similarity(query, faq_questions):
    """Calculate string-based cosine similarity using TF-IDF vectors"""
    user_question_vector = vectorizer.transform([query])
    similarities = cosine_similarity(user_question_vector, faq_question_vectors).flatten()
    best_match_index = np.argmax(similarities)
    return best_match_index, similarities[best_match_index]

def embedding_cosine_similarity(query, faq_embeddings):
    """Calculate semantic similarity using sentence embeddings"""
    # Encode the user query
    query_embedding = embedding_model.encode([query])
    
    # Calculate cosine similarity with all FAQ embeddings
    similarities = cosine_similarity(query_embedding, faq_embeddings).flatten()
    best_match_index = np.argmax(similarities)
    return best_match_index, similarities[best_match_index]

def simple_string_similarity(query, faq_questions):
    """Calculate simple string similarity using SequenceMatcher"""
    similarities = []
    for faq_q in faq_questions:
        similarity = SequenceMatcher(None, query.lower(), faq_q.lower()).ratio()
        similarities.append(similarity)
    
    best_match_index = np.argmax(similarities)
    return best_match_index, similarities[best_match_index]

@app.route('/', methods=['GET', 'POST'])
def index():
    user_question = ""
    results = {
        'string_cosine': {'question': '', 'answer': '', 'score': 0.0},
        'embedding_cosine': {'question': '', 'answer': '', 'score': 0.0},
        'simple_string': {'question': '', 'answer': '', 'score': 0.0}
    }

    if request.method == 'POST':
        user_question = request.form['user_question']
        if user_question:
            # Method 1: String-based cosine similarity (TF-IDF)
            string_idx, string_score = string_cosine_similarity(user_question, faq_questions)
            if string_score > 0.1:  # Lower threshold for TF-IDF
                results['string_cosine'] = {
                    'question': faqs[string_idx]["question"],
                    'answer': faqs[string_idx]["answer"],
                    'score': string_score
                }
            else:
                results['string_cosine'] = {
                    'question': 'No good match found',
                    'answer': 'Sorry, I couldn\'t find a close match using string-based similarity.',
                    'score': string_score
                }

            # Method 2: Embedding-based cosine similarity
            embed_idx, embed_score = embedding_cosine_similarity(user_question, faq_embeddings)
            if embed_score > 0.3:  # Higher threshold for embeddings (they're usually more accurate)
                results['embedding_cosine'] = {
                    'question': faqs[embed_idx]["question"],
                    'answer': faqs[embed_idx]["answer"],
                    'score': embed_score
                }
            else:
                results['embedding_cosine'] = {
                    'question': 'No good match found',
                    'answer': 'Sorry, I couldn\'t find a close match using semantic similarity.',
                    'score': embed_score
                }

            # Method 3: Simple string similarity
            simple_idx, simple_score = simple_string_similarity(user_question, faq_questions)
            if simple_score > 0.2:  # Moderate threshold for simple string matching
                results['simple_string'] = {
                    'question': faqs[simple_idx]["question"],
                    'answer': faqs[simple_idx]["answer"],
                    'score': simple_score
                }
            else:
                results['simple_string'] = {
                    'question': 'No good match found',
                    'answer': 'Sorry, I couldn\'t find a close match using simple string similarity.',
                    'score': simple_score
                }
        else:
            for method in results:
                results[method] = {
                    'question': '',
                    'answer': 'Please enter a question.',
                    'score': 0.0
                }

    return render_template('index.html',
                           user_question=user_question,
                           results=results)

@app.route('/get_test_question/<int:index>')
def get_test_question(index):
    """API endpoint to get a test question by index"""
    if 0 <= index < len(test_questions):
        return jsonify({
            'question': test_questions[index]['question'],
            'description': test_questions[index]['description'],
            'index': index,
            'total': len(test_questions)
        })
    else:
        return jsonify({
            'question': '',
            'description': 'No more test questions',
            'index': 0,
            'total': len(test_questions)
        })

if __name__ == '__main__':
    import os
    
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Get host from environment variable or default to localhost
    host = os.environ.get('HOST', '0.0.0.0')
    
    # Check if we're in production
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(host=host, port=port, debug=debug_mode)
