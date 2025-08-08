from flask import Flask, request, render_template, redirect, url_for, send_file, jsonify
import uuid
import os
import traceback
from openai import OpenAI
import tempfile
import chardet
from dotenv import load_dotenv
from your_processing_module import (
    add_diacritics_to_text,
    search_for_image_vietnamplus,
    format_article
)

load_dotenv()

# Set up OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store articles and their progress
article_store = {}  # { id: {id, path, status, headline, body, ...} }

def read_file_safely(path):
    with open(path, 'rb') as f:
        raw = f.read()
        detected = chardet.detect(raw)
        encoding = detected['encoding']
        print(f"Detected encoding: {encoding}")
        return raw.decode(encoding)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('files')

        for file in uploaded_files:
            if not file or not file.filename.endswith('.txt'):
                continue

            article_id = str(uuid.uuid4())
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{article_id}_{file.filename}")
            file.save(temp_path)

            article_store[article_id] = {
                'id': article_id,
                'filename': file.filename,
                'path': temp_path,
                'status': 'Starting...'
            }

        return redirect(url_for('summary'))

    return render_template('index.html')

@app.route('/progress/<id>')
def progress(id):
    return render_template('progress.html', id=id)

@app.route('/status/<id>')
def status(id):
    article = article_store.get(id)
    if not article:
        return jsonify({"status": "Not found"}), 404
    return jsonify({"status": article.get("status", "Unknown")})

@app.route('/run/<id>')
def run(id):
    article = article_store.get(id)
    if not article:
        return "Not found", 404

    try:
        article['status'] = "Reading file"
        text = read_file_safely(article['path'])

        article['status'] = "Calling OpenAI"
        processed_text = add_diacritics_to_text(text)

        article['status'] = "Formatting"
        formatted_text = format_article(processed_text)

        headline = formatted_text.split('\n')[0]
        body = '\n'.join(formatted_text.split('\n')[1:])

        article['status'] = "Searching image"
        image_url, image_caption = search_for_image_vietnamplus(headline, formatted_text)

        article.update({
            'headline': headline,
            'body': body,
            'image_url': image_url,
            'image_caption': image_caption,
            'status': 'Complete'
        })

        return '', 204

    except Exception as e:
        article['status'] = f"Error: {str(e)}"
        traceback.print_exc()
        return '', 500

@app.route('/article/<id>')
def article(id):
    article = article_store.get(id)
    if not article:
        return "Article not found", 404
    return render_template('result.html', **article)

@app.route('/summary')
def summary():
    return render_template('summary.html', articles=article_store.values())

if __name__ == '__main__':
    app.run(debug=True)