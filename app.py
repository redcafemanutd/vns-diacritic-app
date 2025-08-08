from flask import Flask, request, render_template, redirect, url_for, send_file
import uuid
import os
import openai
import tempfile
import chardet
import re
import traceback
from your_processing_module import add_diacritics_to_text, search_for_image_vietnamplus, format_article
from dotenv import load_dotenv
load_dotenv()

# Load your OpenAI key (or use dotenv if you prefer)
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or hardcode for local testing

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

article_store = {}

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
        results = []

        for file in uploaded_files:
            if not file or not file.filename.endswith('.txt'):
                continue

            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{file.filename}")
            file.save(temp_path)

            try:
                original_text = read_file_safely(temp_path)
                processed_text = add_diacritics_to_text(original_text)
                formatted_text = format_article(processed_text)

                headline = formatted_text.split('\n')[0]
                body = '\n'.join(formatted_text.split('\n')[1:])
                image_url, image_caption = search_for_image_vietnamplus(headline, formatted_text)

                article_id = str(uuid.uuid4())

                article_data = {
                    'id': article_id,
                    'headline': headline,
                    'body': body,
                    'image_url': image_url,
                    'image_caption': image_caption,
                }

                article_store[article_id] = article_data
                results.append(article_data)

                

            except Exception as e:
                results.append({'error': f"Error with {file.filename}: {e}"})
        return redirect(url_for('summary'))

    return render_template('index.html')

@app.route('/summary')
def summary():
    print("Article store:", article_store)
    return render_template('summary.html', articles=article_store.values())

@app.route('/article/<id>')
def article(id):
    article = article_store.get(id)
    if not article:
        return "Article not found", 404
    return render_template('result.html', **article)

if __name__ == '__main__':
    app.run(debug=True)
