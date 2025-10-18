
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv


from PIL import Image
import pytesseract
from pdfminer.high_level import extract_text
import docx
from moviepy.editor import VideoFileClip
import whisper

# Gemini LLM SDK
from google import genai

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Use absolute path for uploads folder to avoid path issues
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# ------------------ TEXT EXTRACTION HELPERS ------------------

def extract_text_from_image(image_path):
    try:
        return pytesseract.image_to_string(Image.open(image_path))
    except Exception as e:
        return f"Error extracting text from image: {e}"


def extract_text_from_pdf(pdf_path):
    try:
        return extract_text(pdf_path)
    except Exception as e:
        return f"Error extracting text from pdf: {e}"


def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error extracting text from docx: {e}"


import speech_recognition as sr

def extract_text_from_audio(audio_path):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = r.record(source)
        # Using Google Web Speech API (requires internet access)
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Speech Recognition could not understand the audio."
    except sr.RequestError as e:
        return f"Could not request results from Speech Recognition service; {e}"
    except Exception as e:
        return f"Error extracting text from audio: {e}"



def extract_text_from_video(video_path):
    try:
        clip = VideoFileClip(video_path)
        audio_path = video_path.rsplit('.', 1)[0] + ".wav"
        clip.audio.write_audiofile(audio_path)
        return extract_text_from_audio(audio_path)
    except Exception as e:
        return f"Error extracting text from video: {e}"


def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.txt']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext in ['.jpg', '.jpeg', '.png']:
        return extract_text_from_image(file_path)
    elif ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext in ['.mp3', '.wav']:
        return extract_text_from_audio(file_path)
    elif ext in ['.mp4', '.mov']:
        return extract_text_from_video(file_path)
    else:
        return "Unsupported file type."


# ------------------ GEMINI LLM FUNCTION ------------------

def query_gemini_llm(prompt):
    print(prompt)
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return "Error: Gemini API key is missing. Set GEMINI_API_KEY in your environment."
    
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


# ------------------ FLASK ROUTES ------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/ask', methods=['POST'])
def api_ask():
    user_query = request.form.get('query', '')
    uploaded_files = request.files.getlist('file')

    extracted_texts = []
    for f in uploaded_files:
        filename = secure_filename(f.filename)
        print(f"Received file: original name '{f.filename}', secure name '{filename}'")  # Debug print
        
        if not filename:
            # Skip files without a valid filename
            print("Skipped file with empty filename")
            continue
        
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
        print(f"Saving file to: {path}")  # Debug print
        
        f.save(path)
        extracted_texts.append(extract_text_from_file(path))

    combined_input = "\n\n".join(extracted_texts) + f"\n\nQuestion: {user_query}"
    answer = query_gemini_llm(combined_input)

    return jsonify({"answer": answer})



if __name__ == '__main__':
    app.run(debug=True)
