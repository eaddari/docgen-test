import os
import markdown
from dotenv import load_dotenv
import re
import unicodedata
from nltk.corpus import stopwords
stopwords_list = set(stopwords.words("english"))
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from openai import AzureOpenAI
from data_integration import upload_folder
 
load_dotenv()

client = AzureOpenAI(
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("SUBSCRIPTION_KEY")
)

def explain_python_code_with_azure(code):
    prompt = (
        "Explain the following Python code in detail, line by line. "
        "Do not rewrite the code. Instead, provide a natural language explanation "
        "that clearly describes the purpose and logic of each part.\n\n"
        f"{code}"
    )
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=800,
        temperature=0.1,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=os.getenv("DEPLOYMENT")
    )
    return response.choices[0].message.content

def process_python_files(input_folder, output_folder):
    processed_data = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
 
                try:
                    explanation = explain_python_code_with_azure(code)
                except Exception as e:
                    print(f"Errore nella spiegazione di {file}: {e}")
                    continue
 
                rel_dir = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, rel_dir)
                os.makedirs(output_dir, exist_ok=True)
                output_file_path = os.path.join(output_dir, file.replace('.py', '_explanation.txt'))
 
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(explanation)
 
                processed_data.append({'filename': file, 'plain_text': explanation})
    return processed_data


def convert_markdown_to_html(text):
    html = markdown.markdown(text)
    return html

def html_to_structured(html):
    soup = BeautifulSoup(html, "html.parser")
    lines = []
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
        if tag.name.startswith('h'):
            lines.append(f"[{tag.name.upper()}] {tag.get_text(strip=True)}")
        elif tag.name == 'li':
            lines.append(f"- {tag.get_text(strip=True)}")
        else:
            lines.append(tag.get_text(strip=True))
    return '\n'.join(lines)

# Regex 'modulare'
def remove_non_printable_characters(text):
    return re.sub(r'[^\x20-\x7E]', '', text)
def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()
def remove_urls(text):
    return re.sub(r'http[s]?://\S+', '', text)
def remove_emails(text):
    return re.sub(r'\S+@\S+\.\S+', '', text)
def remove_markdown_comments(text):
    return re.sub(r'\s*>\s*.*\n?', '', text)

def regex_text(text):
    text = remove_non_printable_characters(text)
    text = remove_extra_whitespace(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_markdown_comments(text)
    return text

def normalize_whitespace(text):
    return '\n'.join(line.strip() for line in text.splitlines())
def remove_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c))
def normalize_newlines(text):
    return text.replace('\r\n', '\n').replace('\r', '\n')
def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s.,;:!?\'\"()-]', '', text)

def remove_stopwords(text, stopwords_list):
    words = text.split()
    return ' '.join([w for w in words if w not in stopwords_list])

def standardize_text(text):
    text = text.lower()
    text = normalize_newlines(text)
    text = normalize_whitespace(text)
    text = remove_accents(text)
    text = remove_special_characters(text)
    return text

def preprocess_markdown_folder(folder_path, output_folder):
    processed_data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw = f.read()
                    text = regex_text(raw)
                    #text = standardize_text(text)
                    text = remove_stopwords(text, stopwords.words("english"))
                html = convert_markdown_to_html(text)
                plain_text = html_to_structured(html)
                processed_data.append({'filename': file, 'plain_text': plain_text})
                rel_dir = os.path.relpath(root, folder_path)
                output_dir = os.path.join(output_folder, rel_dir)
                os.makedirs(output_dir, exist_ok=True)
                output_file_path = os.path.join(output_dir, file.replace('.md', '.txt'))
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(plain_text)
    return processed_data

def text_chunking(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

def save_chunks(filename, chunks, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    chunk_file = os.path.join(output_folder, filename.replace('.md', '_chunks.txt').replace('.txt', '_chunks.txt'))
    with open(chunk_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            clean_chunk = chunk.replace('\n', ' ').strip()
            f.write(clean_chunk + '\n')

def chunk_folder(input_folder, chunks_output, chunk_size=1000, chunk_overlap=200):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    plain_text = f.read()
                chunks = text_chunking(plain_text, chunk_size, chunk_overlap)
                rel_dir = os.path.relpath(root, input_folder)
                output_dir = os.path.join(chunks_output, rel_dir)
                os.makedirs(output_dir, exist_ok=True)
                save_chunks(file, chunks, output_dir)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings(text):
    return embedding_model.embed_documents([text])[0]

def save_embeddings(filename, embeddings, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    embedding_file = os.path.join(output_folder, filename.replace('.md', '_embeddings.json').replace('.txt', '_embeddings.json'))
    with open(embedding_file, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=4)

def embed_chunks(chunks_folder, embeddings_output):
    for root, _, files in os.walk(chunks_folder):
        for file in files:
            if file.endswith('_chunks.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = [line.strip() for line in f]
                embeddings = [get_embeddings(chunk) for chunk in chunks]
                rel_dir = os.path.relpath(root, chunks_folder)
                output_dir = os.path.join(embeddings_output, rel_dir)
                os.makedirs(output_dir, exist_ok=True)
                save_embeddings(file.replace('_chunks.txt', '_embeddings.json'), embeddings, output_dir)
                
if __name__ == "__main__":

    preprocess_output = "preprocess_output"
    os.makedirs(preprocess_output, exist_ok=True)

    preprocess_markdown_folder("downloaded_data", preprocess_output)
    process_python_files("downloaded_data", preprocess_output)
    upload_folder("preprocess_output", "preprocess_output")

    chunks_output = "chunks_output"
    os.makedirs(chunks_output, exist_ok=True)

    chunk_folder(preprocess_output, chunks_output)
    upload_folder("chunks_output", "chunks_output")

    embeddings_output = "embeddings_output"
    os.makedirs(embeddings_output, exist_ok=True)  
    
    embed_chunks(chunks_output, embeddings_output)
    upload_folder("embeddings_output", "embeddings_output")

