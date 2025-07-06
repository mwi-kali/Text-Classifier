# Text Classifier & Summarizer

---

## Overview

Text-Classifier is a modular, efficient, and high‑performance Python project for extracting, preprocessing, classifying, and summarizing text from a variety of sources (PDFs, plain text, web pages, and YouTube videos) using industry‑standard NLP pipelines. The system leverages SpaCy for linguistic preprocessing, Hugging Face transformers for sentiment, zero‑shot classification, and abstractive summarization, and LangChain’s utilities for map‑reduce style summarization workflows. 


## Project Structure

```
Text-Classifier/
├── text_classifier/
│   ├── __init__.py
│   ├── app.py               
│   ├── classification.py  
│   ├── ingestion.py       
│   ├── summarization.py    
│   ├── schemas.py         
│   ├── utils.py            
│   └── config.py         
├── requirements.txt       
├── setup.py           
└── README.md       
```

## Installation

1. **Clone the repo**:

   ```bash
   git clone https://github.com/mwi-kali/text-classifier.git
   cd Text-Classifier
   ```

2. **Create & activate a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:

```bash
streamlit run text_classifier/app.py
```

* Upload a file or enter a URL in the sidebar.
* Toggle classification & summarization settings.
* View results in the interactive UI.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to your branch (`git push origin feature/name`).
5. Open a Pull Request.
