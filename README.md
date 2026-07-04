# Text Summarizer Web App

A lightweight web application built with **FastAPI** and **Hugging Face Transformers** (T5 model) to generate quick summaries of dialogue and text content.

## Features

- **FastAPI Backend**: Clean and efficient REST API to handle text summarization requests.
- **T5 Model Summarization**: Utilizes a fine-tuned Hugging Face T5 model trained on dialogue summarization datasets (SAMSum).
- **Interactive UI**: Responsive and simple frontend interface built with HTML, CSS, and vanilla JavaScript to submit text and view summaries instantly.
- **Hardware Acceleration**: Automatically detects and leverages GPU/MPS (Apple Silicon) if available for faster model inference.

## Project Structure

```
├── static/
│   ├── flow.js               # Frontend JavaScript for API requests
│   └── style.css             # Main stylesheet for UI styling
├── templates/
│   └── index.html            # Main UI HTML layout
├── saved_summary_model/      # Pre-trained Hugging Face T5 model files (ignored in git)
├── app.py                    # FastAPI server entry point
├── requirements.txt          # Python dependencies (optimized for CPU for cloud hosting)
├── .gitignore                # Git ignore configuration
└── README.md                 # Project documentation
```

## Getting Started

### Prerequisites

Make sure you have Python installed. You can install all required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Running Locally

1. Clone this repository.
2. Download or save your trained model inside a folder named `saved_summary_model/` at the root of the project.
3. Start the FastAPI local server:
   ```bash
   uvicorn app:app --reload
   ```
4. Open your browser and go to:
   ```
   http://127.0.0.1:8000
   ```

---

## Deploying to Render

To deploy this app on [Render](https://render.com/):

### 1. Model Storage Considerations
Since the fine-tuned T5 model files (~242MB) are ignored by git (due to large file sizes), you have two options for Render:
- **Option A (Recommended)**: Upload the model to the Hugging Face Hub (public or private), and change `app.py` to load the model from Hugging Face instead of a local folder (e.g., `model = T5ForConditionalGeneration.from_pretrained("your-username/your-model-name")`).
- **Option B**: Commit the model files using **Git LFS** (Git Large File Storage), though note that Render build times might be longer.

### 2. Render Settings
When creating a new **Web Service** on Render:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn -w 1 -k uvicorn.workers.UvicornWorker app:app -b 0.0.0.0:$PORT` (uses 1 worker to fit Render's 512MB RAM limit)

---

## API Endpoints

### 1. Home Page
- **URL**: `/`
- **Method**: `GET`
- **Response**: Serves the interactive Web UI.

### 2. Summarize Text
- **URL**: `/summarize/`
- **Method**: `POST`
- **Payload**:
  ```json
  {
    "dialogue": "Insert your long dialogue or paragraph here."
  }
  ```
- **Response**:
  ```json
  {
    "summary": "Generated summary text here."
  }
  ```
