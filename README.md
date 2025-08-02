# CMKL Medical Hackathon 🏥

A Thai medical document analysis system using Typhoon OCR and advanced language models for enhanced text extraction and question-answering capabilities.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/PatiphanAK/cmkl-med-hackathon
cd cmkl-med-hackathon
```

2. **Create and activate virtual environment**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
# Install all dependencies from pyproject.toml
uv sync

```

4. **Verify installation**
```bash
uv pip list
```

## 🏃‍♂️ Running the Application

### Start the FastAPI server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Launch Jupyter Notebook (for development)
```bash
jupyter notebook
```

### Run in production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## ⚙️ Dependencies Overview

| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| **ML Framework** | torch | >=2.7.1 | PyTorch deep learning |
| **Transformers** | transformers | >=4.54.1 | Hugging Face models |
| **Acceleration** | accelerate | >=1.9.0 | Model optimization |
| **Quantization** | bitsandbytes | >=0.46.1 | Memory efficiency |
| **Embeddings** | sentence-transformers | >=5.0.0 | Text embeddings |
| **LangChain** | langchain* | >=0.3.27 | LLM orchestration |
| **Vector DB** | chromadb | >=1.0.15 | Vector storage |
| **API** | fastapi | >=0.116.1 | Web framework |
| **Data** | pandas | >=2.3.1 | Data manipulation |
| **PDF** | pypdf2 | >=3.0.1 | PDF processing |

## 💡 Why Dual Typhoon Architecture?

This project leverages **both Typhoon models** for optimal medical document processing:

### 🔍 Typhoon OCR (7B) - Document Processing
As demonstrated in our presentation (page 6), [Typhoon OCR 7B](https://huggingface.co/scb10x/typhoon-ocr-7b) offers superior capabilities:
- **Better text structure preservation** compared to pytesseract
- **Enhanced Thai character recognition** for medical documents
- **Optimized for official PDF extraction** with medical context
- **7B parameters** dedicated to OCR tasks

### 🧠 Typhoon 2.1 (4B) - Language Understanding
[Typhoon 2.1 Gemma3 4B](https://huggingface.co/scb10x/typhoon2.1-gemma3-4b) handles the reasoning:
- **Advanced Thai-English question answering**
- **Medical context comprehension**
- **Efficient 4B parameter architecture**
- **Optimized for conversational AI**

## 📊 Model Performance

Based on benchmark results (page 7):
- **Typhoon2.1-4B** achieves exceptional Thai-language QA performance
- Particularly strong on:
  - IF-Eval benchmarks
  - Code-Switching scenarios
  - General QA tasks

## 📈 Achievement

Our best submission score (page 2):
> 🔥 **Final Score: 0.82000**
> 
> Significant improvement from baseline: 0.39600
> 
> *(Initial version without RAG implementation)*

## 🏗️ Architecture Overview

```
Medical Document → Typhoon OCR 7B → Extracted Text → Typhoon 2.1 4B → Answer
                      ↓                    ↓              ↓
                 Text Structure      BGE-M3 Embeddings   ChromaDB
                 Preservation         Vector Storage      Retrieval
```

### Pipeline Flow:
1. **Document Input**: Medical PDFs with Thai/English content
2. **OCR Processing**: Typhoon OCR 7B extracts and structures text
3. **Embedding**: BGE-M3 creates semantic vectors
4. **Storage**: ChromaDB stores document vectors
5. **Query Processing**: User questions processed through RAG
6. **Answer Generation**: Typhoon 2.1 4B generates contextual responses

## 🛠️ Project Structure

```
cmkl-med-hackathon/
├── main.py                 # FastAPI application
├── data/                   # Training/test data
├── notebooks/              # Jupyter notebooks
├── requirements/           # Dependency files
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## 🔧 Configuration

### Model Settings
- **OCR Model**: [scb10x/typhoon-ocr-7b](https://huggingface.co/scb10x/typhoon-ocr-7b)
- **Language Model**: [scb10x/typhoon2.1-gemma3-4b](https://huggingface.co/scb10x/typhoon2.1-gemma3-4b)
- **Embedder**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- **Vector Store**: ChromaDB

### Environment Variables
```bash
# Add to .env file
HUGGINGFACE_TOKEN=your_token_here
OCR_MODEL_PATH=scb10x/typhoon-ocr-7b
LANGUAGE_MODEL_PATH=scb10x/typhoon2.1-gemma3-4b
EMBEDDER_PATH=BAAI/bge-m3
```

## 🚀 Future Improvements

- [ ] **Scalable Vector Search**: Integrate FAISS or PgVector
- [ ] **User Interface**: Add Gradio UI for query input
- [ ] **Batch Processing**: Support multiple documents simultaneously
- [ ] **API Enhancement**: RESTful endpoints for document upload
- [ ] **Monitoring**: Add logging and performance metrics

## 🧠 Credits

- **👨‍💻 Author**: Patiphan
- **🏫 Institution**: CMKL University
- **🔍 OCR Model**: [Typhoon-OCR-7B](https://huggingface.co/scb10x/typhoon-ocr-7b)
- **🤖 Language Model**: [Typhoon2.1-Gemma3-4B](https://huggingface.co/scb10x/typhoon2.1-gemma3-4b)
- **📊 Embedding Model**: [BGE-M3](https://huggingface.co/BAAI/bge-m3)

## 📝 Usage Example


## 🐛 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

2. **Memory Issues**
```bash
# Use CPU-only mode
export CUDA_VISIBLE_DEVICES=""
```

3. **Dependency Conflicts**
```bash
# Clean install
uv venv --clear
source .venv/bin/activate
uv sync
```

## 📄 License

This project is developed for the CMKL Medical Hackathon 2025. Please refer to the competition guidelines for usage terms.

**Happy Coding! 🚀**