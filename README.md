# cmkl-med-hackathon

## Init Project
```
uv venv

source .venv/bin/activate
```
 
---

## ⚙️ Dependencies

- `torch`
- `transformers`
- `sentence-transformers`
- `pandas`
- `tqdm`
- `typhoon-ocr` (custom or external script)
- `faiss-cpu` *(optional)*

---

## 💡 Why Typhoon OCR?

As shown on *page 6* of the presentation, Typhoon OCR demonstrates better text structure preservation and Thai character recognition compared to `pytesseract`, making it ideal for extracting medical context from official PDFs.

---

## 📊 Model Performance

Refer to *page 7* for benchmark comparison:
- `Typhoon2.1-4B` achieves high Thai-language QA performance
- Especially strong on IF-Eval, Code-Switching, and general QA benchmarks

---

## 📈 Best Submission Score

As highlighted on *page 2*, this system achieved:
> 🔥 **Score: 0.82000**  
> Compared to earlier baseline: 0.39600  
> (Without using RAG initially!)

---

## 🧠 Credits

- 🧑‍💻 Author: Patiphan
- 🏥 Institution: CMKL University
- 🧪 Model: [scb10x/typhoon2.1-gemma3-4b](https://huggingface.co/scb10x/typhoon2.1-gemma3-4b)
- 🔍 Embedder: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)

---

## 🚀 Future Improvements

- Integrate FAISS for scalable vector search
- Add UI with Gradio for query input
- Support multiple documents in batch
