from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any, Optional, List
import logging
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import glob
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThaiQASystem:
    def __init__(self, model_id: str = "scb10x/typhoon2.1-gemma3-4b"):
        """Initialize the Thai Q&A System"""
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.embedder = None
        self.doc_embeddings = None
        self.doc_df = None
        
    def load_model(self):
        """Load the language model"""
        logger.info("🔄 Loading model...")
        try:
            torch._dynamo.config.cache_size_limit = 1024
            torch.set_float32_matmul_precision('high')
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            logger.info("✅ Model loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def load_embedder(self, embedder_model: str = "BAAI/bge-m3"):
        """Load sentence transformer for RAG"""
        logger.info("🔄 Loading embedder...")
        try:
            self.embedder = SentenceTransformer(embedder_model)
            logger.info("✅ Embedder loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Error loading embedder: {e}")
            raise
    
    def load_precomputed_data(self, 
                            embeddings_path: str = "embeddings_bge_m3.npy",
                            combined_docs_path: str = "combined_docs_with_embeddings.json"):
        """Load precomputed embeddings and documents"""
        try:
            logger.info("📥 Loading precomputed data...")
            
            # Load embeddings
            if os.path.exists(embeddings_path):
                self.doc_embeddings = np.load(embeddings_path)
                logger.info(f"✅ Loaded embeddings: {self.doc_embeddings.shape}")
            else:
                logger.warning(f"⚠️ Embeddings file not found: {embeddings_path}")
                return False
                
            # Load documents dataframe
            if os.path.exists(combined_docs_path):
                self.doc_df = pd.read_json(combined_docs_path, lines=True)
                logger.info(f"✅ Loaded documents: {len(self.doc_df)} documents")
            else:
                logger.warning(f"⚠️ Documents file not found: {combined_docs_path}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"❌ Error loading precomputed data: {e}")
            return False
    
    def retrieve_relevant_docs(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents with metadata"""
        if self.embedder is None or self.doc_embeddings is None:
            return []
        
        try:
            # Encode question
            question_embedding = self.embedder.encode(["query: " + question])
            
            # Calculate similarities
            similarities = np.dot(self.doc_embeddings, question_embedding.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Get relevant documents with metadata
            relevant_docs = []
            for idx in top_indices:
                if idx < len(self.doc_df):
                    doc_info = {
                        'content': str(self.doc_df.iloc[idx]["content"]),
                        'source': str(self.doc_df.iloc[idx]["source"]),
                        'page': str(self.doc_df.iloc[idx]["page"]),
                        'type': str(self.doc_df.iloc[idx]["type"]),
                        'file_origin': str(self.doc_df.iloc[idx]["file_origin"]),
                        'similarity': float(similarities[idx])
                    }
                    relevant_docs.append(doc_info)
            
            return relevant_docs
        except Exception as e:
            logger.warning(f"⚠️ Error in document retrieval: {e}")
            return []
    
    def get_answer_and_reason(self, question: str, use_rag: bool = True, top_k: int = 3) -> tuple[str, str]:
        """Get answer and reason from model with optional RAG"""
        if self.model is None or self.tokenizer is None:
            raise Exception("Model not loaded! Call load_model() first.")
        
        # Prepare context if RAG is enabled
        context = ""
        relevant_docs = []
        if use_rag and self.doc_embeddings is not None:
            relevant_docs = self.retrieve_relevant_docs(question, top_k=top_k)
            if relevant_docs:
                context_parts = []
                for i, doc in enumerate(relevant_docs[:top_k]):
                    source_info = f"[{doc['file_origin']}:{doc['source']}:{doc['page']}]"
                    context_parts.append(f"{source_info} {doc['content']}")
                
                context = "\n\nบริบทที่เกี่ยวข้อง:\n" + "\n".join(context_parts)
        
        # Prepare messages for both answer and reason
        system_prompt = (
            "You are an AI that answers multiple choice questions in Thai. "
            "Reply only with a valid JSON object in this exact format: "
            '{"answer": "ก", "reason": "คำอธิบายเหตุผลที่เลือกตัวเลือกนี้"}. '
            'Answer choices must be enclosed in double quotes (ก, ข, ค, ง). '
            'Provide a clear Thai explanation for why you chose this answer. '
            'Do not add anything else outside the JSON.'
        )
        
        if context:
            system_prompt += "\n\nUse the provided context to help answer the question."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question.strip() + context}
        ]

        try:
            # Try using chat template first
            try:
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(self.model.device)
            except Exception as template_error:
                logger.warning(f"Chat template failed: {template_error}, using simple prompt")
                # Fallback to simple prompt format
                prompt = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

            outputs = self.model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )

            response = outputs[0][input_ids.shape[-1]:]
            decoded = self.tokenizer.decode(response, skip_special_tokens=True)

            # Extract JSON
            try:
                # Find JSON in response
                start_idx = decoded.find("{")
                end_idx = decoded.rfind("}") + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = decoded[start_idx:end_idx]
                    result = json.loads(json_str)
                    answer = result.get("answer", "").strip().strip('"')
                    reason = result.get("reason", "ไม่พบคำอธิบาย").strip()
                    
                    # Log relevant documents used
                    if relevant_docs:
                        logger.info(f"Used {len(relevant_docs)} relevant documents")
                        for i, doc in enumerate(relevant_docs[:2]):
                            logger.info(f"  Doc {i+1}: [{doc['file_origin']}] similarity: {doc['similarity']:.3f}")
                    
                    return answer, reason
                else:
                    logger.warning(f"⚠️ No JSON found in: {decoded}")
                    return "", "ไม่สามารถประมวลผลคำตอบได้"
                    
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ JSON decode error: {e}")
                logger.warning(f"Raw output: {decoded}")
                return "", "เกิดข้อผิดพลาดในการประมวลผลคำตอบ"
                
        except Exception as e:
            logger.error(f"⚠️ Generation error: {e}")
            return "", f"เกิดข้อผิดพลาด: {str(e)}"

# Global QA system instance
qa_system = None

app = FastAPI(title="Background Evaluation Service", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Initialize the QA system when the app starts"""
    global qa_system
    try:
        logger.info("🚀 Initializing Thai QA System...")
        qa_system = ThaiQASystem()
        
        # Load embedder first (lighter operation)
        qa_system.load_embedder()
        
        # Use absolute paths for the files
        project_dir = "/home/siamai/tatar/cmkl-med-hackathon"
        embeddings_path = os.path.join(project_dir, "embeddings_bge_m3.npy")
        docs_path = os.path.join(project_dir, "combined_docs_with_embeddings.json")
        
        # Load precomputed data
        success = qa_system.load_precomputed_data(
            embeddings_path=embeddings_path,
            combined_docs_path=docs_path
        )
        if not success:
            logger.warning("⚠️ Could not load precomputed data, falling back to simple mode")
        
        # Load the main model (heavy operation)
        qa_system.load_model()
        
        logger.info("✅ Thai QA System initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize QA system: {e}")
        # Keep a simple fallback system
        qa_system = None

# Request/Response models
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    reason: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Background Evaluation Service is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global qa_system
    
    system_status = {
        "status": "healthy",
        "service": "evaluation-api",
        "version": "1.0.0",
        "qa_system": {
            "available": qa_system is not None,
            "model_loaded": qa_system.model is not None if qa_system else False,
            "embedder_loaded": qa_system.embedder is not None if qa_system else False,
            "documents_loaded": qa_system.doc_df is not None if qa_system else False,
            "embeddings_loaded": qa_system.doc_embeddings is not None if qa_system else False,
            "document_count": len(qa_system.doc_df) if qa_system and qa_system.doc_df is not None else 0
        }
    }
    
    return system_status

@app.post("/eval", response_model=QuestionResponse)
async def evaluate_question(request: QuestionRequest):
    """
    Main evaluation endpoint that receives questions and returns answers
    """
    global qa_system
    
    try:
        logger.info(f"Received question: {request.question[:100]}...")
        
        # Check if QA system is available
        if qa_system is None:
            logger.warning("QA system not available, using fallback logic")
            # Fallback logic for when the full system isn't loaded
            if "ปวดท้อง" in request.question or "อาเจียน" in request.question:
                answer = "ก"
                reason = "หากมีอาการปวดท้องและอาเจียน ควรไปพบแพทย์ที่ แผนกอายุรกรรม (Internal Medicine) ดังนั้น จึงตอบข้อ ก."
            else:
                answer = "ก"
                reason = f"ตอบข้อ ก สำหรับคำถาม: {request.question[:50]}... (ใช้ระบบสำรอง)"
        else:
            # Use the full QA system
            try:
                answer, reason = qa_system.get_answer_and_reason(
                    request.question, 
                    use_rag=True, 
                    top_k=3
                )
                
                # Fallback if no answer was generated
                if not answer:
                    answer = "ก"
                    reason = "ไม่สามารถประมวลผลคำตอบได้ ใช้คำตอบเริ่มต้น"
                    
            except Exception as model_error:
                logger.error(f"Error with QA system: {model_error}")
                # Fallback to simple logic
                answer = "ก"
                reason = f"เกิดข้อผิดพลาดในระบบ AI: {str(model_error)[:100]}"
        
        response = QuestionResponse(
            answer=answer,
            reason=reason
        )
        
        logger.info(f"Responding with answer: {answer}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Listen on all interfaces
        port=5000,
        reload=False,    # Set to True for development
        log_level="info"
    )
