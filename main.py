# main.py
import os
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from config import DB_FAISS_PATH, EMBEDDING_MODEL, OLLAMA_MODEL, MAX_UPLOAD_SIZE, ALLOWED_AUDIO_EXTENSIONS
from prompts import SYSTEM_PROMPT
import tempfile
import uuid
import shutil
from pathlib import Path
import traceback
import re

# Initialize FastAPI app
app = FastAPI(title="Caregiver AI Assistant", description="A multilingual RAG-powered caregiving assistant with voice capabilities")

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL VARIABLES ---
# This dictionary will hold our loaded models and retriever to avoid reloading on each request
# We'll populate it in the startup event
chatbot_globals = {}

# Mount static files and templates for serving the frontend UI
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

@app.on_event("startup")
async def startup_event():
    """
    Load all the necessary models and retriever at application startup.
    This is the recommended practice to avoid long loading times for each API call.
    """
    print("--- Application Startup: Initializing Chatbot ---")

    # Load embeddings model
    print("Loading embedding model...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if available
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        print("Please ensure sentence-transformers is installed: pip install sentence-transformers")
        raise e

    # Load FAISS vector store
    if not os.path.exists(DB_FAISS_PATH):
        print(f"Vector database not found at {DB_FAISS_PATH}.")
        print("Attempting to create it by running ingest.create_vector_db()...")
        try:
            # Import here to avoid circular import at module level
            from ingest import create_vector_db
            create_vector_db()
        except Exception as ie:
            print(f"Failed to create FAISS index: {ie}")
            print("Please ensure you have documents in the 'data/' directory and run 'python ingest.py' manually.")
            # Create empty vector store for demo purposes
            from langchain.schema import Document
            sample_doc = Document(page_content="Welcome to the Caregiver AI Assistant. I'm here to help with caregiving questions.", metadata={"source": "system"})
            vector_store = FAISS.from_documents([sample_doc], embeddings)
            vector_store.save_local(DB_FAISS_PATH)
            print("Created demo vector store with sample content.")

    print(f"Loading vector database from {DB_FAISS_PATH}...")
    try:
        # Attempt to load existing FAISS index
        vector_store = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Vector database loaded successfully.")
    except Exception as e:
        # If loading fails (e.g., pickle schema mismatch), attempt to recreate the index
        print(f"Failed to load FAISS index: {e}")
        print("Attempting to rebuild the FAISS index by running ingest.create_vector_db()...")
        try:
            # Import here to avoid circular import at module level
            from ingest import create_vector_db
            create_vector_db()
        except Exception as ie:
            print(f"Failed to rebuild FAISS index: {ie}")
            print("Creating minimal vector store for demo...")
            from langchain.schema import Document
            sample_doc = Document(page_content="Welcome to the Caregiver AI Assistant. I'm here to help with caregiving questions.", metadata={"source": "system"})
            vector_store = FAISS.from_documents([sample_doc], embeddings)
            vector_store.save_local(DB_FAISS_PATH)

        # Retry loading after rebuild
        try:
            vector_store = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            print("Vector database rebuilt and loaded successfully.")
        except Exception as final_e:
            print(f"Final attempt to load FAISS index failed: {final_e}")
            raise final_e

    retriever = vector_store.as_retriever(search_kwargs={'k': 4})

    # Initialize Ollama LLM
    print(f"Initializing Ollama model: {OLLAMA_MODEL}...")
    try:
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)
        # Test the connection
        test_response = llm.invoke("Hello")
        print("Ollama model initialized and tested successfully.")
    except Exception as e:
        print(f"Error initializing Ollama: {e}")
        print(f"Please ensure Ollama is running and the model '{OLLAMA_MODEL}' is downloaded.")
        print("To install Ollama: https://ollama.ai/")
        print(f"To download the model: ollama pull {OLLAMA_MODEL}")
        # Continue without LLM for now - will handle in endpoints
        llm = None

    # Create the RAG chain
    if llm:
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ])
        document_chain = create_stuff_documents_chain(llm, answer_prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        print("RAG chain created.")
    else:
        retrieval_chain = None
        print("RAG chain not created due to missing LLM.")

    # Store the chain in the global dictionary
    chatbot_globals['retrieval_chain'] = retrieval_chain
    chatbot_globals['retriever'] = retriever
    chatbot_globals['llm'] = llm
    print("--- Chatbot initialization complete! ---")


def _safe_write_upload_to_temp(upload_file: UploadFile) -> Path:
    """Save UploadFile to a temporary file and return its Path."""
    suffix = Path(upload_file.filename).suffix or ".wav"
    temp_dir = Path(tempfile.gettempdir()) / "rag_audio"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4().hex}{suffix}"
    with temp_path.open("wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return temp_path


def _cleanup_temp(path: Path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


# --- API ENDPOINTS ---

@app.post("/api/chat")
async def chat(request: Request):
    """
    Handles the chat logic. Receives user input and streams back the chatbot's response.
    """
    try:
        data = await request.json()
        user_input = data.get("message")
        if not user_input:
            return JSONResponse({"error": "No message provided."}, status_code=400)

        retrieval_chain = chatbot_globals.get('retrieval_chain')
        if not retrieval_chain:
            return JSONResponse({"error": "Chatbot is not initialized. Please check that Ollama is running and the model is available."}, status_code=500)

        async def stream_generator():
            """A generator function to stream the response chunks."""
            full_answer = ""
            try:
                async for chunk in retrieval_chain.astream({"input": user_input}):
                    if answer_part := chunk.get("answer"):
                        # Yield each part of the answer as it's generated
                        yield answer_part
                        full_answer += answer_part
            except Exception as e:
                yield f"Sorry, I encountered an error while processing your request: {str(e)}"

        return StreamingResponse(stream_generator(), media_type="text/plain")

    except Exception as e:
        print(f"An error occurred in /api/chat: {e}")
        return JSONResponse({"error": "An internal error occurred."}, status_code=500)


# --- Compatibility alias routes for frontend (old paths) ---
@app.post("/chat")
async def chat_alias(request: Request):
    return await chat(request)


@app.post("/api/audio-chat")
async def audio_chat(file: UploadFile = File(...)):
    """Endpoint to handle audio-to-audio chat.

    Steps:
    1. Save uploaded audio to temp file
    2. Transcribe using OpenAI Whisper (via openai-whisper or whisper.cpp) - we'll use `whisper` (pip) if available
    3. Pass text through existing retrieval_chain
    4. Synthesize response via gTTS (supports Arabic + English)
    5. Return JSON with text and audio file
    """
    temp_audio_path = None
    temp_tts_path = None
    try:
        # Validate file extension
        ext = Path(file.filename or "").suffix.lower()
        if ext and ext not in ALLOWED_AUDIO_EXTENSIONS:
            return JSONResponse({"error": "Unsupported audio format."}, status_code=400)

        # Save upload to temp file
        temp_audio_path = _safe_write_upload_to_temp(file)

        # Validate file size
        try:
            if temp_audio_path.stat().st_size > MAX_UPLOAD_SIZE:
                _cleanup_temp(temp_audio_path)
                return JSONResponse({"error": "File too large."}, status_code=400)
        except Exception:
            pass

        # Try faster-whisper first (better Windows support); fall back to openai-whisper
        transcription = ""
        language = ""
        # Attempt faster-whisper
        try:
            from faster_whisper import WhisperModel
            try:
                fw_model = WhisperModel("small", device="cpu")
                segments, info = fw_model.transcribe(str(temp_audio_path), beam_size=5)
                transcription = " ".join([segment.text for segment in segments]).strip()
                language = getattr(info, 'language', '') or ''
                # If transcription isn't Arabic, try forcing Arabic decoding
                if not re.search(r"[\u0600-\u06FF]", transcription):
                    try:
                        segments2, info2 = fw_model.transcribe(str(temp_audio_path), beam_size=5, language='ar')
                        forced_text = " ".join([s.text for s in segments2]).strip()
                        if re.search(r"[\u0600-\u06FF]", forced_text):
                            transcription = forced_text
                            language = 'ar'
                    except Exception:
                        pass
            except Exception as fw_exc:
                print("faster-whisper model error:", repr(fw_exc))
                raise
        except Exception:
            # Fallback to the whisper package
            try:
                import whisper
                model = whisper.load_model("small")
                result = model.transcribe(str(temp_audio_path))
                if result and 'text' in result:
                    transcription = (result.get("text") or "").strip()
                    language = result.get("language") or ""
                    # If transcription isn't Arabic, try a forced Arabic pass
                    if not re.search(r"[\u0600-\u06FF]", transcription):
                        try:
                            result2 = model.transcribe(str(temp_audio_path), language='ar')
                            forced_text = (result2.get("text") or "").strip()
                            if re.search(r"[\u0600-\u06FF]", forced_text):
                                transcription = forced_text
                                language = 'ar'
                        except Exception:
                            pass
                else:
                    print("Whisper returned no transcription result:", result)
                    return JSONResponse({"error": "Could not transcribe audio."}, status_code=400)
            except Exception as ie:
                print("STT import or runtime failed:", repr(ie))
                print(traceback.format_exc())
                # Graceful degradation: return text-only message so endpoint doesn't 500
                return JSONResponse({
                    "text": "لم أستطع تحويل الصوت إلى نص على الخادم حالياً. من فضلك جرّب الكتابة بدل التسجيل.",
                    "audio_url": None,
                    "transcription": "",
                    "warning": "STT backend not available on server. Install 'faster-whisper' or 'whisper' and ensure ffmpeg is on PATH."
                }, status_code=200)

        # Fallback: detect Arabic characters in the transcription if language isn't provided
        if not language:
            if re.search(r"[\u0600-\u06FF]", transcription):
                language = 'ar'
            else:
                language = 'en'

        if not transcription:
            return JSONResponse({
                "text": "لم ألتقط أي كلام واضح من التسجيل. جرّب التحدّث بوضوح أو اكتب رسالتك.",
                "audio_url": None,
                "transcription": ""
            }, status_code=200)

        # Run through RAG pipeline (sync wrapper)
        retrieval_chain = chatbot_globals.get('retrieval_chain')
        if not retrieval_chain:
            return JSONResponse({
                "text": "النظام غير جاهز حالياً للرد الذكي. يمكنك طرح سؤالك نصياً لاحقاً.",
                "audio_url": None,
                "transcription": transcription
            }, status_code=200)

        # Use existing async streaming interface but collect full answer synchronously
        user_query = transcription
        if language and language.startswith('ar'):
            user_query = "تعليمات مهمة: أجب باللغة العربية الفصحى فقط دون استخدام أي لغة أخرى. ثم أجب على السؤال التالي:\n" + transcription
        full_answer = ""
        try:
            async for chunk in retrieval_chain.astream({"input": user_query}):
                if part := chunk.get("answer"):
                    full_answer += part
        except Exception as e:
            full_answer = f"Sorry, I encountered an error while processing your request: {str(e)}"

        # Synthesize TTS using gTTS
        try:
            from gtts import gTTS
        except Exception as te:
            print("gTTS not available:", te)
            # Graceful degradation: return text only
            return {
                "text": full_answer or "تم استلام سؤالك وتم توليد الرد النصي فقط.",
                "audio_url": None,
                "transcription": transcription,
                "warning": "TTS backend not available on server."
            }

        # Choose language code for gTTS (default: Arabic 'ar' or English 'en')
        tts_lang = 'en'
        if language and language.startswith('ar'):
            tts_lang = 'ar'

        temp_tts_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=full_answer, lang=tts_lang)
        tts.save(str(temp_tts_path))

        # Return text and audio file path as attachment
        # We'll return the audio as a streamed FileResponse and the text in JSON
        # To keep the API simple for the frontend, return a multipart-like JSON with a URL to download audio
        audio_url = f"/api/audio/{temp_tts_path.name}"

        # Store mapping for serving (simple approach: copy file to temp dir served by FileResponse)
        serve_dir = Path(tempfile.gettempdir()) / "rag_tts"
        serve_dir.mkdir(parents=True, exist_ok=True)
        target_path = serve_dir / temp_tts_path.name
        shutil.copy(str(temp_tts_path), str(target_path))

        return {"text": full_answer, "audio_url": f"/api/audio/{target_path.name}", "transcription": transcription}

    except Exception as e:
        print("Error in /api/audio-chat:", e)
        return JSONResponse({"error": "Internal server error."}, status_code=500)
    finally:
        # cleanup uploaded audio file; keep TTS file for short time so frontend can fetch it
        if temp_audio_path:
            _cleanup_temp(temp_audio_path)


@app.post("/audio-chat")
async def audio_chat_alias(file: UploadFile = File(...)):
    return await audio_chat(file)


@app.get("/api/audio/{filename}")
async def serve_audio(filename: str):
    serve_dir = Path(tempfile.gettempdir()) / "rag_tts"
    file_path = serve_dir / filename
    if not file_path.exists():
        return JSONResponse({"error": "Audio file not found."}, status_code=404)

    return FileResponse(path=str(file_path), media_type="audio/mpeg", filename=filename)


@app.get("/audio/{filename}")
async def serve_audio_alias(filename: str):
    return await serve_audio(filename)

@app.get("/api/health")
async def health_check():
    """A simple endpoint to check if the server is running."""
    llm_status = "available" if chatbot_globals.get('llm') else "unavailable"
    retrieval_status = "ready" if chatbot_globals.get('retrieval_chain') else "not ready"
    
    return {
        "status": "ok",
        "llm_status": llm_status,
        "retrieval_status": retrieval_status,
        "vector_db_path": DB_FAISS_PATH,
        "embedding_model": EMBEDDING_MODEL,
        "ollama_model": OLLAMA_MODEL
    }


@app.get("/health")
async def health_alias():
    return await health_check()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main UI page."""
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT

    # Use configured host and port so `python main.py` respects `config.py`
    uvicorn.run(app, host=HOST, port=PORT)
