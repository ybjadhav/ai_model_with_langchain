from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import uvicorn
from app.services.extractor_logic import DocumentExtractor
from app.domain.schemas import DocumentExtraction
from dotenv import load_dotenv

load_dotenv()

# Singleton logic layer
extractor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global extractor
    print("--- Startup: Loading Google Gemini Service ---")
    try:
        extractor = DocumentExtractor()
    except Exception as e:
        print(f"CRITICAL ERROR during model load: {e}")
    yield
    print("--- Shutdown: Cleaning up ---")

app = FastAPI(title="Quickplot Extraction API", lifespan=lifespan)

@app.get("/")
async def root():
    status = "active" if extractor else "inactive"
    return {"status": status, "service": "LangChain + Gemini"}

@app.post("/extract")
async def extract_plot_data(file: UploadFile = File(...)):
    if not extractor:
        raise HTTPException(status_code=503, detail="Model initialization failed.")

    try:
        # # Read file into memory
        # file_bytes = await file.read()
        # if not file_bytes:
        #     raise HTTPException(status_code=400, detail="Empty file uploaded.")

        # # Process through Service Logic
        # result = extractor.process(file_bytes, file.filename)
        # return result
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        # Gemini returns DocumentExtraction
        extraction: DocumentExtraction = extractor.process(file_bytes, file.filename)

        return extraction
    except ValueError as e:
         raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run("app.main:app",port=2026, reload=False)
