
# import os
# import io
# import json
# import torch
# from PIL import Image
# from transformers import (
#     TrOCRProcessor,
#     VisionEncoderDecoderModel,
#     AutoTokenizer,
#     AutoModelForCausalLM,
# )
# from app.domain.schemas import DocumentExtraction, PlotData

# # =============================
# # OCR SERVICE (IMAGE → TEXT)
# # =============================

# class OCRService:
#     def __init__(self):
#         print("--- Loading OCR Model (TrOCR) ---")
#         try:
#             self.processor = TrOCRProcessor.from_pretrained(
#                 "microsoft/trocr-base-printed"
#             )
#             self.model = VisionEncoderDecoderModel.from_pretrained(
#                 "microsoft/trocr-base-printed"
#             )
#             # Use GPU if available
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
#             self.model.to(self.device)
#             print(f"OCR loaded on {self.device}")
#         except Exception as e:
#             print(f"Failed to load OCR model: {e}")
#             raise e

#     def extract_text(self, image: Image.Image) -> str:
#         if image.mode != "RGB":
#             image = image.convert("RGB")
            
#         pixel_values = self.processor(
#             image, return_tensors="pt"
#         ).pixel_values.to(self.device)

#         with torch.no_grad():
#             ids = self.model.generate(pixel_values)

#         text = self.processor.batch_decode(
#             ids, skip_special_tokens=True
#         )[0]

#         return text.strip()


# # =============================
# # TEXT → STRUCTURED JSON
# # =============================

# class TextStructuringService:
#     def __init__(self):
#         print("--- Loading Text LLM (Qwen2.5-7B) ---")
#         self.model_id = "Qwen/Qwen2.5-7B-Instruct"
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.model_id,
#                 device_map="auto",
#                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#                 low_cpu_mem_usage=True
#             )
#         except Exception as e:
#              print(f"Failed to load LLM: {e}")
#              raise e

#     def structure(self, text: str) -> DocumentExtraction:
#         prompt = f"""
# You are a document extraction system.

# Return ONLY valid JSON matching exactly this schema:
# {{
#   "global_elevation": null,
#   "global_notes": null,
#   "plots": [
#     {{
#       "lot_no": null,
#       "block": null,
#       "address": null,
#       "model_selected": null,
#       "elevation": null,
#       "garage_swing": null,
#       "external_structure": null,
#       "optional_notes": null
#     }}
#   ]
# }}

# DOCUMENT TEXT:
# {text}
# """

#         inputs = self.tokenizer(
#             prompt, return_tensors="pt"
#         ).to(self.model.device)

#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=1024,
#                 temperature=0.1
#             )

#         response = self.tokenizer.decode(
#             outputs[0], skip_special_tokens=True
#         )

#         try:
#             # Extract JSON substring
#             json_text = response
#             if "{" in json_text:
#                 json_text = json_text[json_text.find("{"): json_text.rfind("}") + 1]
            
#             return DocumentExtraction.model_validate_json(json_text)
#         except Exception as e:
#             print(f"JSON Parsing Error: {e}")
#             return DocumentExtraction(
#                 global_elevation=None,
#                 global_notes=None,
#                 plots=[PlotData(optional_notes=f"Parsing Error: {str(e)} -- Raw: {response[:100]}...")]
#             )


# # =============================
# # MAIN SERVICE WRAPPER
# # =============================

# class LocalHuggingFaceService:
#     _instance = None

#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super(LocalHuggingFaceService, cls).__new__(cls)
#             cls._instance._initialize_services()
#         return cls._instance

#     def _initialize_services(self):
#         self.ocr = OCRService()
#         self.text_llm = TextStructuringService()

#     def query_document(self, image_parts):
#         """
#         image_parts: List of dicts, but for local service dealing with raw images might be cleaner.
#         The extractor logic passes base64 strings in the dict.
#         We need to decode them back to PIL.
#         """
#         import base64
        
#         extracted_texts = []
        
#         for part in image_parts:
#             if part.get("type") == "image_url":
#                 url = part["image_url"]["url"]
#                 # Format: "data:image/jpeg;base64,..."
#                 if url.startswith("data:image"):
#                     base64_str = url.split(",")[1]
#                     image_data = base64.b64decode(base64_str)
#                     image = Image.open(io.BytesIO(image_data))
                    
#                     # OCR processing
#                     text = self.ocr.extract_text(image)
#                     extracted_texts.append(text)

#         full_text = "\n".join(extracted_texts)
#         print(f"--- Extracted Text Preview: {full_text[:50]}... ---")

#         if not full_text.strip():
#              return DocumentExtraction(
#                 global_elevation=None,
#                 global_notes=None,
#                 plots=[PlotData(optional_notes="No text detected in document.")]
#             )

#         # Structuring
#         return self.text_llm.structure(full_text)
