# import io
# import base64
# from PIL import Image
# import pymupdf # imports as fitz
# from app.domain.schemas import PlotData

# class DocumentExtractor:
#     def __init__(self):
#         # Lazy load infrastructure to avoid circular imports or immediate failures
#         from app.infrastructure.gemini_service import GeminiService
#         self.ai = GeminiService()

#     def _pil_to_base64_content(self, img: Image.Image):
#         """Convert PIL image to user message content format for LangChain/Gemini."""
#         buffered = io.BytesIO()
#         # Save as JPEG for efficiency, or PNG if needed.
#         if img.mode != 'RGB':
#             img = img.convert('RGB')
#         img.save(buffered, format="JPEG")
#         img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
#         return {
#             "type": "image_url", 
#             "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
#         }

#     def process(self, file_bytes: bytes, filename: str) -> PlotData:
#         """
#         Determines file type, prepares inputs, and queries Gemini.
#         """
#         images_to_process = []
#         is_pdf = filename.lower().endswith(".pdf")

#         if is_pdf:
#             # Convert PDF pages to images
#             try:
#                 doc = pymupdf.open(stream=file_bytes, filetype="pdf")
#                 for page_num in range(len(doc)):
#                     if page_num > 0: break # Only process first page for now
#                     page = doc.load_page(page_num)
#                     pix = page.get_pixmap()
#                     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#                     images_to_process.append(img)
#             except Exception as e:
#                 print(f"Error converting PDF: {e}")
#                 raise ValueError("Failed to process PDF file.")
#         else:
#             # Standard Image
#             try:
#                 img = Image.open(io.BytesIO(file_bytes))
#                 images_to_process.append(img)
#             except Exception as e:
#                 print(f"Error opening image: {e}")
#                 raise ValueError("Invalid image file.")

#         if not images_to_process:
#             raise ValueError("No valid content found to process.")

#         # Prepare images for LangChain (List of dicts)
#         formatted_images = [self._pil_to_base64_content(img) for img in images_to_process]

#         # Query Gemini
#         # It returns a PlotData Pydantic object directly now due to .with_structured_output()
#         try:
#             plot_data = self.ai.query_document(formatted_images)
            
#             # Post-processing if needed (like standardizing Garage Swing)
#             if plot_data.garage_swing:
#                 swing = plot_data.garage_swing.upper()
#                 if swing.startswith("R"): plot_data.garage_swing = "Right"
#                 elif swing.startswith("L"): plot_data.garage_swing = "Left"
#                 elif swing.startswith("S"): plot_data.garage_swing = "Straight"
            
#             return plot_data

#         except Exception as e:
#             print(f"Validation Error or API Error: {e}")
#             # Fallback invalid return
#             return PlotData(optional_notes=f"Processing failed: {str(e)}")


import io
import base64
from PIL import Image
import pymupdf
from app.domain.schemas import DocumentExtraction, PlotData


class DocumentExtractor:
    def __init__(self):
        from app.infrastructure.gemini_service import GeminiService
        self.ai = GeminiService()
        # from app.infrastructure.openai_service import OpenAIService
        # self.ai = OpenAIService()
        # from app.infrastructure.huggingface_service import HuggingFaceService
        # self.ai = HuggingFaceService()

    def _pil_to_base64_content(self, img: Image.Image):
        buffered = io.BytesIO()

        if img.mode != "RGB":
            img = img.convert("RGB")

        img.save(buffered, format="JPEG", quality=90)

        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
        }

    def process(self, file_bytes: bytes, filename: str) -> DocumentExtraction:
        """
        Determines file type, prepares inputs, and queries Gemini.
        ALWAYS returns DocumentExtraction.
        """

        images_to_process = []
        is_pdf = filename.lower().endswith(".pdf")

        if is_pdf:
            try:
                doc = pymupdf.open(stream=file_bytes, filetype="pdf")

                # For now only first page (you can remove limit later)
                page = doc.load_page(0)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images_to_process.append(img)

            except Exception as e:
                print(f"Error converting PDF: {e}")
                raise ValueError("Failed to process PDF file.")

        else:
            try:
                img = Image.open(io.BytesIO(file_bytes))
                images_to_process.append(img)

            except Exception as e:
                print(f"Error opening image: {e}")
                raise ValueError("Invalid image file.")

        if not images_to_process:
            raise ValueError("No valid content found to process.")

        formatted_images = [
            self._pil_to_base64_content(img) for img in images_to_process
        ]

        try:
            extraction: DocumentExtraction = self.ai.query_document(formatted_images)

            # -------- NORMALIZE PER-LOT FIELDS SAFELY --------
            for plot in extraction.plots:
                if plot.garage_swing:
                    swing = plot.garage_swing.strip().upper()
                    if swing.startswith("R"):
                        plot.garage_swing = "Right"
                    elif swing.startswith("L"):
                        plot.garage_swing = "Left"
                    elif swing.startswith("S"):
                        plot.garage_swing = "Straight"

                if plot.lot_no:
                    plot.lot_no = plot.lot_no.strip()

            return extraction

        except Exception as e:
            print(f"Gemini extraction error: {e}")

            # Return valid empty structure — NEVER PlotData
            return DocumentExtraction(
                global_elevation=None,
                global_notes=None,
                plots=[
                    PlotData(optional_notes=f"Processing failed: {str(e)}")
                ]
            )
