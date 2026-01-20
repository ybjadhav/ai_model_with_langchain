import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from app.domain.schemas import DocumentExtraction, PlotData

class HuggingFaceService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HuggingFaceService, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        self.api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not self.api_key:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
        
        print(f"--- Configuring Hugging Face Service (ChatHuggingFace) ---")
        
        # Reverting to Instruct model (User set to Embedding model previously which is invalid for chat)
        self.repo_id = "Qwen/Qwen2-VL-7B-Instruct"
        
        # Initialize Endpoint (Remote Inference API)
        self.llm = HuggingFaceEndpoint(
            repo_id=self.repo_id,
            huggingfacehub_api_token=self.api_key,
            task="text-generation",
            temperature=0.1,
            max_new_tokens=4096
        )
        
        # Initialize Chat Interface
        self.chat_model = ChatHuggingFace(llm=self.llm)
        aa = self.chat_model.invoke('what is the capital of india')
        print('0---------------------------------',aa)

    def query_document(self, image_parts):
        """
        image_parts: List of dicts with "type": "image_url" and "image_url": {"url": "data:..."}
        """
        prompt_text = """
            You are an expert real-estate and construction document analysis system.
            Return JSON matching this structure perfectly:
            {
              "global_elevation": "... or null",
              "global_notes": "... or null",
              "plots": [
                {
                  "lot_no": "...",
                  "block": "...",
                  "address": "...",
                  "model_selected": "...",
                  "elevation": "...",
                  "garage_swing": "...",
                  "external_structure": "...",
                  "optional_notes": "..."
                }
              ]
            }
            Strictly output VALID JSON ONLY. No markdown blocks.
            """

        try:
            content = [
                {"type": "text", "text": prompt_text},
            ]
            content.extend(image_parts)
            
            message = HumanMessage(content=content)
            
            # Invoke
            result = self.chat_model.invoke([message])
            
            raw_response = result.content
            
            # Clean response
            if "```json" in raw_response:
                raw_response = raw_response.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_response:
                raw_response = raw_response.split("```")[1].split("```")[0].strip()
                
            return DocumentExtraction.model_validate_json(raw_response)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return DocumentExtraction(
                global_elevation=None, 
                global_notes=None, 
                plots=[PlotData(optional_notes=f"HF Error: {repr(e)}")]
            )
