import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from app.domain.schemas import PlotData, DocumentExtraction

load_dotenv()

class GeminiService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiService, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        print(f"--- Configuring Gemini with Key: {self.api_key[:5]}... ---")
        
        # Initialize LangChain Chat Model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=self.api_key
        )
        
        # Configure structured output
        # self.structured_llm = self.llm.with_structured_output(PlotData)
        self.structured_llm = self.llm.with_structured_output(DocumentExtraction)


    def query_document(self, image_parts):
        """
        image_parts: List of base64 encoded image data dicts or PIL Images.
        LangChain handles PIL images in HumanMessage content blocks.
        """
        # Prompt to steer the model
        # prompt_text = (
        #     "Analyze this document. Extract: "
        #     "1. Lot Number "
        #     "2. Block Number "
        #     "3. Full Property Address "
        #     "4. House Model/Plan Name "
        #     "5. Elevation "
        #     "6. Garage Swing (Left, Right, Straight) "
        #     "7. External Structure materials "
        #     "8. Any relevant notes."
        # )

        prompt_text = """
            You are an expert real-estate and construction document analysis system.

These images may include:
- Lot Specific Order Forms
- Spec Request Forms
- Builder Configuration Sheets
- Construction Option Lists
- Scanned PDFs and photos

Documents often contain MULTIPLE LOT SECTIONS in a single page.

Each lot section must be extracted as a SEPARATE structured record.

The document may be rotated, blurry, handwritten, or partially cropped.

Your task is to perform FORENSIC-LEVEL multi-entity extraction.

Follow these rules strictly:

------------------------------------------------------------------

### 1. GLOBAL vs LOT-SPECIFIC DATA

First identify:

GLOBAL fields (apply to all lots unless overridden):
- Project-wide elevation
- Community name
- Builder name
- Global notes

Then identify REPEATING LOT BLOCKS.

A LOT BLOCK is defined by patterns such as:
- "Lot 116 - 1685"
- "Lot 117:"
- "Homesite 118"
- "Lot #119"

Each detected lot block represents ONE independent property.

------------------------------------------------------------------

### 2. LOT BLOCK DETECTION RULES

You MUST:

- Scan the entire page top to bottom
- Identify EVERY lot header
- Split the document into logical LOT SECTIONS
- Extract each lot independently

NEVER merge data across different lots.

If 6 lots appear → output 6 plot objects.

------------------------------------------------------------------

### 3. HIERARCHICAL INHERITANCE

If a field appears globally and not repeated inside a lot block:

- Inherit it into each lot ONLY if clearly global

Example:
"Elevation B" at top of page → applies to all lots

But:

If a lot block overrides it → use the lot value.

------------------------------------------------------------------

### 4. FIELD NORMALIZATION

Normalize strictly:

- LotNumber → numeric only
- BlockNumber → numeric only
- GarageSwing → Left | Right | Straight
- Elevation → Single letter/code only

------------------------------------------------------------------

### 5. LOT-SPECIFIC NOTES HANDLING

For each lot block:

Capture bullet items such as:

- Garage Left / Right
- Dual sinks
- Shower in lieu of tub
- Patio slab
- Vanity upgrades

These MUST go into:

optional_notes (lot-specific only)

------------------------------------------------------------------

### 6. STRICT ANTI-HALLUCINATION RULES

- Do NOT guess missing values
- Do NOT copy values between lots
- Do NOT invent addresses or models
- Only extract what is visible

------------------------------------------------------------------

### 7. REQUIRED OUTPUT FORMAT (STRICT JSON)

Return JSON matching this structure:

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

There must be ONE plots[] entry PER detected lot block.

------------------------------------------------------------------

Extract now.

        """

        content = [{"type": "text", "text": prompt_text}]

        messages = [
            HumanMessage(content=content + image_parts)
        ]

        try:
            # Invoking with structured output returns a Pydantic object directly
            result = self.structured_llm.invoke(messages)
            return result
        except Exception as e:
            print(f"Gemini API Error: {e}")
            raise e
