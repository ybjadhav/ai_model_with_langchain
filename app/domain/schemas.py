from pydantic import BaseModel, Field
from typing import Optional, List

# class PlotData(BaseModel):
#     lot_no: Optional[str] = Field(None, description="The lot number of the property")
#     block: Optional[str] = Field(None, description="The block number of the property")
#     address: Optional[str] = Field(None, description="The full address of the property")
#     model_selected: Optional[str] = Field(None, description="The model name or plan name")
#     elevation: Optional[str] = Field(None, description="The elevation style")
#     garage_swing: Optional[str] = Field(None, description="The garage swing (Left, Right, Straight)")
#     external_structure: Optional[str] = Field(None, description="External structure materials")
#     optional_notes: Optional[str] = Field(None, description="Any other relevant notes")


class PlotData(BaseModel):
    lot_no: Optional[str] = Field(None, description="Lot number (numeric only)")
    block: Optional[str] = Field(None, description="Block number if present")
    address: Optional[str] = Field(None, description="Full property address")
    model_selected: Optional[str] = Field(None, description="Model / Plan name")
    elevation: Optional[str] = Field(None, description="Elevation style (A, B, C, etc.)")
    garage_swing: Optional[str] = Field(None, description="Garage swing: Left, Right, Straight")
    external_structure: Optional[str] = Field(None, description="Exterior materials")
    optional_notes: Optional[str] = Field(None, description="Lot-specific options and notes")


class DocumentExtraction(BaseModel):
    global_elevation: Optional[str] = Field(None, description="Elevation applying to all lots if global")
    global_notes: Optional[str] = Field(None, description="Notes applying to entire document")
    plots: List[PlotData] = Field(..., description="One entry per detected lot block")