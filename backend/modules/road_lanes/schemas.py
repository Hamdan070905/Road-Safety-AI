from pydantic import BaseModel, Field

class LaneSettings(BaseModel):
    opacity: float = Field(.45, ge=0, le=1)
    sample_interval_s: float = Field(1.0, ge=.2, le=10)
    output_width: int = Field(1280, ge=320, le=1280)
