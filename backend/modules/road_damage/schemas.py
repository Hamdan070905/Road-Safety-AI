from pydantic import BaseModel, Field

class DamageSettings(BaseModel):
    confidence: float = Field(0.20, ge=0.01, le=0.99)
    iou: float = Field(0.45, ge=0.05, le=0.95)
    sample_interval_s: float = Field(0.5, ge=0.1, le=10.0)
    classes: list[int] | None = None
    imgsz: int = Field(640, ge=320, le=1920)
    preset: str = Field('balanced', pattern='^(fast|balanced|high_quality)$')
    tiled_inference: bool = Field(False)
    per_class_confidence: dict[str, float] | None = None

class DamageReview(BaseModel):
    status: str
    note: str = Field('', max_length=2000)
