from pydantic import BaseModel, ConfigDict, Field


class ScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    review_seconds: float = Field(default=900, gt=0)
    max_deferral_seconds: float = Field(default=3600, gt=0)
    minimum_interval_seconds: float = Field(default=30, ge=0)
    max_candidates: int = Field(default=6, gt=0)
    coding_slots: int = Field(default=1, gt=0)
    max_passes_per_hour: int = Field(default=12, gt=0)
    source_fresh_seconds: float = Field(default=300, gt=0)
    watch_seconds: float = Field(default=180, gt=0)
    failure_retry_seconds: float = Field(default=120, gt=0)
