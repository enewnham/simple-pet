from pydantic import BaseModel, ConfigDict, Field, field_validator


class Response(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speak: str | None = Field(
        max_length=240,
        description="One optional companionable sentence, at most 30 words. Null means silence.",
    )

    @field_validator("speak")
    @classmethod
    def short_remark(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or len(value.split()) > 30:
            raise ValueError("Care chatter requires one short remark or silence")
        return value
