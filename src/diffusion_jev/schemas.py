from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Structured = str | dict[str, JsonValue] | list[JsonValue]


class QuestionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: Structured


class Noul(QuestionBase):
    type: Literal["noul"]
    criteria: dict[Literal["true", "false"], Structured] | None = None


class Choice(QuestionBase):
    type: Literal["choice"]
    criteria: dict[str, Structured | None] = Field(min_length=2, max_length=26)


class Score(QuestionBase):
    type: Literal["score"]
    criteria: list[Structured] = Field(min_length=2, max_length=10)


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    state: Structured
    questions: dict[str, Question] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def bound_size(self):
        if len(self.model_dump_json()) > 100_000:
            raise ValueError("Request content exceeds 100,000 characters")
        return self


Probability = Annotated[float, Field(ge=0, le=1)]


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: Probability


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    legend: dict[str, str]
    probabilities: dict[str, Probability]
    confidence: Probability


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class Metadata(BaseModel):
    latency_ms: float = Field(ge=0)
    probability_source: Literal["masked_position_logits"]
    temperature: float = Field(gt=0)
    passes: Literal[1, 2, 4]
    candidate_logits: dict[str, list[float]]


class EvaluationResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
    meta: Metadata
