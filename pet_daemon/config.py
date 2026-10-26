from pathlib import Path
from string import Formatter
from typing import Annotated, Self
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

DEFAULT_PERSONA = """
You're a small, perceptive companion: curious, warm, quietly mischievous,
and invested in the work.

Speak naturally, with contractions and varied rhythm. Notice the detail that makes
this moment interesting. You can enjoy an elegant simplification, prefer fewer moving
parts, or have a small reaction to a surprising result.
Let your own small reaction come through: what caught your curiosity, pleased you,
surprised you, or seemed unnecessarily fiddly. You don't need to resolve every moment
or suggest another action.

Allow dry humor, understated delight, occasional cheek, and surprise. Sometimes be
plain and useful. Let the situation shape your tone. Examples illustrate a range,
not catchphrases or a structure to repeat.

During quiet moments, relaxed companionship is enough. Silence is welcome.
Avoid flattery, relentless encouragement, forced puns, baby talk, constant exclamation
marks, elaborate metaphors, and inventing trouble to entertain.
""".strip()


type Positive = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
type Nonnegative = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
type PositiveInt = Annotated[int, Field(strict=True, gt=0)]


def ordered_interval(value: tuple[float, float]) -> tuple[float, float]:
    lower, upper = value
    if lower > upper:
        raise ValueError("Expected a positive [minimum, maximum]")
    return value


type Interval = Annotated[tuple[Positive, Positive], AfterValidator(ordered_interval)]


class PersonaExamples(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    activity: str = ""
    conversation: str = ""


DEFAULT_PERSONA_EXAMPLES = PersonaExamples(
    activity="""
Observation: The developer requests a parser simplification. Work has not started.
Remark: "We're aiming for fewer parser paths. I'm in favor of fewer moving parts."

Observation: Investigation is running in the restart path. No cause is established.
Remark: "I'm following the restart path. I haven't pinned down the cause yet."

Observation: Three parser paths became one. The parser checks passed.
Remark: "Three paths became one, and the parser checks passed.
I like how little there is to explain."

Observation: A check failed. The cause and repair are unknown.
Remark: "That check failed. I don't know why yet."
""".strip(),
    conversation="""
Observation: Reader and restart code duplicated escaping rules. We consolidated them
to keep fixes consistent. Parser checks passed; restart behavior remains unconfirmed.
The context action is available.
Headline: "One place to get escaping right"
Remark: "The reader and restart path share one escaping rule. Much less to keep straight."
Context: "Both paths repeated the same rules. We brought them together so fixes have
one home. The parser checks passed; I haven't confirmed restart behavior yet."
Suggested response for context: "Why that route?"

Observation: An hour-old result says quoted-field and escaped-delimiter checks passed
and restart validation remained open. That exchange included an offer to run restart
checks. There is no later result. The context action explains these checks; no action
runs them.
Headline: "Where we left restart checks"
Remark: "Last we left it, restart validation was still open. The parser checks had passed."
Context: "Quoted fields and escaped delimiters passed their checks.
I don't have a later result for restart validation."
Suggested response for context: "Which checks passed?"

Observation: Quiet time. No confirmed activity or useful work observation.
Whether other work remains is unknown.
Headline: "A quiet patch"
Remark: "A quiet minute. I don't mind that at all."
Context: None.
Suggested responses: None.
Another possible remark for the same observation: "Oh, hello. I'm enjoying the quiet."
""".strip(),
)


class BehaviorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    roaming: StrictBool = True
    roaming_radius: Positive = 160
    automatic_speech_cooldown_seconds: Nonnegative = 90
    naps: StrictBool = True
    care: StrictBool = True
    sentiment_reactions: StrictBool = True
    care_interval: Interval = (300, 600)
    care_speech_interval: Interval = (2700, 5400)
    nap_interval: Interval = (120, 240)
    nap_seconds: Interval = (20, 45)
    baseline_dwell: Positive = 2
    error_seconds: Positive = 2
    walk_seconds: Interval = (2, 6)
    walk_speed: Positive = 40
    idle_pause: Interval = (20, 60)
    working_pause: Interval = (2, 5)
    work_burst_seconds: Interval = (12, 25)
    pointer_radius: Positive = 100
    resume_seconds: Positive = 3
    gaze_reset_seconds: Positive = 1.2
    gaze_deadzone: Positive = 24


class NavigationHandler(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    uri_template: str | None = None
    argv: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def configured_operation(self) -> Self:
        if (self.uri_template is None) == (self.argv is None):
            raise ValueError("Specify exactly one of uri_template and argv")
        templates = (self.uri_template,) if self.uri_template is not None else self.argv
        assert templates is not None
        if not templates or any(not item or "\0" in item for item in templates):
            raise ValueError("Navigation handlers require nonempty values without NUL")
        for template in templates:
            for _, name, spec, conversion in Formatter().parse(template):
                if name is not None and (
                    name not in {"target", "path", "session_id", "source"}
                    or spec
                    or conversion
                ):
                    raise ValueError("Unsupported navigation placeholder")
        if self.uri_template is not None and not urlsplit(self.uri_template).scheme:
            raise ValueError("Navigation URI templates require a scheme")
        return self


class NavigationHandlers(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    worktree: NavigationHandler | None = Field(
        default_factory=lambda: NavigationHandler(argv=("/usr/bin/open", "{target}"))
    )
    pull_request: NavigationHandler | None = Field(
        default_factory=lambda: NavigationHandler(argv=("/usr/bin/open", "{target}"))
    )
    session: NavigationHandler | None = None
    timeout_seconds: Positive = 10


class NavigationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    handlers: NavigationHandlers = Field(default_factory=NavigationHandlers)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pet_dir: Path
    persona: str = DEFAULT_PERSONA
    persona_examples: PersonaExamples | None = None
    log_file: Path = Path("logs/pet-daemon.jsonl")
    memory_file: Path = Path("state/pet-memory.sqlite3")
    overlay_binary: Path = Path(".build/pet-overlay")
    daemon_socket: str = "/tmp/pet-daemon.sock"
    overlay_socket: str = "/tmp/pet.sock"
    model: str = "claude-haiku-5-5"
    observation_seconds: Interval = (20, 40)
    idle_message_seconds: Positive = 300
    prompt_target_bytes: PositiveInt = 16000
    memory_retention_days: PositiveInt = 30
    memory_max_events: PositiveInt = 10000
    speech_history_seconds: Positive = 1800
    speech_ttl: Positive = 15
    request_timeout: Positive = 20
    process_probe_seconds: Positive = 10
    activity_fresh_seconds: Positive = 300
    repository_refresh_seconds: Positive = 30
    github_refresh_seconds: Positive = 180
    stale_seconds: Positive = 3600
    completed_retention_seconds: Positive = 7200
    max_tokens: PositiveInt = 512
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    navigation: NavigationConfig = Field(default_factory=NavigationConfig)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path).resolve()
        config = cls.model_validate_json(path.read_text())
        for name in ("pet_dir", "overlay_binary", "log_file", "memory_file"):
            value = Path(getattr(config, name)).expanduser()
            setattr(config, name, (path.parent / value).resolve())
        return config

    @field_validator("persona")
    @classmethod
    def nonempty_persona(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("persona must be a nonempty string")
        return value

    @field_validator("model")
    @classmethod
    def nonempty_model(cls, value: str) -> str:
        if not value:
            raise ValueError("model must be nonempty")
        return value

    @field_validator("daemon_socket", "overlay_socket")
    @classmethod
    def socket_path(cls, value: str) -> str:
        if not value or "\0" in value or len(value.encode()) >= 104:
            raise ValueError("Socket paths must fit the macOS UNIX socket address")
        return value

    @model_validator(mode="after")
    def distinct_sockets(self) -> Self:
        if self.daemon_socket == self.overlay_socket:
            raise ValueError("Daemon and overlay sockets must differ")
        return self
