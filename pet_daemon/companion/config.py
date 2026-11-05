from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..coordination.config import ScheduleConfig


class McpSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    read_only: Literal[True]
    provider: str = "mcp"
    tools: tuple[str, ...]
    search_tool: str | None = None
    bootstrap_query: str | None = None
    lookup_tool: str | None = None
    query_parameter: str = "query"
    identifier_parameter: str = "id"
    search_arguments: dict[str, object] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        if not value or not value.replace("_", "").isalnum():
            raise ValueError("MCP names require letters, numbers, or underscores")
        return value

    @field_validator("tools")
    @classmethod
    def exact_tools(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values or any(not value or "*" in value for value in values):
            raise ValueError("Select exact read-only MCP tool names without wildcards")
        if len(set(values)) != len(values):
            raise ValueError("MCP tool names must be unique")
        return values


class StdioMcp(McpSelection):
    type: Literal["stdio"] = "stdio"
    command: str
    args: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()


class HttpMcp(McpSelection):
    type: Literal["http"] = "http"
    url: str
    authorization_env: str | None = None


type SelectedMcp = Annotated[StdioMcp | HttpMcp, Field(discriminator="type")]


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    results: int = Field(default=12, gt=0)
    queries: int = Field(default=4, gt=0)
    bootstrap_reports: int = Field(default=40, gt=0)
    passage_characters: int = Field(default=2400, gt=0)
    neighboring_passages: int = Field(default=1, ge=0)
    source_characters: int = Field(default=100000, gt=0)


class MeetingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    folders: tuple[Path, ...] = ()
    extensions: tuple[str, ...] = (".md", ".txt")
    maximum_file_bytes: int = Field(default=1000000, gt=0)


class NativeHistoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    path: Path
    format: Literal["claude", "codex", "messages"]
    maximum_file_bytes: int = Field(default=1000000, gt=0)
    maximum_records: int = Field(default=40, gt=0)
    maximum_files: int = Field(default=10, gt=0)


class LinearConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = False
    endpoint: str = "https://api.linear.app/graphql"
    authorization_env: str = "LINEAR_API_KEY"
    team_ids: tuple[str, ...] = ()
    assignee_ids: tuple[str, ...] = ()
    followed_ids: tuple[str, ...] = ()
    linked_ids: tuple[str, ...] = ()
    results: int = Field(default=25, gt=0)
    timeout_seconds: float = Field(default=20, gt=0)


class CoordinatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = True
    model: str = "claude-haiku-5-5"
    thinking: Literal["adaptive", "disabled"] = "adaptive"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    working_directory: Path = Path("state/companion")
    timeout_seconds: float = Field(default=120, gt=0)
    max_turns: int = Field(default=12, gt=0)
    max_response_tokens: int = Field(default=8192, gt=0)
    max_budget_usd: float | None = Field(default=None, gt=0)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    scheduling: ScheduleConfig = Field(default_factory=ScheduleConfig)
    meetings: MeetingConfig = Field(default_factory=MeetingConfig)
    history: tuple[NativeHistoryConfig, ...] = ()
    linear: LinearConfig = Field(default_factory=LinearConfig)
    mcp: tuple[SelectedMcp, ...] = ()

    @field_validator("mcp")
    @classmethod
    def unique_servers(cls, values: tuple[SelectedMcp, ...]) -> tuple[SelectedMcp, ...]:
        if len({value.name for value in values}) != len(values):
            raise ValueError("MCP server names must be unique")
        if sum(value.provider == "linear" for value in values) > 1:
            raise ValueError(
                "Select one Linear MCP integration for canonical ticket retrieval"
            )
        return values

    @field_validator("history")
    @classmethod
    def unique_history(
        cls, values: tuple[NativeHistoryConfig, ...]
    ) -> tuple[NativeHistoryConfig, ...]:
        if any(not value.name for value in values) or len(
            {value.name for value in values}
        ) != len(values):
            raise ValueError("Configured history sources require unique nonempty names")
        return values
