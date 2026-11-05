import asyncio
import json
from asyncio import StreamReader

from pydantic import ValidationError

from ..coordination.models import OperationOutcome
from .config import DesktopConfig
from .protocol import MAX_ADAPTER_BYTES, AdapterRequest, AdapterResult


class AdapterError(Exception):
    def __init__(
        self, message: str, outcome: OperationOutcome = OperationOutcome.UNKNOWN
    ) -> None:
        super().__init__(message)
        self.outcome = outcome


class AdapterClient:
    def __init__(self, config: DesktopConfig) -> None:
        self.config = config

    async def request(self, request: AdapterRequest) -> AdapterResult:
        if request.operation.kind not in self.config.capabilities:
            raise AdapterError(
                "The configured adapter does not support this operation",
                OperationOutcome.FAILED,
            )
        payload = json.dumps(request.packet()).encode()
        if len(payload) > MAX_ADAPTER_BYTES:
            raise AdapterError(
                "The adapter request exceeds its protocol limit",
                OperationOutcome.FAILED,
            )
        try:
            process = await asyncio.create_subprocess_exec(
                *self.config.argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as error:
            raise AdapterError(str(error), OperationOutcome.FAILED) from error
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                assert (
                    process.stdin is not None
                    and process.stdout is not None
                    and process.stderr is not None
                )
                process.stdin.write(payload)
                await process.stdin.drain()
                process.stdin.close()
                output, _ = await asyncio.gather(
                    self._read(process.stdout), self._read(process.stderr, discard=True)
                )
                await process.wait()
                if process.returncode != 0:
                    raise AdapterError(
                        f"The adapter exited with status {process.returncode}"
                    )
        except (TimeoutError, BrokenPipeError, ConnectionResetError) as error:
            raise AdapterError(
                "The adapter outcome could not be established"
            ) from error
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        try:
            result = AdapterResult.model_validate_json(output)
        except ValidationError as error:
            raise AdapterError("The adapter returned an invalid result") from error
        if (
            result.task_id != request.task.id
            or result.command_id != request.operation.id
        ):
            raise AdapterError(
                "The adapter returned a different task or command identity"
            )
        if result.session is not None:
            session = result.session
            if session.adapter != self.config.name or not all(
                (
                    session.harness,
                    session.session,
                    session.workspace,
                    session.external_id,
                )
            ):
                raise AdapterError("The adapter returned an invalid native session")
        return result

    @staticmethod
    async def _read(stream: StreamReader, *, discard: bool = False) -> bytes:
        chunks: list[bytes] = []
        size = 0
        while data := await stream.read(8192):
            size += len(data)
            if not discard:
                if size > MAX_ADAPTER_BYTES:
                    raise AdapterError(
                        "The adapter response exceeds its protocol limit"
                    )
                chunks.append(data)
        return b"".join(chunks)
