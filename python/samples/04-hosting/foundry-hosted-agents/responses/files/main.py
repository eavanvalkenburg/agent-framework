# Copyright (c) Microsoft. All rights reserved.

"""Read only explicitly uploaded files from this Foundry hosted session."""

import asyncio
import os
import stat
from pathlib import Path

from agent_framework import Agent, tool
from agent_framework.foundry import FoundryChatClient, FoundryToolbox
from agent_framework_foundry_hosting import ResponsesHostServer
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()


def _files_root() -> Path:
    """Use the session's dedicated upload folder, not the container working directory."""
    return Path.home() / "sample_files"


def _open_files_root() -> tuple[int, int]:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_only = getattr(os, "O_DIRECTORY", None)
    if not isinstance(no_follow, int) or not isinstance(directory_only, int):
        raise RuntimeError("Secure session-file reading requires directory and no-follow open support.")
    return os.open(_files_root(), os.O_RDONLY | directory_only | no_follow), no_follow


@tool(description="List files in the current session's sample_files directory.", approval_mode="never_require")
def list_files() -> list[str]:
    """List only regular files in the sample's scoped upload directory."""
    if not _files_root().exists():
        return []
    directory, _ = _open_files_root()
    try:
        with os.scandir(directory) as entries:
            return sorted(entry.name for entry in entries if entry.is_file(follow_symlinks=False))
    finally:
        os.close(directory)


@tool(description="Read one named file from this session's sample_files directory.", approval_mode="never_require")
def read_file(filename: str) -> str:
    """Reject directory traversal, symlinks, and unexpectedly large files."""
    if not filename or filename in (".", "..") or Path(filename).name != filename:
        raise ValueError("filename must be a single file name in sample_files.")
    directory, no_follow = _open_files_root()
    try:
        descriptor = os.open(filename, os.O_RDONLY | no_follow, dir_fd=directory)
    except OSError as exc:
        raise ValueError("filename must refer to a regular file in sample_files.") from exc
    finally:
        os.close(directory)
    with os.fdopen(descriptor, "rb") as file:
        if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
            raise ValueError("filename must refer to a regular file in sample_files.")
        data = file.read(1_000_001)
    if len(data) > 1_000_000:
        raise ValueError("Only regular UTF-8 files of at most 1 MB can be read.")
    return data.decode("utf-8")


def create_agent() -> Agent:
    """Keep the Toolbox's MCP connection within one request's caller context."""
    credential = DefaultAzureCredential()
    toolbox = FoundryToolbox(credential)
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        credential=credential,
    )
    return Agent(
        client=client,
        instructions=(
            "You are a helpful assistant. Use list_files and read_file only for files in sample_files. "
            "Use the code interpreter for calculations."
        ),
        tools=[list_files, read_file, toolbox],
    )


async def main() -> None:
    server = ResponsesHostServer(agent=create_agent, inner_history="host")
    await server.run_async()


if __name__ == "__main__":
    asyncio.run(main())
