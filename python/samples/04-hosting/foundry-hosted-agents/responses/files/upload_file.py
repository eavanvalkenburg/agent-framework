# Copyright (c) Microsoft. All rights reserved.

"""Upload one small UTF-8 sample file to the chosen Foundry hosted session."""

import argparse
import os
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a file to this agent's session-scoped sample_files folder.")
    parser.add_argument("session_id", help="Foundry agent_session_id, not a MAF AgentSession.session_id")
    parser.add_argument("file", type=Path, help="Local UTF-8 file to upload")
    args = parser.parse_args()

    source: Path = args.file
    if not source.is_file() or source.stat().st_size > 1_000_000:
        raise ValueError("The source must be a regular file of at most 1 MB.")
    with (
        AzureCliCredential() as credential,
        AIProjectClient(
            endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
            credential=credential,
            allow_preview=True,
        ) as project,
    ):
        uploaded = project.agents.upload_session_file(
            agent_name=os.environ["FOUNDRY_AGENT_NAME"],
            session_id=args.session_id,
            content=source.read_bytes(),
            path=f"sample_files/{source.name}",
        )
        print(f"Uploaded {uploaded.path} to Foundry session {args.session_id}.")


if __name__ == "__main__":
    main()
