"""CLI module for auditing audio-transcript alignment."""

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import re
import sys
import pandas as pd

from cxas_scrapi.core.conversation_history import ConversationHistory
from cxas_scrapi.utils.eval_utils import EvalUtils, evaluate_expectations, ExpectationStatus
from cxas_scrapi.utils.gcs_utils import GCSUtils
from cxas_scrapi.utils.gemini import GeminiGenerate

from typing import Any, Dict, List

logger = logging.getLogger(__name__)

def reconstruct_trace(raw_turns: List[Dict[str, Any]]) -> List[str]:
    trace = []
    for turn in raw_turns:
        user_text = ""
        agent_parts = []
        
        messages = turn.get("messages", [])
        for msg in messages:
            role = msg.get("role")
            chunks = msg.get("chunks", [])
            
            # Extract text/transcript
            text = " ".join([c.get("text", c.get("transcript", "")) for c in chunks if "text" in c or "transcript" in c]).strip()
            
            if role == "user":
                if text:
                    user_text = text
            elif role in ("root_agent", "agent"):
                if text:
                    agent_parts.append(text)
                for chunk in chunks:
                    if "tool_call" in chunk:
                        tc = chunk["tool_call"]
                        tool_name = tc.get("display_name", tc.get("name", tc.get("tool", "")))
                        agent_parts.append(f"[Tool Call: {tool_name}]")
                    elif "tool_response" in chunk:
                        tr = chunk["tool_response"]
                        tool_name = tr.get("display_name", tr.get("name", tr.get("tool", "")))
                        agent_parts.append(f"[Tool Response: {tool_name}]")
        
        if user_text:
            trace.append(f"User: {user_text}")
        else:
            trace.append("User: <silent>")
            
        if agent_parts:
            trace.append(f"Agent: {' '.join(agent_parts)}")
        else:
            trace.append("Agent: <silent>")
            
    return trace

def populate_audit_parser(subparsers):
    """Adds the 'audit-audio' command to the evals subparsers."""
    parser_audit = subparsers.add_parser(
        "audit-audio",
        help="Audit production calls for audio-transcript mismatch and log to BigQuery.",
    )
    parser_audit.add_argument(
        "--app-name",
        required=True,
        help="The CXAS App ID (projects/.../locations/.../apps/...).",
    )
    
    # Mutually exclusive group for session selection
    group = parser_audit.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--session-id",
        help="The single session/conversation ID to audit.",
    )
    group.add_argument(
        "--lookback-interval",
        help="Relative time filter (e.g. '10m', '1h', '1d') to fetch and audit recent sessions.",
    )

    parser_audit.add_argument(
        "--audio-dir-uri",
        required=True,
        help="GCS URI to the directory containing audio files (e.g., gs://my-bucket/calls/).",
    )
    parser_audit.add_argument(
        "--bq-table",
        required=True,
        help="Target BigQuery table (dataset.table_name) to write results.",
    )
    parser_audit.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use for evaluation. Defaults to gemini-2.5-flash.",
    )
    parser_audit.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of conversations to audit.",
    )
    parser_audit.set_defaults(func=handle_audit)

def handle_audit(args: argparse.Namespace) -> None:
    """Handles the 'evals audit-audio' command."""
    history_client = ConversationHistory(app_name=args.app_name)
    
    # 1. Determine Session IDs to audit
    if args.session_id:
        session_ids = [args.session_id]
    else:
        print(f"Fetching sessions from the last {args.lookback_interval}...")
        try:
            # Filter for LIVE conversations by default for production audits
            conversations = history_client.list_conversations(
                time_filter=args.lookback_interval,
                source_filter="LIVE"
            )
            session_ids = [c.name.split("/")[-1] for c in conversations]
            print(f"Found {len(session_ids)} sessions to audit.")
            if args.limit:
                session_ids = session_ids[:args.limit]
                print(f"Limiting audit to the first {args.limit} sessions.")
        except Exception as e:
            print(f"Failed to list conversations: {e}")
            sys.exit(1)

    if not session_ids:
        print("No sessions found to audit.")
        sys.exit(0)

    # Initialize utility clients
    gcs_utils = GCSUtils(creds=history_client.creds)
    gemini_client = GeminiGenerate(
        project_id=history_client.project_id,
        location=history_client.location or "global",
        credentials=history_client.creds,
    )
    eval_utils = EvalUtils(app_name=args.app_name, creds=history_client.creds)

    expectations = [{
        "title": "Audio Mismatch Audit",
        "expectation": (
            "The spoken audio in the turn must semantically match the text transcript. "
            "Ignore minor differences in wording, formatting (e.g., '1 8 0' vs 'one eight zero'), "
            "filler words, or contractions, as long as the core meaning, intent, and instructions "
            "are identical. Flag as FAILED only if there is a semantic contradiction (e.g., 'required' "
            "vs 'not required'), a change in key information (like dates, identifiers, or service names), "
            "or if critical information is added or omitted in the audio that changes the meaning. "
            "\n\nFormatting Instruction for justification on failure:\n"
            "If the audit FAILS, the justification MUST be formatted as a numbered list of issues. "
            "Each issue must reference the Turn number where it occurred and describe the mismatch clearly. "
            "Example:\n"
            "1. Turn 3: Audio omitted 'please retry'.\n"
            "2. Turn 5: Audio said '123' but text transcript shows '456'."
        ),
        "requires_audio_paths": True
    }]

    rows = []

    # 2. Loop through sessions and run evaluation
    for session_id in session_ids:
        print(f"Auditing session: {session_id}")
        
        # Fetch Transcript
        try:
            conversation = history_client.get_conversation(conversation_id=session_id)
            conv_dict = type(conversation).to_dict(conversation)
        except Exception as e:
            print(f"  Failed to fetch conversation history: {e}")
            continue

        # Reconstruct detailed trace
        trace = reconstruct_trace(conv_dict.get("turns", []))

        if not trace:
            print(f"  No turns found for session {session_id}")
            continue

        # List GCS Files
        try:
            bucket_name, prefix = GCSUtils._parse_gcs_uri(args.audio_dir_uri, require_path=False)
            if prefix and not prefix.endswith("/"):
                prefix += "/"
            prefix = f"{prefix}{session_id}/"
            
            gcs_files = gcs_utils.list_with_prefix(f"gs://{bucket_name}", prefix)
        except Exception as e:
            print(f"  Failed to list GCS files: {e}")
            continue

        # Map GCS files to turn numbers (heuristically)
        audio_paths = {}
        for uri in gcs_files:
            filename = uri.split("/")[-1]
            if "full" in filename.lower() or "session" in filename.lower():
                continue
            prod_match = re.search(r'agent-turn-(\d+)', filename)
            if prod_match:
                turn_num = int(prod_match.group(1)) - 1
                audio_paths[turn_num] = uri
                continue
            
            match = re.search(r'(?:turn_)?(\d+)', filename)
            if match:
                turn_num = int(match.group(1))
                if "agent" in filename.lower() or "bot" in filename.lower() or not ("user" in filename.lower() or "customer" in filename.lower()):
                     audio_paths[turn_num] = uri

        if not audio_paths:
            print(f"  Warning: No agent audio files found in {args.audio_dir_uri}{session_id}/")
            continue

        # Run Gemini Evaluation
        try:
            results = evaluate_expectations(
                gemini_client=gemini_client,
                model_name=args.model,
                trace=trace,
                expectations=expectations,
                audio_paths=audio_paths,
            )
            for res in results:
                passed = res.status == ExpectationStatus.MET
                rows.append({
                    "session_id": session_id,
                    "timestamp": pd.Timestamp.now(tz="UTC"),
                    "expectation": res.expectation,
                    "passed": passed,
                    "explanation": res.justification
                })
                print(f"  Audit Result: {'PASSED' if passed else 'FAILED'}")
        except Exception as e:
            print(f"  Gemini evaluation failed: {e}")
            continue

        # Batch write to BigQuery
        if len(rows) >= 5:
            try:
                df = pd.DataFrame(rows)
                print(f"Writing batch of {len(df)} results to BigQuery...")
                eval_utils.to_bigquery(df, args.bq_table)
                rows.clear()
            except Exception as e:
                print(f"Failed to write batch to BigQuery: {e}")
                rows.clear()

    # 3. Write remaining results to BigQuery
    if rows:
        try:
            df = pd.DataFrame(rows)
            print(f"Writing final batch of {len(df)} results to BigQuery...")
            eval_utils.to_bigquery(df, args.bq_table)
        except Exception as e:
            print(f"Failed to write final batch to BigQuery: {e}")
            sys.exit(1)
    else:
        print("No remaining audit results to write.")
