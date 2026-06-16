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
from cxas_scrapi.utils.eval_utils import EvalUtils, evaluate_expectations
from cxas_scrapi.utils.gcs_utils import GCSUtils
from cxas_scrapi.utils.gemini import GeminiGenerate

logger = logging.getLogger(__name__)

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
        except Exception as e:
            print(f"Failed to list conversations: {e}")
            sys.exit(1)

    if not session_ids:
        print("No sessions found to audit.")
        sys.exit(0)

    # Initialize utility clients
    gcs_utils = GCSUtils(creds=history_client.creds)
    gemini_client = GeminiGenerate(creds=history_client.creds)
    
    expectations = [{
        "title": "Audio Mismatch Audit",
        "expectation": "The spoken audio in the turn must exactly match the text transcript in wording and meaning.",
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
            yaml_transcript = ConversationHistory.conversation_dict_to_yaml(conv_dict)
        except Exception as e:
            print(f"  Failed to fetch conversation history: {e}")
            continue

        # Reconstruct detailed trace
        trace = []
        for turn in yaml_transcript.get("turns", []):
            for role, text in turn.items():
                trace.append(f"{role}: {text}")

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
                rows.append({
                    "session_id": session_id,
                    "timestamp": pd.Timestamp.now(tz="UTC"),
                    "expectation": res.expectation,
                    "passed": res.passed,
                    "explanation": res.explanation
                })
                print(f"  Audit Result: {'PASSED' if res.passed else 'FAILED'}")
        except Exception as e:
            print(f"  Gemini evaluation failed: {e}")
            continue

    # 3. Write all results to BigQuery
    if rows:
        df = pd.DataFrame(rows)
        eval_utils = EvalUtils(app_name=args.app_name, creds=history_client.creds)
        try:
            print(f"Writing {len(df)} results to BigQuery table {args.bq_table}...")
            eval_utils.to_bigquery(df, args.bq_table)
        except Exception as e:
            print(f"Failed to write to BigQuery: {e}")
            sys.exit(1)
    else:
        print("No audit results to write.")
