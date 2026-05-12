"""
Script 1: Extract and Clean Transcripts from CSV Files
=======================================================
This script reads the post-call CSV data for all three shops,
extracts the transcript field, parses it into structured turn-by-turn
format, and outputs clean data for further analysis.

Author: Gasmi Haithem Aissa Khalil
Thesis: Enhancing Operational Efficiency in U.S. Automotive Repair Shops
"""

import csv
import os
import json
import re
from datetime import datetime

DOWNLOADS = "C:/Users/gasmi/Downloads"
OUTPUT_DIR = "C:/Users/gasmi/Documents/thesis_analysis/output"

FILES = {
    "Engines Express": "Engines_Express_agent_post-call_data&transcript.csv",
    "Johns Automotive": "Johns_Automotive_agent_post-call-data&transcript.csv",
    "Miramar": "Miramar_agent_post_call_data&transcripts.csv",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_transcript(raw_transcript):
    """
    Parse a raw transcript string into a list of turns.
    Format: "Agent: ...\nUser: ...\nAgent: ..."
    Returns list of dicts: [{"role": "agent"|"user", "text": "..."}]
    """
    if not raw_transcript or not raw_transcript.strip():
        return []

    turns = []
    current_role = None
    current_text = []

    for line in raw_transcript.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("Agent:"):
            if current_role is not None:
                turns.append({"role": current_role, "text": " ".join(current_text).strip()})
            current_role = "agent"
            current_text = [line[6:].strip()]
        elif line.startswith("User:"):
            if current_role is not None:
                turns.append({"role": current_role, "text": " ".join(current_text).strip()})
            current_role = "user"
            current_text = [line[5:].strip()]
        else:
            # Continuation of previous turn
            current_text.append(line)

    if current_role is not None:
        turns.append({"role": current_role, "text": " ".join(current_text).strip()})

    return turns


def extract_tool_calls(transcript_with_tools):
    """
    Extract tool call events from the 'Transcript With Tool Calls' JSON field.
    Returns list of tool call dicts.
    """
    if not transcript_with_tools or not transcript_with_tools.strip():
        return []

    try:
        entries = json.loads(transcript_with_tools)
    except json.JSONDecodeError:
        return []

    tool_calls = []
    for entry in entries:
        if entry.get("role") == "tool_call_invocation":
            tool_calls.append({
                "tool_name": entry.get("name", ""),
                "arguments": entry.get("arguments", ""),
            })
        elif entry.get("role") == "tool_call_result":
            tool_calls.append({
                "tool_result": entry.get("content", ""),
                "tool_name": entry.get("name", ""),
            })

    return tool_calls


def count_turns(turns):
    """Count agent and user turns."""
    agent_turns = sum(1 for t in turns if t["role"] == "agent")
    user_turns = sum(1 for t in turns if t["role"] == "user")
    return agent_turns, user_turns


def count_words(turns, role=None):
    """Count total words in turns, optionally filtered by role."""
    total = 0
    for t in turns:
        if role is None or t["role"] == role:
            total += len(t["text"].split())
    return total


def detect_ai_disclosure(turns):
    """Check if the agent disclosed its AI/virtual assistant identity."""
    if not turns:
        return False
    first_agent = next((t for t in turns if t["role"] == "agent"), None)
    if first_agent:
        text = first_agent["text"].lower()
        return any(kw in text for kw in ["virtual assistant", "ai", "artificial", "automated"])
    return False


def detect_caller_intents(turns):
    """
    Classify caller intents from user turns using keyword matching.
    Returns a list of detected intents.
    """
    intents = set()
    user_text = " ".join(t["text"].lower() for t in turns if t["role"] == "user")

    intent_keywords = {
        "appointment_booking": ["appointment", "schedule", "book", "come in", "bring in", "set up", "slot", "available"],
        "quote_request": ["quote", "estimate", "how much", "price", "cost", "pricing"],
        "hours_inquiry": ["hours", "open", "close", "what time", "when do you"],
        "location_inquiry": ["where", "located", "address", "directions", "find you"],
        "services_inquiry": ["services", "do you do", "can you", "offer", "work on", "fix", "repair"],
        "towing_request": ["tow", "towing", "broke down", "broken down", "stranded", "stuck"],
        "urgent_request": ["urgent", "emergency", "keys", "locked out", "asap"],
        "existing_customer": ["picking up", "pick up", "left my car", "dropped off", "status", "update on my"],
        "leave_message": ["message", "leave a message", "tell them", "let them know"],
        "warranty_inquiry": ["warranty", "guarantee", "covered"],
        "payment_inquiry": ["payment", "pay", "credit card", "financing", "accept"],
    }

    for intent, keywords in intent_keywords.items():
        if any(kw in user_text for kw in keywords):
            intents.add(intent)

    if not intents:
        intents.add("general_inquiry")

    return list(intents)


def detect_sentiment_markers(turns):
    """
    Detect positive and negative sentiment markers in user turns.
    Returns counts of positive and negative markers.
    """
    positive_markers = [
        "thank", "thanks", "great", "awesome", "perfect", "wonderful", "appreciate",
        "helpful", "good", "nice", "excellent", "amazing", "cool", "sounds good",
        "that works", "oh great", "fantastic"
    ]
    negative_markers = [
        "frustrated", "annoying", "terrible", "horrible", "waste", "useless",
        "ridiculous", "unacceptable", "angry", "upset", "disappointed", "worst",
        "hang up", "real person", "human", "actual person", "speak to someone"
    ]

    user_text = " ".join(t["text"].lower() for t in turns if t["role"] == "user")

    pos_count = sum(1 for m in positive_markers if m in user_text)
    neg_count = sum(1 for m in negative_markers if m in user_text)

    return pos_count, neg_count


def detect_human_request(turns):
    """Check if the caller requested to speak to a human."""
    user_text = " ".join(t["text"].lower() for t in turns if t["role"] == "user")
    human_keywords = [
        "real person", "human", "actual person", "speak to someone",
        "talk to someone", "representative", "manager", "owner",
        "can I talk to", "is anyone there", "live person"
    ]
    return any(kw in user_text for kw in human_keywords)


def main():
    all_calls = []

    for shop, fname in FILES.items():
        path = os.path.join(DOWNLOADS, fname)
        print(f"\nProcessing: {shop}")

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"  Total rows: {len(rows)}")

        for row in rows:
            transcript_raw = row.get("Transcript", "")
            transcript_tools = row.get("Transcript With Tool Calls", "")

            turns = parse_transcript(transcript_raw)
            tool_calls = extract_tool_calls(transcript_tools)
            agent_turns, user_turns = count_turns(turns)
            agent_words = count_words(turns, "agent")
            user_words = count_words(turns, "user")
            intents = detect_caller_intents(turns)
            pos_markers, neg_markers = detect_sentiment_markers(turns)
            human_requested = detect_human_request(turns)
            ai_disclosed = detect_ai_disclosure(turns)

            # Parse duration
            dur_str = row.get("Call Duration", "").strip()
            duration_sec = None
            if dur_str and ":" in dur_str:
                parts = dur_str.split(":")
                if len(parts) == 2:
                    duration_sec = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:
                    duration_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

            # Tool names invoked
            tools_used = list(set(
                tc.get("tool_name", "") for tc in tool_calls
                if tc.get("tool_name") and "tool_result" not in tc
            ))

            call_record = {
                "shop": shop,
                "call_id": row.get("Call ID", ""),
                "timestamp": row.get("Time", ""),
                "duration_sec": duration_sec,
                "call_status": row.get("Call Status", ""),
                "call_successful": row.get("Call Successful", ""),
                "disconnection_reason": row.get("Disconnection Reason", ""),
                "user_sentiment": row.get("User Sentiment", ""),
                "appointment_activity": row.get("appointment_activity", ""),
                "latency_ms": row.get("End to End Latency", ""),
                "cost": row.get("Cost", ""),
                "agent_name": row.get("Agent Name", ""),
                # Transcript analysis fields
                "num_turns_total": agent_turns + user_turns,
                "num_agent_turns": agent_turns,
                "num_user_turns": user_turns,
                "agent_word_count": agent_words,
                "user_word_count": user_words,
                "total_word_count": agent_words + user_words,
                "detected_intents": intents,
                "tools_used": tools_used,
                "num_tool_calls": len([tc for tc in tool_calls if "tool_name" in tc and "tool_result" not in tc]),
                "positive_sentiment_markers": pos_markers,
                "negative_sentiment_markers": neg_markers,
                "human_requested": human_requested,
                "ai_disclosure_detected": ai_disclosed,
                "transcript_turns": turns,
            }

            all_calls.append(call_record)

    # Save structured data
    output_path = os.path.join(OUTPUT_DIR, "all_calls_structured.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_calls, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_calls)} call records to: {output_path}")

    # Save summary CSV (without full transcripts)
    csv_path = os.path.join(OUTPUT_DIR, "all_calls_summary.csv")
    csv_fields = [
        "shop", "call_id", "timestamp", "duration_sec", "call_successful",
        "disconnection_reason", "user_sentiment", "appointment_activity",
        "latency_ms", "cost", "num_turns_total", "num_agent_turns",
        "num_user_turns", "agent_word_count", "user_word_count",
        "total_word_count", "detected_intents", "tools_used",
        "num_tool_calls", "positive_sentiment_markers",
        "negative_sentiment_markers", "human_requested",
        "ai_disclosure_detected"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for call in all_calls:
            row = {k: call[k] for k in csv_fields}
            row["detected_intents"] = "; ".join(call["detected_intents"])
            row["tools_used"] = "; ".join(call["tools_used"])
            writer.writerow(row)
    print(f"Saved summary CSV to: {csv_path}")

    # Print quick stats
    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total calls extracted: {len(all_calls)}")
    has_transcript = sum(1 for c in all_calls if c["num_turns_total"] > 0)
    print(f"Calls with transcripts: {has_transcript}")
    print(f"Calls without transcripts: {len(all_calls) - has_transcript}")

    for shop in FILES.keys():
        shop_calls = [c for c in all_calls if c["shop"] == shop]
        shop_with_tx = [c for c in shop_calls if c["num_turns_total"] > 0]
        print(f"\n  {shop}:")
        print(f"    Total: {len(shop_calls)}, With transcript: {len(shop_with_tx)}")
        if shop_with_tx:
            avg_turns = sum(c["num_turns_total"] for c in shop_with_tx) / len(shop_with_tx)
            avg_words = sum(c["total_word_count"] for c in shop_with_tx) / len(shop_with_tx)
            print(f"    Avg turns/call: {avg_turns:.1f}")
            print(f"    Avg words/call: {avg_words:.1f}")


if __name__ == "__main__":
    main()
