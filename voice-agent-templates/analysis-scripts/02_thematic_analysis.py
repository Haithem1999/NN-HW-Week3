"""
Script 2: Thematic Analysis of Call Transcripts
=================================================
This script performs systematic thematic coding on the structured
transcript data produced by Script 1. It implements a hybrid
deductive-inductive coding approach:
  - Deductive codes derived from the literature (trust, helpfulness, empathy)
  - Inductive codes emerging from patterns in the data

Author: Gasmi Haithem Aissa Khalil
Thesis: Enhancing Operational Efficiency in U.S. Automotive Repair Shops
"""

import json
import os
import csv
import re
from collections import Counter, defaultdict

OUTPUT_DIR = "C:/Users/gasmi/Documents/thesis_analysis/output"


def load_data():
    """Load structured call data from Script 1 output."""
    path = os.path.join(OUTPUT_DIR, "all_calls_structured.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# CODING FAMILIES (Deductive - from thesis Appendix B codebook)
# ============================================================

TRUST_POSITIVE_CODES = {
    "trust_information_acceptance": [
        "okay", "alright", "got it", "makes sense", "i see",
        "that works", "sounds right", "oh okay"
    ],
    "trust_willingness_to_proceed": [
        "yes please", "go ahead", "sure", "let's do it",
        "yes i'd like to", "yeah", "that would be great",
        "absolutely", "definitely"
    ],
    "trust_personal_info_sharing": [
        # Detected by checking if user provides phone/name after being asked
    ],
    "trust_return_intent": [
        "i'll call back", "call again", "thanks i'll",
        "see you", "talk to you"
    ],
}

TRUST_NEGATIVE_CODES = {
    "distrust_questioning": [
        "are you sure", "is that right", "that doesn't sound right",
        "i don't think so", "that can't be", "really?"
    ],
    "distrust_human_request": [
        "real person", "human", "actual person", "speak to someone",
        "talk to someone", "representative", "manager", "owner",
        "live person", "somebody real", "is anyone there"
    ],
    "distrust_ai_skepticism": [
        "you're a robot", "you're a computer", "are you real",
        "am i talking to a machine", "is this automated",
        "i hate these things", "bot"
    ],
}

HELPFULNESS_CODES = {
    "helpful_info_provided": [
        "thank you for", "thanks for the information",
        "that's helpful", "good to know", "i appreciate",
        "that answers my question"
    ],
    "helpful_problem_resolved": [
        "perfect", "great", "awesome", "that's all i needed",
        "you've been helpful", "that's exactly what i needed"
    ],
    "helpful_appointment_success": [
        "got the text", "received", "booked", "scheduled",
        "see you then", "appointment set"
    ],
    "unhelpful_info_missing": [
        "you don't know", "can't help", "not helpful",
        "that doesn't help", "i need more", "useless"
    ],
    "unhelpful_wrong_info": [
        "that's wrong", "incorrect", "not right", "that's not what",
        "no that's", "you're wrong"
    ],
}

EMPATHY_CODES = {
    "empathy_acknowledged": [
        "i understand", "i'm sorry to hear", "i can see",
        "that must be", "i appreciate your patience"
    ],
    "caller_frustration": [
        "frustrated", "annoying", "ridiculous", "come on",
        "this is", "ugh", "seriously", "are you kidding"
    ],
    "caller_appreciation": [
        "you're so nice", "very kind", "appreciate it",
        "so helpful", "you've been great", "thank you so much",
        "thanks a lot", "wonderful"
    ],
    "caller_surprise_ai": [
        "oh you're", "wait are you", "oh wow", "that's cool",
        "you sound real", "didn't expect", "whoa"
    ],
}

# ============================================================
# INDUCTIVE CODES (emerging from data patterns)
# ============================================================

INDUCTIVE_CODES = {
    "pragmatic_acceptance": [
        "okay well", "alright then", "well anyway",
        "that's fine", "okay so", "no problem"
    ],
    "immediate_gratitude": [
        "oh great someone answered", "glad someone picked up",
        "oh good", "finally", "glad i got through",
        "at least someone answered"
    ],
    "after_hours_awareness": [
        "i know you're closed", "after hours",
        "you guys closed", "are you open", "is anyone there",
        "tomorrow", "in the morning", "when you open"
    ],
    "vehicle_urgency": [
        "broke down", "won't start", "stranded", "stuck",
        "emergency", "need it fixed", "can't drive",
        "overheating", "leaking", "smoke"
    ],
    "price_sensitivity": [
        "how much", "what's the cost", "expensive",
        "affordable", "budget", "price range", "cheapest"
    ],
    "conversational_breakdown": [
        "what?", "i didn't understand", "say that again",
        "can you repeat", "huh?", "what did you say",
        "sorry?", "excuse me?"
    ],
}


def code_turns(turns, code_dict):
    """
    Apply a set of codes to the user turns in a transcript.
    Returns a dict of {code_name: count} for codes that were triggered.
    """
    results = {}
    user_text = " ".join(t["text"].lower() for t in turns if t["role"] == "user")

    for code_name, keywords in code_dict.items():
        count = sum(1 for kw in keywords if kw in user_text)
        if count > 0:
            results[code_name] = count

    return results


def code_agent_turns(turns, code_dict):
    """Apply codes to agent turns (used for empathy detection in agent responses)."""
    results = {}
    agent_text = " ".join(t["text"].lower() for t in turns if t["role"] == "agent")

    for code_name, keywords in code_dict.items():
        count = sum(1 for kw in keywords if kw in agent_text)
        if count > 0:
            results[code_name] = count

    return results


def detect_info_sharing(turns):
    """
    Detect if the user shared personal information (name, phone number).
    This is a trust indicator — willingness to share PII with an AI agent.
    """
    shared = {"name_shared": False, "phone_shared": False}

    for i, turn in enumerate(turns):
        if turn["role"] == "agent":
            text_lower = turn["text"].lower()
            # Check if agent asked for name
            if any(kw in text_lower for kw in ["your name", "first name", "spell your"]):
                # Check if next user turn provides it
                if i + 1 < len(turns) and turns[i + 1]["role"] == "user":
                    next_text = turns[i + 1]["text"]
                    # If it's a short response (likely a name), mark as shared
                    if len(next_text.split()) <= 5 and len(next_text) > 1:
                        shared["name_shared"] = True

            # Check if agent asked for phone
            if any(kw in text_lower for kw in ["phone number", "calling from", "reach you"]):
                if i + 1 < len(turns) and turns[i + 1]["role"] == "user":
                    next_text = turns[i + 1]["text"]
                    # Check for digits
                    digits = re.findall(r'\d', next_text)
                    if len(digits) >= 7 or "yes" in next_text.lower():
                        shared["phone_shared"] = True

    return shared


def analyse_call(call):
    """Perform complete thematic analysis on a single call."""
    turns = call.get("transcript_turns", [])
    if not turns:
        return None

    # Apply all coding families
    trust_pos = code_turns(turns, TRUST_POSITIVE_CODES)
    trust_neg = code_turns(turns, TRUST_NEGATIVE_CODES)
    helpfulness = code_turns(turns, HELPFULNESS_CODES)
    empathy_user = code_turns(turns, EMPATHY_CODES)
    empathy_agent = code_agent_turns(turns, {"empathy_acknowledged": EMPATHY_CODES["empathy_acknowledged"]})
    inductive = code_turns(turns, INDUCTIVE_CODES)

    # Info sharing analysis
    info_shared = detect_info_sharing(turns)

    # Aggregate trust score (simple: positive - negative)
    trust_score = sum(trust_pos.values()) - sum(trust_neg.values())

    # Determine dominant theme
    theme_scores = {
        "trust_positive": sum(trust_pos.values()),
        "trust_negative": sum(trust_neg.values()),
        "helpfulness_positive": sum(v for k, v in helpfulness.items() if "helpful_" in k and "unhelpful" not in k),
        "helpfulness_negative": sum(v for k, v in helpfulness.items() if "unhelpful" in k),
        "empathy_appreciation": empathy_user.get("caller_appreciation", 0),
        "empathy_frustration": empathy_user.get("caller_frustration", 0),
        "pragmatic_acceptance": inductive.get("pragmatic_acceptance", 0),
        "ai_surprise": empathy_user.get("caller_surprise_ai", 0),
    }

    return {
        "call_id": call["call_id"],
        "shop": call["shop"],
        "trust_positive_codes": trust_pos,
        "trust_negative_codes": trust_neg,
        "helpfulness_codes": helpfulness,
        "empathy_codes_user": empathy_user,
        "empathy_codes_agent": empathy_agent,
        "inductive_codes": inductive,
        "info_shared": info_shared,
        "trust_score": trust_score,
        "theme_scores": theme_scores,
        "user_sentiment": call.get("user_sentiment", ""),
        "call_successful": call.get("call_successful", ""),
        "appointment_activity": call.get("appointment_activity", ""),
        "human_requested": call.get("human_requested", False),
    }


def main():
    print("Loading structured call data...")
    calls = load_data()
    print(f"Loaded {len(calls)} calls")

    # Analyse all calls with transcripts
    results = []
    for call in calls:
        if call.get("num_turns_total", 0) > 0:
            analysis = analyse_call(call)
            if analysis:
                results.append(analysis)

    print(f"Analysed {len(results)} calls with transcripts")

    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================

    print(f"\n{'='*70}")
    print("THEMATIC ANALYSIS RESULTS")
    print(f"{'='*70}")

    # 1. Trust Analysis
    print(f"\n--- TRUST INDICATORS ---")
    trust_pos_total = Counter()
    trust_neg_total = Counter()
    for r in results:
        trust_pos_total.update(r["trust_positive_codes"])
        trust_neg_total.update(r["trust_negative_codes"])

    print("Positive trust codes:")
    for code, count in trust_pos_total.most_common():
        print(f"  {code}: {count}")

    print("Negative trust codes:")
    for code, count in trust_neg_total.most_common():
        print(f"  {code}: {count}")

    # Trust score distribution
    trust_scores = [r["trust_score"] for r in results]
    pos_trust = sum(1 for s in trust_scores if s > 0)
    neu_trust = sum(1 for s in trust_scores if s == 0)
    neg_trust = sum(1 for s in trust_scores if s < 0)
    print(f"\nTrust score distribution:")
    print(f"  Positive (>0): {pos_trust} ({pos_trust/len(results)*100:.1f}%)")
    print(f"  Neutral (=0): {neu_trust} ({neu_trust/len(results)*100:.1f}%)")
    print(f"  Negative (<0): {neg_trust} ({neg_trust/len(results)*100:.1f}%)")

    # Info sharing
    name_shared = sum(1 for r in results if r["info_shared"]["name_shared"])
    phone_shared = sum(1 for r in results if r["info_shared"]["phone_shared"])
    print(f"\nPersonal info sharing (trust indicator):")
    print(f"  Name shared: {name_shared}/{len(results)} ({name_shared/len(results)*100:.1f}%)")
    print(f"  Phone shared: {phone_shared}/{len(results)} ({phone_shared/len(results)*100:.1f}%)")

    # Human requested
    human_req = sum(1 for r in results if r["human_requested"])
    print(f"  Human requested: {human_req}/{len(results)} ({human_req/len(results)*100:.1f}%)")

    # 2. Helpfulness Analysis
    print(f"\n--- HELPFULNESS INDICATORS ---")
    help_total = Counter()
    for r in results:
        help_total.update(r["helpfulness_codes"])
    for code, count in help_total.most_common():
        print(f"  {code}: {count}")

    # 3. Empathy Analysis
    print(f"\n--- EMPATHY INDICATORS ---")
    empathy_total = Counter()
    for r in results:
        empathy_total.update(r["empathy_codes_user"])
    for code, count in empathy_total.most_common():
        print(f"  {code}: {count}")

    agent_empathy = Counter()
    for r in results:
        agent_empathy.update(r["empathy_codes_agent"])
    print("Agent empathy expressions:")
    for code, count in agent_empathy.most_common():
        print(f"  {code}: {count}")

    # 4. Inductive Codes
    print(f"\n--- INDUCTIVE (EMERGENT) CODES ---")
    inductive_total = Counter()
    for r in results:
        inductive_total.update(r["inductive_codes"])
    for code, count in inductive_total.most_common():
        print(f"  {code}: {count}")

    # 5. Per-Shop Breakdown
    print(f"\n--- PER-SHOP THEMATIC BREAKDOWN ---")
    for shop in ["Engines Express", "Johns Automotive", "Miramar"]:
        shop_results = [r for r in results if r["shop"] == shop]
        if not shop_results:
            continue

        print(f"\n  {shop} ({len(shop_results)} calls):")

        # Trust
        shop_trust = [r["trust_score"] for r in shop_results]
        pos = sum(1 for s in shop_trust if s > 0)
        neg = sum(1 for s in shop_trust if s < 0)
        print(f"    Trust: Positive={pos}, Negative={neg}")

        # Human request
        hr = sum(1 for r in shop_results if r["human_requested"])
        print(f"    Human requested: {hr}")

        # Info shared
        ns = sum(1 for r in shop_results if r["info_shared"]["name_shared"])
        ps = sum(1 for r in shop_results if r["info_shared"]["phone_shared"])
        print(f"    Name shared: {ns}, Phone shared: {ps}")

        # Top inductive codes
        shop_inductive = Counter()
        for r in shop_results:
            shop_inductive.update(r["inductive_codes"])
        if shop_inductive:
            top3 = shop_inductive.most_common(3)
            print(f"    Top inductive codes: {', '.join(f'{c}={n}' for c, n in top3)}")

    # 6. Cross-tabulation: Sentiment vs Trust
    print(f"\n--- CROSS-TABULATION: Sentiment × Trust ---")
    cross_tab = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})
    for r in results:
        sent = r["user_sentiment"]
        if r["trust_score"] > 0:
            cross_tab[sent]["positive"] += 1
        elif r["trust_score"] < 0:
            cross_tab[sent]["negative"] += 1
        else:
            cross_tab[sent]["neutral"] += 1

    print(f"  {'Sentiment':<12} {'Trust+':<10} {'Trust=':<10} {'Trust-':<10}")
    for sent in ["Positive", "Neutral", "Negative", "Unknown"]:
        if sent in cross_tab:
            ct = cross_tab[sent]
            print(f"  {sent:<12} {ct['positive']:<10} {ct['neutral']:<10} {ct['negative']:<10}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    # Save full thematic analysis
    thematic_path = os.path.join(OUTPUT_DIR, "thematic_analysis_results.json")
    with open(thematic_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved thematic analysis to: {thematic_path}")

    # Save summary CSV
    csv_path = os.path.join(OUTPUT_DIR, "thematic_analysis_summary.csv")
    fields = [
        "call_id", "shop", "user_sentiment", "call_successful",
        "appointment_activity", "trust_score", "human_requested",
        "name_shared", "phone_shared",
        "trust_pos_count", "trust_neg_count",
        "helpfulness_pos_count", "helpfulness_neg_count",
        "caller_appreciation", "caller_frustration", "ai_surprise",
        "pragmatic_acceptance", "after_hours_awareness",
        "vehicle_urgency", "conversational_breakdown"
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {
                "call_id": r["call_id"],
                "shop": r["shop"],
                "user_sentiment": r["user_sentiment"],
                "call_successful": r["call_successful"],
                "appointment_activity": r["appointment_activity"],
                "trust_score": r["trust_score"],
                "human_requested": r["human_requested"],
                "name_shared": r["info_shared"]["name_shared"],
                "phone_shared": r["info_shared"]["phone_shared"],
                "trust_pos_count": sum(r["trust_positive_codes"].values()),
                "trust_neg_count": sum(r["trust_negative_codes"].values()),
                "helpfulness_pos_count": sum(v for k, v in r["helpfulness_codes"].items() if "unhelpful" not in k),
                "helpfulness_neg_count": sum(v for k, v in r["helpfulness_codes"].items() if "unhelpful" in k),
                "caller_appreciation": r["empathy_codes_user"].get("caller_appreciation", 0),
                "caller_frustration": r["empathy_codes_user"].get("caller_frustration", 0),
                "ai_surprise": r["empathy_codes_user"].get("caller_surprise_ai", 0),
                "pragmatic_acceptance": r["inductive_codes"].get("pragmatic_acceptance", 0),
                "after_hours_awareness": r["inductive_codes"].get("after_hours_awareness", 0),
                "vehicle_urgency": r["inductive_codes"].get("vehicle_urgency", 0),
                "conversational_breakdown": r["inductive_codes"].get("conversational_breakdown", 0),
            }
            writer.writerow(row)
    print(f"Saved thematic summary CSV to: {csv_path}")


if __name__ == "__main__":
    main()
