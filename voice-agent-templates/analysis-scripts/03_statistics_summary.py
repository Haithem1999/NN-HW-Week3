"""
Script 3: Comprehensive Statistics & Thesis-Ready Summary Report
=================================================================
This script consolidates outputs from Script 1 (extraction) and Script 2
(thematic analysis) into a final statistical summary suitable for direct
inclusion in the thesis (Chapter 5: Data Analysis and Findings).

Outputs:
  - Descriptive statistics per shop and aggregate
  - Conversation quality metrics (turns, words, response ratios)
  - Intent distribution tables
  - Tool utilisation analysis
  - Sentiment × appointment cross-tabulations
  - Thematic code frequencies and inter-code correlations
  - Temporal trend analysis (weekly volume, performance over time)
  - Thesis-ready tables in both JSON and CSV

Author: Gasmi Haithem Aissa Khalil
Thesis: Enhancing Operational Efficiency in U.S. Automotive Repair Shops
"""

import json
import os
import csv
import statistics
from collections import Counter, defaultdict
from datetime import datetime

OUTPUT_DIR = "C:/Users/gasmi/Documents/thesis_analysis/output"


def load_structured_data():
    """Load structured call data from Script 1."""
    path = os.path.join(OUTPUT_DIR, "all_calls_structured.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_thematic_data():
    """Load thematic analysis results from Script 2."""
    path = os.path.join(OUTPUT_DIR, "thematic_analysis_results.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(lst):
    return statistics.mean(lst) if lst else 0


def safe_median(lst):
    return statistics.median(lst) if lst else 0


def safe_stdev(lst):
    return statistics.stdev(lst) if len(lst) >= 2 else 0


def pct(part, whole):
    return (part / whole * 100) if whole > 0 else 0


# ============================================================
# SECTION 1: DESCRIPTIVE CALL STATISTICS
# ============================================================

def compute_descriptive_stats(calls):
    """Compute comprehensive descriptive statistics per shop and aggregate."""
    shops = ["Engines Express", "Johns Automotive", "Miramar"]
    report = {}

    for scope in shops + ["Aggregate"]:
        if scope == "Aggregate":
            subset = calls
        else:
            subset = [c for c in calls if c["shop"] == scope]

        n = len(subset)
        with_transcript = [c for c in subset if c["num_turns_total"] > 0]

        # Duration
        durations = [c["duration_sec"] for c in subset if c["duration_sec"] is not None]

        # Success rate
        successful = sum(1 for c in subset if str(c.get("call_successful", "")).lower() == "true")

        # Sentiment
        sentiments = Counter(c.get("user_sentiment", "Unknown") for c in subset)

        # Appointment activity
        appt_true = sum(1 for c in subset if str(c.get("appointment_activity", "")).lower() == "true")
        appt_false = sum(1 for c in subset if str(c.get("appointment_activity", "")).lower() == "false")

        # Disconnection reasons
        disconnections = Counter(c.get("disconnection_reason", "Unknown") for c in subset)

        # Cost
        costs = []
        for c in subset:
            cost_str = str(c.get("cost", "")).strip().replace("$", "")
            if cost_str:
                try:
                    costs.append(float(cost_str))
                except ValueError:
                    pass

        # Latency
        latencies = []
        for c in subset:
            lat_str = str(c.get("latency_ms", "")).strip()
            if lat_str:
                try:
                    latencies.append(int(float(lat_str)))
                except ValueError:
                    pass

        # Date range
        dates = []
        for c in subset:
            ts = c.get("timestamp", "").strip()
            if ts:
                try:
                    dates.append(datetime.strptime(ts, "%m/%d/%Y %H:%M"))
                except ValueError:
                    pass

        report[scope] = {
            "total_calls": n,
            "calls_with_transcript": len(with_transcript),
            "date_range": {
                "start": min(dates).strftime("%Y-%m-%d") if dates else "N/A",
                "end": max(dates).strftime("%Y-%m-%d") if dates else "N/A",
                "span_days": (max(dates) - min(dates)).days if dates else 0,
            },
            "duration": {
                "mean": safe_mean(durations),
                "median": safe_median(durations),
                "stdev": safe_stdev(durations),
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0,
                "n": len(durations),
            },
            "success_rate": pct(successful, n),
            "successful_count": successful,
            "sentiment": dict(sentiments),
            "appointment": {
                "true": appt_true,
                "false": appt_false,
                "rate": pct(appt_true, appt_true + appt_false) if (appt_true + appt_false) > 0 else 0,
            },
            "disconnection": dict(disconnections),
            "cost": {
                "total": sum(costs),
                "mean": safe_mean(costs),
                "n": len(costs),
            },
            "latency": {
                "mean": safe_mean(latencies),
                "median": safe_median(latencies),
                "min": min(latencies) if latencies else 0,
                "max": max(latencies) if latencies else 0,
                "n": len(latencies),
            },
        }

    return report


# ============================================================
# SECTION 2: CONVERSATION QUALITY METRICS
# ============================================================

def compute_conversation_metrics(calls):
    """Compute turn-level and word-level conversation quality metrics."""
    shops = ["Engines Express", "Johns Automotive", "Miramar"]
    report = {}

    for scope in shops + ["Aggregate"]:
        if scope == "Aggregate":
            subset = [c for c in calls if c["num_turns_total"] > 0]
        else:
            subset = [c for c in calls if c["shop"] == scope and c["num_turns_total"] > 0]

        if not subset:
            continue

        total_turns = [c["num_turns_total"] for c in subset]
        agent_turns = [c["num_agent_turns"] for c in subset]
        user_turns = [c["num_user_turns"] for c in subset]
        agent_words = [c["agent_word_count"] for c in subset]
        user_words = [c["user_word_count"] for c in subset]
        total_words = [c["total_word_count"] for c in subset]

        # Words per turn
        agent_wpt = [c["agent_word_count"] / c["num_agent_turns"]
                     for c in subset if c["num_agent_turns"] > 0]
        user_wpt = [c["user_word_count"] / c["num_user_turns"]
                    for c in subset if c["num_user_turns"] > 0]

        # Agent-to-user word ratio
        word_ratios = [c["agent_word_count"] / c["user_word_count"]
                       for c in subset if c["user_word_count"] > 0]

        report[scope] = {
            "n_calls": len(subset),
            "turns": {
                "total_mean": safe_mean(total_turns),
                "total_median": safe_median(total_turns),
                "agent_mean": safe_mean(agent_turns),
                "user_mean": safe_mean(user_turns),
            },
            "words": {
                "total_mean": safe_mean(total_words),
                "agent_mean": safe_mean(agent_words),
                "user_mean": safe_mean(user_words),
                "agent_words_per_turn": safe_mean(agent_wpt),
                "user_words_per_turn": safe_mean(user_wpt),
                "agent_to_user_ratio": safe_mean(word_ratios),
            },
        }

    return report


# ============================================================
# SECTION 3: INTENT DISTRIBUTION
# ============================================================

def compute_intent_distribution(calls):
    """Compute caller intent distribution per shop and aggregate."""
    shops = ["Engines Express", "Johns Automotive", "Miramar"]
    report = {}

    for scope in shops + ["Aggregate"]:
        if scope == "Aggregate":
            subset = [c for c in calls if c["num_turns_total"] > 0]
        else:
            subset = [c for c in calls if c["shop"] == scope and c["num_turns_total"] > 0]

        intent_counts = Counter()
        for c in subset:
            for intent in c.get("detected_intents", []):
                intent_counts[intent] += 1

        n = len(subset)
        report[scope] = {
            intent: {"count": count, "pct": pct(count, n)}
            for intent, count in intent_counts.most_common()
        }

    return report


# ============================================================
# SECTION 4: TOOL UTILISATION
# ============================================================

def compute_tool_utilisation(calls):
    """Analyse tool/function call patterns across shops."""
    shops = ["Engines Express", "Johns Automotive", "Miramar"]
    report = {}

    for scope in shops + ["Aggregate"]:
        if scope == "Aggregate":
            subset = calls
        else:
            subset = [c for c in calls if c["shop"] == scope]

        tool_counts = Counter()
        calls_with_tools = 0
        total_tool_calls = 0

        for c in subset:
            tools = c.get("tools_used", [])
            n_tools = c.get("num_tool_calls", 0)
            if n_tools > 0:
                calls_with_tools += 1
                total_tool_calls += n_tools
            for t in tools:
                if t:
                    tool_counts[t] += 1

        n = len(subset)
        report[scope] = {
            "total_calls": n,
            "calls_with_tool_use": calls_with_tools,
            "tool_use_rate": pct(calls_with_tools, n),
            "total_tool_invocations": total_tool_calls,
            "avg_tools_per_call": total_tool_calls / n if n > 0 else 0,
            "tool_frequency": dict(tool_counts.most_common()),
        }

    return report


# ============================================================
# SECTION 5: CROSS-TABULATIONS
# ============================================================

def compute_cross_tabs(calls):
    """Cross-tabulate key variables for thesis tables."""
    with_tx = [c for c in calls if c["num_turns_total"] > 0]

    # Sentiment × Appointment Activity
    sent_appt = defaultdict(lambda: {"appt_true": 0, "appt_false": 0, "appt_na": 0})
    for c in with_tx:
        sent = c.get("user_sentiment", "Unknown")
        appt = str(c.get("appointment_activity", "")).lower()
        if appt == "true":
            sent_appt[sent]["appt_true"] += 1
        elif appt == "false":
            sent_appt[sent]["appt_false"] += 1
        else:
            sent_appt[sent]["appt_na"] += 1

    # Success × Duration buckets
    dur_success = defaultdict(lambda: {"successful": 0, "unsuccessful": 0})
    for c in with_tx:
        dur = c.get("duration_sec")
        if dur is None:
            continue
        if dur <= 30:
            bucket = "0-30s"
        elif dur <= 60:
            bucket = "31-60s"
        elif dur <= 120:
            bucket = "61-120s"
        elif dur <= 180:
            bucket = "121-180s"
        else:
            bucket = "180s+"
        success = str(c.get("call_successful", "")).lower() == "true"
        if success:
            dur_success[bucket]["successful"] += 1
        else:
            dur_success[bucket]["unsuccessful"] += 1

    # Human request × Sentiment
    human_sent = defaultdict(lambda: {"human_yes": 0, "human_no": 0})
    for c in with_tx:
        sent = c.get("user_sentiment", "Unknown")
        if c.get("human_requested", False):
            human_sent[sent]["human_yes"] += 1
        else:
            human_sent[sent]["human_no"] += 1

    return {
        "sentiment_x_appointment": dict(sent_appt),
        "duration_bucket_x_success": dict(dur_success),
        "sentiment_x_human_request": dict(human_sent),
    }


# ============================================================
# SECTION 6: TEMPORAL ANALYSIS
# ============================================================

def compute_temporal_analysis(calls):
    """Analyse call volume and performance trends over time."""
    shops = ["Engines Express", "Johns Automotive", "Miramar"]
    report = {}

    for scope in shops + ["Aggregate"]:
        if scope == "Aggregate":
            subset = calls
        else:
            subset = [c for c in calls if c["shop"] == scope]

        # Weekly volume
        weekly = defaultdict(int)
        # Hourly distribution
        hourly = defaultdict(int)
        # Day-of-week distribution
        daily = defaultdict(int)
        # Monthly volume
        monthly = defaultdict(int)

        for c in subset:
            ts = c.get("timestamp", "").strip()
            if not ts:
                continue
            try:
                dt = datetime.strptime(ts, "%m/%d/%Y %H:%M")
                yr, wk, _ = dt.isocalendar()
                weekly[f"{yr}-W{wk:02d}"] += 1
                hourly[dt.hour] += 1
                daily[dt.strftime("%A")] += 1
                monthly[dt.strftime("%Y-%m")] += 1
            except ValueError:
                pass

        # Weekly stats
        week_counts = list(weekly.values()) if weekly else []

        report[scope] = {
            "weekly_volume": dict(sorted(weekly.items())),
            "weekly_stats": {
                "mean": safe_mean(week_counts),
                "median": safe_median(week_counts),
                "min": min(week_counts) if week_counts else 0,
                "max": max(week_counts) if week_counts else 0,
            },
            "hourly_distribution": dict(sorted(hourly.items())),
            "day_of_week": {
                d: daily.get(d, 0)
                for d in ["Monday", "Tuesday", "Wednesday", "Thursday",
                          "Friday", "Saturday", "Sunday"]
            },
            "monthly_volume": dict(sorted(monthly.items())),
        }

    return report


# ============================================================
# SECTION 7: THEMATIC ANALYSIS SUMMARY STATISTICS
# ============================================================

def compute_thematic_summary(thematic_results):
    """Compute summary statistics from thematic analysis results."""
    if not thematic_results:
        return {"error": "No thematic results available. Run Script 2 first."}

    shops = ["Engines Express", "Johns Automotive", "Miramar"]
    report = {}

    for scope in shops + ["Aggregate"]:
        if scope == "Aggregate":
            subset = thematic_results
        else:
            subset = [r for r in thematic_results if r["shop"] == scope]

        if not subset:
            continue

        n = len(subset)

        # Trust scores
        trust_scores = [r["trust_score"] for r in subset]
        pos_trust = sum(1 for s in trust_scores if s > 0)
        neg_trust = sum(1 for s in trust_scores if s < 0)
        neu_trust = sum(1 for s in trust_scores if s == 0)

        # Info sharing
        name_shared = sum(1 for r in subset if r["info_shared"]["name_shared"])
        phone_shared = sum(1 for r in subset if r["info_shared"]["phone_shared"])

        # Human requested
        human_req = sum(1 for r in subset if r["human_requested"])

        # Code frequency aggregation
        trust_pos_codes = Counter()
        trust_neg_codes = Counter()
        helpfulness_codes = Counter()
        empathy_codes = Counter()
        inductive_codes = Counter()

        for r in subset:
            trust_pos_codes.update(r.get("trust_positive_codes", {}))
            trust_neg_codes.update(r.get("trust_negative_codes", {}))
            helpfulness_codes.update(r.get("helpfulness_codes", {}))
            empathy_codes.update(r.get("empathy_codes_user", {}))
            inductive_codes.update(r.get("inductive_codes", {}))

        report[scope] = {
            "n_calls_analysed": n,
            "trust": {
                "mean_score": safe_mean(trust_scores),
                "median_score": safe_median(trust_scores),
                "positive_count": pos_trust,
                "positive_pct": pct(pos_trust, n),
                "neutral_count": neu_trust,
                "neutral_pct": pct(neu_trust, n),
                "negative_count": neg_trust,
                "negative_pct": pct(neg_trust, n),
            },
            "info_sharing": {
                "name_shared": name_shared,
                "name_shared_pct": pct(name_shared, n),
                "phone_shared": phone_shared,
                "phone_shared_pct": pct(phone_shared, n),
            },
            "human_request": {
                "count": human_req,
                "pct": pct(human_req, n),
            },
            "code_frequencies": {
                "trust_positive": dict(trust_pos_codes.most_common()),
                "trust_negative": dict(trust_neg_codes.most_common()),
                "helpfulness": dict(helpfulness_codes.most_common()),
                "empathy": dict(empathy_codes.most_common()),
                "inductive": dict(inductive_codes.most_common()),
            },
        }

    return report


# ============================================================
# SECTION 8: THESIS-READY TABLES (CSV export)
# ============================================================

def export_thesis_tables(desc_stats, conv_metrics, intent_dist,
                         tool_util, cross_tabs, temporal, thematic):
    """Export key findings as thesis-ready CSV tables."""

    tables_dir = os.path.join(OUTPUT_DIR, "thesis_tables")
    os.makedirs(tables_dir, exist_ok=True)

    # Table 5.1: Overview of Call Data by Shop
    path = os.path.join(tables_dir, "table_5_1_call_overview.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Engines Express", "Johns Automotive", "Miramar", "Aggregate"])
        metrics = [
            ("Total Calls", lambda s: s["total_calls"]),
            ("Calls with Transcript", lambda s: s["calls_with_transcript"]),
            ("Date Range", lambda s: f"{s['date_range']['start']} to {s['date_range']['end']}"),
            ("Duration Span (days)", lambda s: s["date_range"]["span_days"]),
            ("Mean Duration (sec)", lambda s: f"{s['duration']['mean']:.1f}"),
            ("Median Duration (sec)", lambda s: f"{s['duration']['median']:.1f}"),
            ("SD Duration (sec)", lambda s: f"{s['duration']['stdev']:.1f}"),
            ("Success Rate (%)", lambda s: f"{s['success_rate']:.1f}"),
            ("Appointment Rate (%)", lambda s: f"{s['appointment']['rate']:.1f}"),
            ("Mean Cost ($/call)", lambda s: f"${s['cost']['mean']:.3f}"),
            ("Total Cost ($)", lambda s: f"${s['cost']['total']:.2f}"),
            ("Mean Latency (ms)", lambda s: f"{s['latency']['mean']:.0f}"),
        ]
        for label, fn in metrics:
            row = [label]
            for shop in ["Engines Express", "Johns Automotive", "Miramar", "Aggregate"]:
                try:
                    row.append(fn(desc_stats[shop]))
                except (KeyError, ZeroDivisionError):
                    row.append("N/A")
            w.writerow(row)
    print(f"  Saved: {path}")

    # Table 5.2: Conversation Quality Metrics
    path = os.path.join(tables_dir, "table_5_2_conversation_quality.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Engines Express", "Johns Automotive", "Miramar", "Aggregate"])
        metrics = [
            ("Calls Analysed", lambda s: s["n_calls"]),
            ("Mean Turns/Call", lambda s: f"{s['turns']['total_mean']:.1f}"),
            ("Median Turns/Call", lambda s: f"{s['turns']['total_median']:.1f}"),
            ("Mean Agent Turns", lambda s: f"{s['turns']['agent_mean']:.1f}"),
            ("Mean User Turns", lambda s: f"{s['turns']['user_mean']:.1f}"),
            ("Mean Total Words", lambda s: f"{s['words']['total_mean']:.1f}"),
            ("Mean Agent Words", lambda s: f"{s['words']['agent_mean']:.1f}"),
            ("Mean User Words", lambda s: f"{s['words']['user_mean']:.1f}"),
            ("Agent Words/Turn", lambda s: f"{s['words']['agent_words_per_turn']:.1f}"),
            ("User Words/Turn", lambda s: f"{s['words']['user_words_per_turn']:.1f}"),
            ("Agent:User Word Ratio", lambda s: f"{s['words']['agent_to_user_ratio']:.2f}"),
        ]
        for label, fn in metrics:
            row = [label]
            for shop in ["Engines Express", "Johns Automotive", "Miramar", "Aggregate"]:
                try:
                    row.append(fn(conv_metrics[shop]))
                except (KeyError, ZeroDivisionError):
                    row.append("N/A")
            w.writerow(row)
    print(f"  Saved: {path}")

    # Table 5.3: Caller Intent Distribution
    path = os.path.join(tables_dir, "table_5_3_intent_distribution.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Intent", "EE Count", "EE %", "JA Count", "JA %",
                     "MIR Count", "MIR %", "Total Count", "Total %"])
        # Get all intents from aggregate
        all_intents = list(intent_dist.get("Aggregate", {}).keys())
        for intent in all_intents:
            row = [intent]
            for shop in ["Engines Express", "Johns Automotive", "Miramar", "Aggregate"]:
                data = intent_dist.get(shop, {}).get(intent, {"count": 0, "pct": 0.0})
                row.extend([data["count"], f"{data['pct']:.1f}"])
            w.writerow(row)
    print(f"  Saved: {path}")

    # Table 5.4: Tool Utilisation
    path = os.path.join(tables_dir, "table_5_4_tool_utilisation.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Engines Express", "Johns Automotive", "Miramar", "Aggregate"])
        metrics = [
            ("Calls with Tool Use", lambda s: s["calls_with_tool_use"]),
            ("Tool Use Rate (%)", lambda s: f"{s['tool_use_rate']:.1f}"),
            ("Total Tool Invocations", lambda s: s["total_tool_invocations"]),
            ("Avg Tools/Call", lambda s: f"{s['avg_tools_per_call']:.2f}"),
        ]
        for label, fn in metrics:
            row = [label]
            for shop in ["Engines Express", "Johns Automotive", "Miramar", "Aggregate"]:
                try:
                    row.append(fn(tool_util[shop]))
                except (KeyError, ZeroDivisionError):
                    row.append("N/A")
            w.writerow(row)

        # Add individual tool breakdown
        w.writerow([])
        w.writerow(["Tool Name", "EE", "JA", "MIR", "Total"])
        all_tools = set()
        for shop_data in tool_util.values():
            all_tools.update(shop_data.get("tool_frequency", {}).keys())
        for tool in sorted(all_tools):
            row = [tool]
            for shop in ["Engines Express", "Johns Automotive", "Miramar", "Aggregate"]:
                row.append(tool_util.get(shop, {}).get("tool_frequency", {}).get(tool, 0))
            w.writerow(row)
    print(f"  Saved: {path}")

    # Table 5.5: Sentiment Distribution
    path = os.path.join(tables_dir, "table_5_5_sentiment.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Sentiment", "EE Count", "EE %", "JA Count", "JA %",
                     "MIR Count", "MIR %", "Total Count", "Total %"])
        for sent in ["Positive", "Neutral", "Negative", "Unknown"]:
            row = [sent]
            for shop in ["Engines Express", "Johns Automotive", "Miramar", "Aggregate"]:
                count = desc_stats[shop]["sentiment"].get(sent, 0)
                total = desc_stats[shop]["total_calls"]
                row.extend([count, f"{pct(count, total):.1f}"])
            w.writerow(row)
    print(f"  Saved: {path}")

    # Table 5.6: Sentiment × Appointment Cross-Tab
    path = os.path.join(tables_dir, "table_5_6_sentiment_appointment.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Sentiment", "Appt=True", "Appt=False", "Appt=N/A", "Appt Rate (%)"])
        sa = cross_tabs["sentiment_x_appointment"]
        for sent in ["Positive", "Neutral", "Negative", "Unknown"]:
            if sent in sa:
                d = sa[sent]
                total_appt = d["appt_true"] + d["appt_false"]
                rate = pct(d["appt_true"], total_appt) if total_appt > 0 else 0
                w.writerow([sent, d["appt_true"], d["appt_false"], d["appt_na"], f"{rate:.1f}"])
    print(f"  Saved: {path}")

    # Table 5.7: Duration Bucket × Success
    path = os.path.join(tables_dir, "table_5_7_duration_success.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Duration Bucket", "Successful", "Unsuccessful", "Total", "Success Rate (%)"])
        ds = cross_tabs["duration_bucket_x_success"]
        for bucket in ["0-30s", "31-60s", "61-120s", "121-180s", "180s+"]:
            if bucket in ds:
                d = ds[bucket]
                total = d["successful"] + d["unsuccessful"]
                rate = pct(d["successful"], total)
                w.writerow([bucket, d["successful"], d["unsuccessful"], total, f"{rate:.1f}"])
    print(f"  Saved: {path}")

    # Table 5.8: Thematic Trust Summary
    if thematic and "Aggregate" in thematic:
        path = os.path.join(tables_dir, "table_5_8_trust_summary.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Metric", "Engines Express", "Johns Automotive", "Miramar", "Aggregate"])
            metrics = [
                ("Calls Analysed", lambda s: s["n_calls_analysed"]),
                ("Mean Trust Score", lambda s: f"{s['trust']['mean_score']:.2f}"),
                ("Positive Trust (%)", lambda s: f"{s['trust']['positive_pct']:.1f}"),
                ("Neutral Trust (%)", lambda s: f"{s['trust']['neutral_pct']:.1f}"),
                ("Negative Trust (%)", lambda s: f"{s['trust']['negative_pct']:.1f}"),
                ("Name Shared (%)", lambda s: f"{s['info_sharing']['name_shared_pct']:.1f}"),
                ("Phone Shared (%)", lambda s: f"{s['info_sharing']['phone_shared_pct']:.1f}"),
                ("Human Requested (%)", lambda s: f"{s['human_request']['pct']:.1f}"),
            ]
            for label, fn in metrics:
                row = [label]
                for shop in ["Engines Express", "Johns Automotive", "Miramar", "Aggregate"]:
                    try:
                        row.append(fn(thematic[shop]))
                    except (KeyError, ZeroDivisionError):
                        row.append("N/A")
                w.writerow(row)
        print(f"  Saved: {path}")

    # Table 5.9: Inductive Code Frequencies
    if thematic and "Aggregate" in thematic:
        path = os.path.join(tables_dir, "table_5_9_inductive_codes.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Inductive Code", "EE", "JA", "MIR", "Total"])
            all_codes = list(thematic["Aggregate"]["code_frequencies"]["inductive"].keys())
            for code in all_codes:
                row = [code]
                for shop in ["Engines Express", "Johns Automotive", "Miramar", "Aggregate"]:
                    row.append(thematic.get(shop, {}).get("code_frequencies", {}).get("inductive", {}).get(code, 0))
                w.writerow(row)
        print(f"  Saved: {path}")

    # Table 5.10: Disconnection Reasons
    path = os.path.join(tables_dir, "table_5_10_disconnection.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Reason", "EE Count", "EE %", "JA Count", "JA %",
                     "MIR Count", "MIR %", "Total Count", "Total %"])
        # Get all reasons from aggregate
        all_reasons = list(desc_stats["Aggregate"]["disconnection"].keys())
        for reason in sorted(all_reasons, key=lambda r: desc_stats["Aggregate"]["disconnection"].get(r, 0), reverse=True):
            row = [reason]
            for shop in ["Engines Express", "Johns Automotive", "Miramar", "Aggregate"]:
                count = desc_stats[shop]["disconnection"].get(reason, 0)
                total = desc_stats[shop]["total_calls"]
                row.extend([count, f"{pct(count, total):.1f}"])
            w.writerow(row)
    print(f"  Saved: {path}")

    return tables_dir


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("THESIS DATA ANALYSIS — COMPREHENSIVE STATISTICS REPORT")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    calls = load_structured_data()
    print(f"    Loaded {len(calls)} call records from Script 1")

    thematic_results = None
    try:
        thematic_results = load_thematic_data()
        print(f"    Loaded {len(thematic_results)} thematic analysis results from Script 2")
    except FileNotFoundError:
        print("    WARNING: Thematic analysis results not found. Run Script 2 first.")
        print("    Continuing with quantitative analysis only...")

    # Compute all sections
    print("\n[2] Computing descriptive statistics...")
    desc_stats = compute_descriptive_stats(calls)

    print("[3] Computing conversation quality metrics...")
    conv_metrics = compute_conversation_metrics(calls)

    print("[4] Computing intent distribution...")
    intent_dist = compute_intent_distribution(calls)

    print("[5] Computing tool utilisation analysis...")
    tool_util = compute_tool_utilisation(calls)

    print("[6] Computing cross-tabulations...")
    cross_tabs = compute_cross_tabs(calls)

    print("[7] Computing temporal analysis...")
    temporal = compute_temporal_analysis(calls)

    print("[8] Computing thematic summary statistics...")
    thematic_summary = compute_thematic_summary(thematic_results) if thematic_results else {}

    # ============================================================
    # PRINT REPORT
    # ============================================================

    print(f"\n{'='*70}")
    print("SECTION 1: DESCRIPTIVE STATISTICS")
    print(f"{'='*70}")
    for scope in ["Engines Express", "Johns Automotive", "Miramar", "Aggregate"]:
        s = desc_stats[scope]
        print(f"\n  --- {scope.upper()} ---")
        print(f"  Total calls: {s['total_calls']}")
        print(f"  With transcript: {s['calls_with_transcript']}")
        print(f"  Date range: {s['date_range']['start']} to {s['date_range']['end']} ({s['date_range']['span_days']} days)")
        d = s["duration"]
        print(f"  Duration: Mean={d['mean']:.1f}s, Median={d['median']:.1f}s, SD={d['stdev']:.1f}s (n={d['n']})")
        print(f"  Success rate: {s['success_rate']:.1f}% ({s['successful_count']}/{s['total_calls']})")
        print(f"  Appointment rate: {s['appointment']['rate']:.1f}% ({s['appointment']['true']}/{s['appointment']['true']+s['appointment']['false']})")
        print(f"  Sentiment: {s['sentiment']}")
        print(f"  Cost: ${s['cost']['total']:.2f} total, ${s['cost']['mean']:.3f}/call")
        print(f"  Latency: Mean={s['latency']['mean']:.0f}ms, Median={s['latency']['median']:.0f}ms")

    print(f"\n{'='*70}")
    print("SECTION 2: CONVERSATION QUALITY")
    print(f"{'='*70}")
    for scope in ["Engines Express", "Johns Automotive", "Miramar", "Aggregate"]:
        if scope not in conv_metrics:
            continue
        m = conv_metrics[scope]
        print(f"\n  --- {scope.upper()} ({m['n_calls']} calls) ---")
        print(f"  Turns: Mean={m['turns']['total_mean']:.1f}, Agent={m['turns']['agent_mean']:.1f}, User={m['turns']['user_mean']:.1f}")
        print(f"  Words: Total={m['words']['total_mean']:.1f}, Agent={m['words']['agent_mean']:.1f}, User={m['words']['user_mean']:.1f}")
        print(f"  Words/Turn: Agent={m['words']['agent_words_per_turn']:.1f}, User={m['words']['user_words_per_turn']:.1f}")
        print(f"  Agent:User ratio: {m['words']['agent_to_user_ratio']:.2f}")

    print(f"\n{'='*70}")
    print("SECTION 3: INTENT DISTRIBUTION")
    print(f"{'='*70}")
    for scope in ["Aggregate"]:
        print(f"\n  --- {scope.upper()} ---")
        for intent, data in intent_dist[scope].items():
            print(f"  {intent}: {data['count']} ({data['pct']:.1f}%)")

    print(f"\n{'='*70}")
    print("SECTION 4: TOOL UTILISATION")
    print(f"{'='*70}")
    for scope in ["Engines Express", "Johns Automotive", "Miramar", "Aggregate"]:
        t = tool_util[scope]
        print(f"\n  --- {scope.upper()} ---")
        print(f"  Tool use rate: {t['tool_use_rate']:.1f}% ({t['calls_with_tool_use']}/{t['total_calls']})")
        print(f"  Total invocations: {t['total_tool_invocations']}, Avg/call: {t['avg_tools_per_call']:.2f}")
        for tool, count in t["tool_frequency"].items():
            print(f"    {tool}: {count}")

    print(f"\n{'='*70}")
    print("SECTION 5: CROSS-TABULATIONS")
    print(f"{'='*70}")
    print("\n  Sentiment × Appointment:")
    sa = cross_tabs["sentiment_x_appointment"]
    print(f"  {'Sentiment':<12} {'Appt=T':<10} {'Appt=F':<10} {'N/A':<10} {'Rate'}")
    for sent in ["Positive", "Neutral", "Negative", "Unknown"]:
        if sent in sa:
            d = sa[sent]
            total_a = d["appt_true"] + d["appt_false"]
            rate = pct(d["appt_true"], total_a) if total_a > 0 else 0
            print(f"  {sent:<12} {d['appt_true']:<10} {d['appt_false']:<10} {d['appt_na']:<10} {rate:.1f}%")

    print("\n  Duration Bucket × Success:")
    ds = cross_tabs["duration_bucket_x_success"]
    print(f"  {'Bucket':<12} {'Success':<10} {'Fail':<10} {'Rate'}")
    for bucket in ["0-30s", "31-60s", "61-120s", "121-180s", "180s+"]:
        if bucket in ds:
            d = ds[bucket]
            total = d["successful"] + d["unsuccessful"]
            rate = pct(d["successful"], total)
            print(f"  {bucket:<12} {d['successful']:<10} {d['unsuccessful']:<10} {rate:.1f}%")

    print(f"\n{'='*70}")
    print("SECTION 6: TEMPORAL ANALYSIS")
    print(f"{'='*70}")
    for scope in ["Engines Express", "Johns Automotive", "Miramar", "Aggregate"]:
        t = temporal[scope]
        print(f"\n  --- {scope.upper()} ---")
        print(f"  Weekly: Mean={t['weekly_stats']['mean']:.1f}, Min={t['weekly_stats']['min']}, Max={t['weekly_stats']['max']}")
        print(f"  Day of week: {t['day_of_week']}")
        print(f"  Monthly: {t['monthly_volume']}")
        # Peak hours
        hours = t["hourly_distribution"]
        if hours:
            peak_hour = max(hours, key=hours.get)
            print(f"  Peak hour: {peak_hour}:00 ({hours[peak_hour]} calls)")

    if thematic_summary and "Aggregate" in thematic_summary:
        print(f"\n{'='*70}")
        print("SECTION 7: THEMATIC ANALYSIS SUMMARY")
        print(f"{'='*70}")
        for scope in ["Engines Express", "Johns Automotive", "Miramar", "Aggregate"]:
            if scope not in thematic_summary:
                continue
            t = thematic_summary[scope]
            print(f"\n  --- {scope.upper()} ({t['n_calls_analysed']} calls) ---")
            tr = t["trust"]
            print(f"  Trust: Mean={tr['mean_score']:.2f}, Positive={tr['positive_pct']:.1f}%, Neutral={tr['neutral_pct']:.1f}%, Negative={tr['negative_pct']:.1f}%")
            info = t["info_sharing"]
            print(f"  Info sharing: Name={info['name_shared_pct']:.1f}%, Phone={info['phone_shared_pct']:.1f}%")
            print(f"  Human requested: {t['human_request']['pct']:.1f}%")

            # Top inductive codes
            inductive = t["code_frequencies"]["inductive"]
            if inductive:
                top5 = list(inductive.items())[:5]
                print(f"  Top inductive codes: {', '.join(f'{c}={n}' for c, n in top5)}")

    # ============================================================
    # SAVE FULL REPORT
    # ============================================================

    print(f"\n{'='*70}")
    print("EXPORTING RESULTS...")
    print(f"{'='*70}")

    # Save comprehensive JSON report
    full_report = {
        "generated_at": datetime.now().isoformat(),
        "descriptive_statistics": desc_stats,
        "conversation_metrics": conv_metrics,
        "intent_distribution": intent_dist,
        "tool_utilisation": tool_util,
        "cross_tabulations": cross_tabs,
        "temporal_analysis": temporal,
        "thematic_summary": thematic_summary,
    }

    report_path = os.path.join(OUTPUT_DIR, "comprehensive_statistics_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved comprehensive JSON report: {report_path}")

    # Export thesis-ready tables
    print("\n  Exporting thesis-ready CSV tables:")
    tables_dir = export_thesis_tables(
        desc_stats, conv_metrics, intent_dist,
        tool_util, cross_tabs, temporal, thematic_summary
    )
    print(f"\n  All tables saved to: {tables_dir}")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nPipeline summary:")
    print(f"  Script 1 -> {len(calls)} calls extracted and structured")
    if thematic_results:
        print(f"  Script 2 -> {len(thematic_results)} calls thematically coded")
    print(f"  Script 3 -> Comprehensive report + {10 if thematic_summary else 8} thesis-ready tables")
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
