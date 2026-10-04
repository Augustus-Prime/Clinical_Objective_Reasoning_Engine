import json
import sys
from collections import defaultdict
import matplotlib.pyplot as plt


def process_taxonomy_rollups(
    input_json_data, 
    output_image_path="understanding_scores_distribution.png", 
    output_json_path="taxonomy_rollup_summary.json",
    output_report_path="taxonomy_rollup_report.txt"
):
    """
    Processes event data containing curriculum_mapping to create rollup summaries, visual charts, 
    and a readable plain English text report.
    
    Args:
        input_json_data (dict or str): Dict object or file path to JSON file.
        output_image_path (str): File path to save the generated bar graph.
        output_json_path (str): File path to save the structured JSON summary.
        output_report_path (str): File path to save the plain English summary report.
    """
    # 1. Load Data with Graceful Error Handling
    if isinstance(input_json_data, str):
        try:
            with open(input_json_data, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f" Error: File '{input_json_data}' not found in current directory.")
            return None
        except json.JSONDecodeError as e:
            print(f" Error: File '{input_json_data}' is empty or contains invalid JSON ({e}).")
            return None
    else:
        data = input_json_data

    events = data.get("events", [])

    # ==========================================
    # (1) Bar Graph: understanding_score distribution
    # ==========================================
    score_counts = defaultdict(int)
    
    for event in events:
        score = event.get("analysis", {}).get("understanding_score")
        if score is not None:
            score_counts[score] += 1

    # Plotting the chart
    fig, ax = plt.subplots(figsize=(8, 5))
    scores_sorted = sorted(score_counts.keys())
    counts = [score_counts[s] for s in scores_sorted]

    bars = ax.bar(scores_sorted, counts, color="#3498db", edgecolor="#2980b9", width=0.5)
    ax.set_title("Distribution of Understanding Scores Across All Events", fontsize=14, fontweight="bold")
    ax.set_xlabel("Understanding Score", fontsize=12)
    ax.set_ylabel("Count of Events", fontsize=12)
    ax.set_xticks(scores_sorted)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"Bar graph saved to: '{output_image_path}'")

    # ==========================================
    # Group Events by Topic (composite_numeric_id + topic_id + dimension)
    # ==========================================
    topic_groups = defaultdict(list)
    
    for event in events:
        cm = event.get("curriculum_mapping") or {}
        composite_id = cm.get("composite_numeric_id") or "UNMAPPED"
        topic_id = cm.get("topic_id") or "UNMAPPED"
        dimension = cm.get("dimension") or "UNMAPPED"
        
        topic_key = (composite_id, topic_id, dimension)
        topic_groups[topic_key].append(event)

    # ==========================================
    # (2) Master List of Topics
    # ==========================================
    master_topic_list = []

    for (comp_id, top_id, dim), topic_events in topic_groups.items():
        scores = [
            e.get("analysis", {}).get("understanding_score")
            for e in topic_events
            if e.get("analysis", {}).get("understanding_score") is not None
        ]
        avg_score = round(sum(scores) / len(scores), 2) if scores else None
        
        distinct_event_ids = set(e.get("event_id") for e in topic_events if e.get("event_id"))
        
        distinct_speakers = set()
        for e in topic_events:
            primary = e.get("primary_speaker")
            if primary:
                distinct_speakers.add(primary)
            for responder in e.get("responders", []):
                if responder:
                    distinct_speakers.add(responder)

        master_topic_list.append({
            "composite_numeric_id": comp_id,
            "topic_id": top_id,
            "dimension": dim,
            "average_understanding_score": avg_score,
            "distinct_events_count": len(distinct_event_ids),
            "distinct_speakers_count": len(distinct_speakers),
            "speakers_involved": sorted(list(distinct_speakers)),
            "events": topic_events
        })

    # ==========================================
    # (3) Top 5 Topics with Lowest Average Understanding Score
    # ==========================================
    mapped_topics = [t for t in master_topic_list if t["composite_numeric_id"] != "UNMAPPED"]
    ranking_pool = mapped_topics if mapped_topics else master_topic_list

    sorted_topics = sorted(
        ranking_pool,
        key=lambda x: (x["average_understanding_score"] if x["average_understanding_score"] is not None else float("inf"))
    )
    
    lowest_5_topics_raw = sorted_topics[:5]
    lowest_5_topics = []

    for topic in lowest_5_topics_raw:
        misconceptions = [
            e.get("analysis", {}).get("key_misconception")
            for e in topic["events"]
            if e.get("analysis", {}).get("key_misconception") is not None
            and str(e.get("analysis", {}).get("key_misconception")).strip() != ""
        ]
        
        lowest_5_topics.append({
            "composite_numeric_id": topic["composite_numeric_id"],
            "topic_id": topic["topic_id"],
            "dimension": topic["dimension"],
            "average_understanding_score": topic["average_understanding_score"],
            "key_misconceptions": misconceptions,
            "events": topic["events"]
        })

    # ==========================================
    # (4) Misconceptions - Topics with non-null key_misconception events
    # ==========================================
    topics_with_misconceptions = []

    for topic in master_topic_list:
        misconception_events = [
            e for e in topic["events"]
            if e.get("analysis", {}).get("key_misconception") is not None
            and str(e.get("analysis", {}).get("key_misconception")).strip() != ""
        ]
        
        if misconception_events:
            topics_with_misconceptions.append({
                "composite_numeric_id": topic["composite_numeric_id"],
                "topic_id": topic["topic_id"],
                "dimension": topic["dimension"],
                "misconception_events_count": len(misconception_events),
                "events_with_misconceptions": misconception_events
            })
    # Extract Top 5 Most Misunderstandings without losing the full list above
    top_5_misconception_topics = sorted(
        topics_with_misconceptions,
        key=lambda x: x["misconception_events_count"],
        reverse=True
    )[:5]

    # ==========================================
    # (5) List of misconceptions - All non-null key_misconception fields
    # ==========================================
    all_key_misconceptions = []

    for event in events:
        km = event.get("analysis", {}).get("key_misconception")
        if km is not None and str(km).strip() != "":
            cm = event.get("curriculum_mapping") or {}
            all_key_misconceptions.append({
                "event_id": event.get("event_id"),
                "composite_numeric_id": cm.get("composite_numeric_id") or "UNMAPPED",
                "topic_id": cm.get("topic_id") or "UNMAPPED",
                "dimension": cm.get("dimension") or "UNMAPPED",
                "key_misconception": km
            })

    # Final summary container (JSON)
    final_summary = {
        "summary_metadata": {
            "total_events_processed": len(events),
            "total_topics_identified": len(master_topic_list),
            "total_misconceptions_found": len(all_key_misconceptions)
        },
        "master_topic_list": master_topic_list,
        "lowest_5_understanding_score_topics": lowest_5_topics,
        "top_5_misconception_topics": top_5_misconception_topics,  # Top 5 slice for easy UI toggle
        "topics_with_misconceptions": topics_with_misconceptions,  # Full metadata preserved
        "all_key_misconceptions_list": all_key_misconceptions
    }

    # Save summary JSON
    with open(output_json_path, "w") as f:
        json.dump(final_summary, f, indent=2)

    print(f"Summary JSON saved to: '{output_json_path}'")

    # ==========================================
    # (6) Plain English Text Report Generation
    # ==========================================
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("        TAXONOMY ROLLUP & ANALYSIS REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")

    # 1. Executive Summary
    report_lines.append("1. EXECUTIVE SUMMARY")
    report_lines.append("--------------------")
    report_lines.append(f"• Total Events Processed : {len(events)}")
    report_lines.append(f"• Total Topics Identified: {len(master_topic_list)}")
    report_lines.append(f"• Total Misconceptions   : {len(all_key_misconceptions)}")
    report_lines.append("")

    # 2. Score Distribution
    report_lines.append("2. UNDERSTANDING SCORE DISTRIBUTION")
    report_lines.append("-----------------------------------")
    if score_counts:
        for score in scores_sorted:
            report_lines.append(f"• Score {score}: {score_counts[score]} event(s)")
    else:
        report_lines.append("• No understanding scores recorded.")
    report_lines.append("")

    # 3. Lowest 5 Topics
    report_lines.append("3. TOP 5 TOPICS WITH LOWEST UNDERSTANDING SCORES")
    report_lines.append("------------------------------------------------")
    if lowest_5_topics:
        for idx, topic in enumerate(lowest_5_topics, 1):
            avg_str = f"{topic['average_understanding_score']}" if topic['average_understanding_score'] is not None else "N/A"
            report_lines.append(f"{idx}. Topic: {topic['topic_id']}")
            report_lines.append(f"   - Composite ID : {topic['composite_numeric_id']}")
            report_lines.append(f"   - Dimension    : {topic['dimension']}")
            report_lines.append(f"   - Avg Score    : {avg_str}")
            if topic['key_misconceptions']:
                report_lines.append("   - Misconceptions Recorded:")
                for m in topic['key_misconceptions']:
                    report_lines.append(f"     * \"{m}\"")
            else:
                report_lines.append("   - Misconceptions Recorded: None")
            report_lines.append("")
    else:
        report_lines.append("No topic data available.\n")

    # 4. Topics with Misconceptions
    report_lines.append("4. TOPICS WITH MISCONCEPTIONS")
    report_lines.append("-----------------------------")
    if topics_with_misconceptions:
        for t in topics_with_misconceptions:
            report_lines.append(f"• Topic: {t['topic_id']} (ID: {t['composite_numeric_id']} | Dim: {t['dimension']})")
            report_lines.append(f"  - Misconception Events Count: {t['misconception_events_count']}")
        report_lines.append("")
    else:
        report_lines.append("No misconceptions recorded across topics.\n")

    # 5. All Key Misconceptions
    report_lines.append("5. DETAILED LIST OF ALL KEY MISCONCEPTIONS")
    report_lines.append("-------------------------------------------")
    if all_key_misconceptions:
        for item in all_key_misconceptions:
            report_lines.append(f"• Event ID: {item['event_id']}")
            report_lines.append(f"  Topic: {item['topic_id']} (ID: {item['composite_numeric_id']} | Dim: {item['dimension']})")
            report_lines.append(f"  Misconception: \"{item['key_misconception']}\"")
            report_lines.append("")
    else:
        report_lines.append("None identified.\n")

    # 6. Master Topic List Summary
    report_lines.append("6. MASTER TOPIC SUMMARY LIST")
    report_lines.append("----------------------------")
    for topic in master_topic_list:
        avg_str = f"{topic['average_understanding_score']}" if topic['average_understanding_score'] is not None else "N/A"
        speakers_str = ", ".join(topic['speakers_involved']) if topic['speakers_involved'] else "None listed"
        report_lines.append(f"• Topic: {topic['topic_id']}")
        report_lines.append(f"  - Composite ID    : {topic['composite_numeric_id']}")
        report_lines.append(f"  - Dimension       : {topic['dimension']}")
        report_lines.append(f"  - Average Score   : {avg_str}")
        report_lines.append(f"  - Distinct Events : {topic['distinct_events_count']}")
        report_lines.append(f"  - Speakers ({topic['distinct_speakers_count']})   : {speakers_str}")
        report_lines.append("")

    # Save Report File
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Plain English report saved to: '{output_report_path}'")

    return final_summary


if __name__ == "__main__":
    # Option 2: Command line argument check
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "sample_endo_case.json"
        print(f"No input file specified. Defaulting to: '{input_file}'")

    print(f"Processing input file: '{input_file}'...\n")
    process_taxonomy_rollups(input_file)