import json

def get_status_class(score):
    if score < 3.5:
        return "status-review", "review"
    elif score < 4.5:
        return "status-developing", "developing"
    else:
        return "status-solid", "solid"

def format_topic_name(topic_id):
    if topic_id == "UNMAPPED":
        return "Unmapped Topic"
    return " ".join(word.capitalize() for word in topic_id.split('_'))

def generate_topic_cards_html(topic_list, start_idx=1):
    cards_html = ""
    for idx, topic in enumerate(topic_list, start=start_idx):
        topic_id = topic.get("topic_id", "UNMAPPED")
        dimension = topic.get("dimension", "UNMAPPED").upper()
        avg_score = topic.get("average_understanding_score", 0)
        
        status_card, status_text = get_status_class(avg_score)
        topic_name = format_topic_name(topic_id)
        
        misconception_count = 0
        events_html = ""
        for event in topic.get("events", []):
            event_id = event.get("event_id", "")
            speaker = event.get("primary_speaker", "")
            analysis = event.get("analysis", {})
            score = analysis.get("understanding_score", 0)
            misconception = analysis.get("key_misconception")
            
            if misconception:
                misconception_count += 1
                miscon_html = f"""<div class="misconception"><span class="evt-id">{event_id}</span>{misconception}</div>"""
            else:
                miscon_html = ""

            events_html += f"""
            <div class="detail-block">
                <div class="event-row">
                    <div class="event-id">{event_id}</div>
                    <div class="event-score">{score}/5</div>
                    <div class="event-speaker">{speaker}</div>
                </div>
                {miscon_html}
            </div>
            """

        miscon_flag_html = f'<div class="miscon-flag">{misconception_count} Misconception{"s" if misconception_count != 1 else ""}</div>' if misconception_count > 0 else ''
        gauge_percent = (avg_score / 5.0) * 100

        cards_html += f"""
        <div class="topic-card {status_card}" data-name="{topic_name.lower()}" data-dim="{dimension.lower()}" data-score="{avg_score}">
          <div class="topic-row">
            <div class="topic-main">
              <div class="topic-rank">{idx}</div>
              <div class="topic-name">{topic_name}</div>
              <div class="dim-tag">{dimension}</div>
              {miscon_flag_html}
            </div>
            <div class="topic-meta">
              <div class="gauge-wrap">
                <div class="gauge-score {status_text}">{avg_score:.2f}</div>
                <div class="gauge">
                  <div class="gauge-track"></div>
                  <div class="gauge-dot {status_text}" style="left: {gauge_percent}%;"></div>
                </div>
              </div>
              <div class="chev">▼</div>
            </div>
          </div>
          <div class="topic-detail">
            <div class="topic-detail-inner">
              {events_html}
            </div>
          </div>
        </div>
        """
    return cards_html

def generate_html_report(json_input_path="taxonomy_rollup_summary.json", output_path="faculty_report.html"):
    try:
        with open(json_input_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {json_input_path}.")
        return

    summary = data.get("summary", {})
    master_topics = data.get("master_topic_list", [])
    lowest_3_topics = data.get("lowest_3_understanding_score_topics", [])

    # Filter out UNMAPPED topics from our lists
    filtered_master_topics = [t for t in master_topics if t.get("topic_id", "UNMAPPED").upper() != "UNMAPPED"]
    filtered_lowest_3_topics = [t for t in lowest_3_topics if t.get("topic_id", "UNMAPPED").upper() != "UNMAPPED"]

    # Extract unique dimensions for the dropdown filter (using filtered topics)
    unique_dimensions = set()
    for topic in filtered_master_topics:
        dim = topic.get("dimension", "UNMAPPED")
        if dim:
            unique_dimensions.add(dim.upper())
    
    dimension_options_html = ""
    for dim in sorted(list(unique_dimensions)):
        dimension_options_html += f'<option value="{dim.lower()}">{dim}</option>'

    # Vitals Generation
    vitals_html = f"""
      <div class="vital"><div class="num">{summary.get('total_events_processed', 0)}</div><div class="label">TOTAL EVENTS</div></div>
      <div class="vital"><div class="num">{summary.get('total_topics_identified', 0)}</div><div class="label">TOPICS IDENTIFIED</div></div>
      <div class="vital"><div class="num">{summary.get('total_misconceptions_found', 0)}</div><div class="label">MISCONCEPTIONS</div></div>
      <div class="vital"><div class="num">{summary.get('overall_avg_score', 0):.2f}</div><div class="label">AVG UNDERSTANDING</div></div>
    """

    # Score Distribution Graph (using filtered topics)
    score_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for topic in filtered_master_topics:
        for event in topic.get("events", []):
            score = event.get("analysis", {}).get("understanding_score", 0)
            if 1 <= score <= 5:
                score_counts[int(round(score))] += 1

    max_count = max(score_counts.values()) if score_counts.values() else 1
    chart_html = '<div class="chart-container">'
    for score_val in range(1, 6):
        count = score_counts[score_val]
        height_pct = (count / max_count) * 100 if max_count > 0 else 0
        color_class = "bar-low" if score_val <= 2 else ("bar-med" if score_val == 3 else "bar-high")
        
        chart_html += f"""
        <div class="bar-col">
            <div class="bar-count">{count}</div>
            <div class="bar-fill {color_class}" style="height: {height_pct}%;"></div>
            <div class="bar-label">Score {score_val}</div>
        </div>
        """
    chart_html += '</div>'

    # Generate Topic HTML sections (using filtered topics)
    full_topics_html = generate_topic_cards_html(filtered_master_topics, start_idx=1)
    
    if filtered_lowest_3_topics:
        areas_for_review_html = generate_topic_cards_html(filtered_lowest_3_topics, start_idx=1)
        areas_for_review_section = f"""
  <section id="areas-for-review">
    <div class="section-head">
      <span class="section-tag">Action Required</span>
      <h2>Areas for Review</h2>
    </div>
    <p class="section-note">These are the bottom 3 topics showing the lowest average understanding scores. They represent prime candidates for instructional intervention.</p>
    <div id="review-list-items">
      {areas_for_review_html}
    </div>
  </section>
        """
    else:
        areas_for_review_section = ""

    # HTML Shell & CSS
    html_content = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Faculty Report Rollup</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{{
    --cream:#F6F1E7; --paper:#FFFDF8; --ink:#2A2420; --ink-soft:#6B6255;
    --line:#DDD3C0; --rust:#B14C2C; --rust-bg:#F3E1D8; --gold:#96773A;
    --gold-bg:#F0E6D2; --sage:#5E7350; --sage-bg:#E7EBE1;
  }}
  *{{box-sizing:border-box;}}
  body{{ margin:0; background:var(--cream); color:var(--ink); font-family:'Inter',sans-serif; padding:40px 20px 80px; }}
  .sheet{{ max-width:920px; margin:0 auto; background:var(--paper); border:1px solid var(--line); box-shadow:0 1px 3px rgba(42,36,32,0.05), 0 12px 32px -18px rgba(42,36,32,0.18); }}
  .masthead{{ padding:36px 44px 28px; border-bottom:2px solid var(--ink); }}
  .eyebrow{{ font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:0.14em; text-transform:uppercase; color:var(--rust); margin:0 0 10px; }}
  h1{{ font-family:'Fraunces',serif; font-weight:600; font-size:36px; line-height:1.08; margin:0 0 8px; letter-spacing:-0.01em; }}
  .subtitle{{ font-size:15px; color:var(--ink-soft); max-width:60ch; margin:0; }}
  .vitals{{ display:grid; grid-template-columns:repeat(4,1fr); border-bottom:1px solid var(--line); }}
  .vital{{ padding:20px 18px; border-right:1px solid var(--line); }}
  .vital:last-child{{border-right:none;}}
  .vital .num{{ font-family:'Fraunces',serif; font-weight:600; font-size:27px; line-height:1; }}
  .vital .label{{ font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:0.07em; text-transform:uppercase; color:var(--ink-soft); margin-top:6px; }}
  
  /* Chart CSS */
  .chart-container {{ display: flex; align-items: flex-end; gap: 20px; height: 160px; margin: 20px 0 10px; padding-bottom: 0; }}
  .bar-col {{ display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; flex: 1; gap: 8px; }}
  .bar-fill {{ width: 100%; max-width: 60px; border-radius: 4px 4px 0 0; min-height: 2px; }}
  .bar-low {{ background: var(--rust); }}
  .bar-med {{ background: var(--gold); }}
  .bar-high {{ background: var(--sage); }}
  .bar-label {{ font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--ink-soft); border-top:1px solid var(--line); width:100%; text-align:center; padding-top:8px; }}
  .bar-count {{ font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:600; color:var(--ink); }}
  
  section{{padding:34px 44px;border-bottom:1px solid var(--line);}}
  .section-head{{ display:flex; align-items:baseline; gap:12px; margin-bottom:6px; flex-wrap:wrap; }}
  .section-tag{{ font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:0.1em; color:var(--paper); background:var(--ink); padding:3px 8px; }}
  h2{{ font-family:'Fraunces',serif; font-weight:600; font-size:20px; margin:0; }}
  .section-note{{ font-size:13px; color:var(--ink-soft); margin:8px 0 20px; max-width:64ch; }}
  
  /* Filtering & Search Controls CSS */
  .controls-bar {{ display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }}
  .controls-bar input, .controls-bar select {{ font-family: 'Inter', sans-serif; font-size: 14px; padding: 10px 14px; background: var(--cream); border: 1px solid var(--line); color: var(--ink); border-radius: 2px; outline: none; }}
  .controls-bar input:focus, .controls-bar select:focus {{ border-color: var(--ink-soft); }}
  .controls-bar input {{ flex-grow: 1; min-width: 200px; }}
  .controls-bar select {{ min-width: 160px; appearance: none; padding-right: 32px; background-image: url('data:image/svg+xml;utf8,<svg fill="%232A2420" height="20" viewBox="0 0 24 24" width="20" xmlns="http://www.w3.org/2000/svg"><path d="M7 10l5 5 5-5z"/><path d="M0 0h24v24H0z" fill="none"/></svg>'); background-repeat: no-repeat; background-position-x: calc(100% - 8px); background-position-y: 50%; cursor: pointer; }}

  .topic-card{{ border:1px solid var(--line); background:var(--paper); margin-bottom:10px; }}
  .topic-card.status-review{{border-left:3px solid var(--rust);}}
  .topic-card.status-developing{{border-left:3px solid var(--gold);}}
  .topic-card.status-solid{{border-left:3px solid var(--sage);}}
  .topic-row{{ display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 18px; cursor:pointer; }}
  .topic-row:hover{{ background-color:var(--cream); }}
  .topic-main{{display:flex;align-items:center;gap:12px;flex-wrap:wrap;}}
  .topic-rank{{ font-family:'Fraunces',serif; font-size:19px; color:var(--line); width:20px; flex-shrink:0; }}
  .topic-name{{ font-family:'Fraunces',serif; font-weight:600; font-size:16px; }}
  .dim-tag{{ font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:0.06em; text-transform:uppercase; color:var(--ink-soft); border:1px solid var(--line); padding:2px 7px; white-space:nowrap; }}
  .miscon-flag{{ font-family:'JetBrains Mono',monospace; font-size:10px; letter-spacing:0.04em; color:var(--rust); white-space:nowrap; }}
  .topic-meta{{display:flex;align-items:center;gap:18px;flex-shrink:0;}}
  .gauge-wrap{{display:flex;flex-direction:column;align-items:flex-end;}}
  .gauge-score{{ font-family:'JetBrains Mono',monospace; font-weight:600; font-size:14px; }}
  .gauge-score.review{{color:var(--rust);}}
  .gauge-score.developing{{color:var(--gold);}}
  .gauge-score.solid{{color:var(--sage);}}
  .gauge{{width:100px;height:14px;position:relative;margin-top:4px;}}
  .gauge-track{{ position:absolute;top:6px;left:0;right:0;height:2px;background:var(--line); }}
  .gauge-dot{{ position:absolute;top:2px;width:10px;height:10px;border-radius:50%; transform:translateX(-50%); border:2px solid var(--paper); box-shadow:0 0 0 1px var(--ink); }}
  .gauge-dot.review{{background:var(--rust);}}
  .gauge-dot.developing{{background:var(--gold);}}
  .gauge-dot.solid{{background:var(--sage);}}
  
  .chev{{ font-size:13px; color:var(--ink-soft); flex-shrink:0; transition: transform 0.2s ease; }}
  .topic-card.open .chev{{ transform: rotate(180deg); }}
  .topic-detail{{ border-top:1px solid var(--line); display: none; }}
  .topic-card.open .topic-detail{{ display: block; }}
  
  .topic-detail-inner{{padding:14px 18px 18px;}}
  .detail-block{{margin-bottom:14px;}}
  .event-row{{ display:flex; align-items:baseline; gap:10px; font-size:13px; padding:6px 0; border-bottom:1px dotted var(--line); }}
  .event-id{{ font-family:'JetBrains Mono',monospace; font-size:10.5px; color:var(--ink-soft); width:70px; flex-shrink:0; }}
  .event-score{{ font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:600; width:36px; flex-shrink:0; }}
  .event-speaker{{ font-family:'JetBrains Mono',monospace; font-size:10.5px; color:var(--ink-soft); width:78px; flex-shrink:0; }}
  .misconception{{ font-size:13.5px; line-height:1.55; padding:10px 12px; background:var(--rust-bg); border-left:2px solid var(--rust); margin-bottom:8px; margin-top:8px; }}
  .misconception .evt-id{{ font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--rust); display:block; margin-bottom:4px; letter-spacing:0.04em; }}
</style></head><body>

<div class="sheet">
  <div class="masthead">
    <p class="eyebrow">Small Group Case Review</p>
    <h1>Thyroid Dysfunction — Small Group</h1>
    <p class="subtitle">A rollup of how this group reasoned through a case discussion — where understanding held, and where it needs a second pass.</p>
  </div>
  
  <div class="vitals">
    {vitals_html}
  </div>

  <section id="distribution">
    <div class="section-head">
      <span class="section-tag">Analytics</span>
      <h2>Understanding Score Distribution</h2>
    </div>
    <p class="section-note">Distribution of understanding scores across all mapped events (Scores 1 through 5).</p>
    {chart_html}
  </section>

  {areas_for_review_section}

  <section id="problem-list">
    <div class="section-head">
      <span class="section-tag">Problem List</span>
      <h2>Full Topic Summary</h2>
    </div>
    <p class="section-note">Every topic the discussion touched, matched against the Endocrine System / Thyroid disorders taxonomy. Search, filter, or sort — click any topic to see its events, misconceptions, and speakers.</p>
    
    <!-- Filtering & Sort Controls -->
    <div class="controls-bar">
      <input type="text" id="search-input" placeholder="Search topics...">
      <select id="filter-select">
        <option value="all">All dimensions</option>
        {dimension_options_html}
      </select>
      <select id="sort-select">
        <option value="score-asc">Score: lowest first</option>
        <option value="score-desc">Score: highest first</option>
        <option value="name-asc">Name: A-Z</option>
      </select>
    </div>

    <div id="problem-list-items">
      {full_topics_html}
    </div>
  </section>
</div>

<!-- Interactivity & Search/Sort Script -->
<script>
  // Accordion Toggle functionality
  document.querySelectorAll('.topic-row').forEach(function(row) {{
    row.addEventListener('click', function() {{
      const card = this.closest('.topic-card');
      card.classList.toggle('open');
    }});
  }});

  // Search, Filter, and Sort functionality
  const searchInput = document.getElementById('search-input');
  const filterSelect = document.getElementById('filter-select');
  const sortSelect = document.getElementById('sort-select');
  const problemList = document.getElementById('problem-list-items');

  function updateList() {{
    const query = searchInput.value.toLowerCase();
    const dimFilter = filterSelect.value.toLowerCase();
    const sortVal = sortSelect.value;
    
    // Convert NodeList to Array for sorting
    let cards = Array.from(problemList.querySelectorAll('.topic-card'));
    
    // 1. Filter Display
    cards.forEach(card => {{
      const name = card.getAttribute('data-name');
      const dim = card.getAttribute('data-dim');
      
      const matchesSearch = name.includes(query);
      const matchesDim = (dimFilter === 'all') || (dim === dimFilter);
      
      if (matchesSearch && matchesDim) {{
        card.style.display = 'block';
      }} else {{
        card.style.display = 'none';
      }}
    }});
    
    // 2. Sort Logic
    cards.sort((a, b) => {{
      const scoreA = parseFloat(a.getAttribute('data-score'));
      const scoreB = parseFloat(b.getAttribute('data-score'));
      const nameA = a.getAttribute('data-name');
      const nameB = b.getAttribute('data-name');
      
      if (sortVal === 'score-asc') {{
        return scoreA - scoreB || nameA.localeCompare(nameB);
      }} else if (sortVal === 'score-desc') {{
        return scoreB - scoreA || nameA.localeCompare(nameB);
      }} else if (sortVal === 'name-asc') {{
        return nameA.localeCompare(nameB);
      }}
      return 0;
    }});
    
    // 3. Re-append in sorted order
    cards.forEach(card => problemList.appendChild(card));
  }}

  // Event Listeners for the UI controls
  searchInput.addEventListener('input', updateList);
  filterSelect.addEventListener('change', updateList);
  sortSelect.addEventListener('change', updateList);
  
  // Initialize sorting on page load
  updateList();
</script>

</body></html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Report successfully generated at: {output_path}")

if __name__ == "__main__":
    generate_html_report()