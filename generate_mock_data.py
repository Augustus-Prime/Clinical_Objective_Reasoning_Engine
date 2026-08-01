import os
import json
import re
import random
import math
from anthropic import Anthropic

# =====================================================================
# 1. API AGNOSTIC LLM HANDLER
# =====================================================================
class LLMProvider:
    """
    Wrapper class to make LLM calls API-agnostic. 
    Currently uses Anthropic, but you can easily add an OpenAI or Local function here.
    """
    def __init__(self, provider="anthropic"):
        self.provider = provider
        if self.provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("❌ Missing ANTHROPIC_API_KEY!")
            self.client = Anthropic(api_key=self.api_key)

    def generate_text(self, system_prompt, user_prompt):
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model="claude-haiku-4-5", # Or claude-3-haiku-20240307
                max_tokens=4000,
                temperature=0.85,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text.strip()
        # elif self.provider == "openai": ...
        # elif self.provider == "local": ...
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

# =====================================================================
# 2. CHAOS MATRIX
# =====================================================================
CHAOS_PROFILES = [
    "[THE SPIRAL] The group gets obsessed with a minor distractor symptom and spends this entire segment going down the wrong rabbit hole.",
    "[THE LONE WOLF] One student knows the correct answer instantly, but the rest of the table aggressively ignores or doubts them.",
    "[THE DEADLOCK] Two students repeatedly talk past each other, fiercely arguing the same point but using different terminology.",
    "[THE WEEKEND DETOUR] The group loses focus completely and spends 2 minutes talking about a difficult exam or weekend plans before snapping back to the case."
]

# =====================================================================
# 3. GENERATION & ORCHESTRATION
# =====================================================================
def generate_long_transcript(block_name, master_case, table_num, target_minutes, add_chaos, llm):
    """Generates a transcript in 5-minute epochs to prevent context drift and token limits."""
    
    # 5 mins = 1 epoch. (e.g., 15 mins = 3 epochs)
    epochs_needed = math.ceil(target_minutes / 5.0)
    all_raw_lines = []
    recent_context = "No previous context. This is the start of the session."

    print(f"\n🔄 Table {table_num} | Target: {target_minutes} mins | Epochs: {epochs_needed}")

    for epoch in range(1, epochs_needed + 1):
        print(f"   -> Generating Epoch {epoch}/{epochs_needed} (Mins {(epoch-1)*5} to {epoch*5})...")
        
        # Determine behavior for this specific epoch
        current_behavior = "Normal, productive clinical reasoning."
        if add_chaos and random.random() > 0.4:  # 60% chance of chaos IF chaos is enabled
            current_behavior = random.choice(CHAOS_PROFILES)
            print(f"      [!] Injecting Chaos: {current_behavior.split(']')[0]}]")

        system_prompt = """
        You are a clinical education data simulation engine. Your sole task is to generate raw, un-sanitized conversational audio transcripts of medical students in a small group.
        NO AI PERFECTION: Real dialogue is messy. Use false starts, interruptions, and vocal fillers ('uh', 'like', 'um').
        OUTPUT FORMAT: You must output ONLY screenplay dialogue. No JSON. No markdown. Format EXACTLY like this:
        SPEAKER_00: Hey, did anyone look at the labs yet?
        SPEAKER_01: Yeah, I have them pulled up.
        """

        user_prompt = f"""
        - Block: {block_name}
        - Master Case: {master_case}
        
        CURRENT PHASE: Epoch {epoch} of {epochs_needed}.
        PREVIOUS CONTEXT (Last few lines spoken):
        {recent_context}
        
        BEHAVIORAL INSTRUCTION FOR THIS SEGMENT:
        {current_behavior}
        
        TASK:
        Generate roughly 5 minutes of continuous dialogue (about 40-60 back-and-forth turns). 
        Start immediately with the dialogue. Do not include scene directions or timestamps.
        """

        raw_epoch_text = llm.generate_text(system_prompt, user_prompt)
        
        # Clean and store the generated lines
        lines = [line.strip() for line in raw_epoch_text.split('\n') if line.strip().startswith('SPEAKER_')]
        all_raw_lines.extend(lines)

        # Grab the last 4 lines to feed into the next epoch so the conversation connects flawlessly
        if len(lines) > 4:
            recent_context = "\n".join(lines[-4:])
        else:
            recent_context = "\n".join(lines)

    return all_raw_lines

# =====================================================================
# 4. TEXT-TO-JSON PARSER & TIME MATH
# =====================================================================
def compile_to_json(raw_lines, output_filename, words_per_second=2.5):
    """Converts raw text lines into perfect JSON with exact timestamps based on word counts."""
    segments = []
    current_time = 0.0
    speaker_times = {}

    for i, line in enumerate(raw_lines):
        match = re.match(r'^(SPEAKER_\d+):\s*(.*)$', line)
        if match:
            speaker, text = match.groups()
            word_count = len(text.split())
            
            # Calculate duration: Words / Speaking Rate, with a minimum floor of 1.0 second
            duration = max(1.0, round(word_count / words_per_second, 1))
            
            # 10% chance of a slight interruption/overlap (negative pause), otherwise normal pause
            pause = random.choice([-0.2, 0.3, 0.5, 0.8])
            if current_time + pause < 0:
                pause = 0
                
            start_time = round(current_time + pause, 1)
            end_time = round(start_time + duration, 1)

            segments.append({
                "id": i,
                "start": start_time,
                "end": end_time,
                "speaker": speaker,
                "text": text
            })
            
            # Update math trackers
            current_time = end_time
            speaker_times[speaker] = round(speaker_times.get(speaker, 0.0) + duration, 1)

    total_duration = round(current_time, 1)

    # Build the exact JSON schema your downstream engine expects
    json_data = {
        "audio_metadata": {
            "file_name": os.path.basename(output_filename).replace(".json", ".m4a"),
            "detected_language": "en",
            "total_duration_seconds": total_duration
        },
        "conversation_analytics": {
            "total_speakers": len(speaker_times),
            "speaker_talk_time_seconds": speaker_times
        },
        "segments": segments
    }

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
        
    return total_duration, len(segments)

# =====================================================================
# 5. MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("📋 --- ENTERPRISE CLINICAL COHORT GENERATOR ---")
    
    # User Inputs
    user_block = input("Enter Course Block (e.g., Cardiology): ")
    user_case = input("Enter Specific Case Details: ")
    user_tables = int(input("How many classroom tables to simulate? (e.g., 1): "))
    user_minutes = float(input("Target transcript length in minutes? (e.g., 5, 15, 30): "))
    
    # Chaos Matrix Prompt
    chaos_input = input("Would you like to occasionally inject conversational chaos (tangents/arguments)? (y/n): ").lower()
    add_chaos = chaos_input.startswith('y')

    try:
        llm_engine = LLMProvider(provider="anthropic")
    except Exception as e:
        print(e)
        exit(1)

    print(f"\n🚀 Launching Simulation Engine...")
    
    for table_num in range(1, user_tables + 1):
        output_file = f"data/synthetic_transcripts/{user_block.replace(' ', '_')}_Table_{table_num:02d}.json"
        
        # 1. Generate text via epoch chaining
        raw_script_lines = generate_long_transcript(
            block_name=user_block, 
            master_case=user_case, 
            table_num=table_num, 
            target_minutes=user_minutes, 
            add_chaos=add_chaos,
            llm=llm_engine
        )
        
        # 2. Parse text into perfect JSON math
        final_duration, total_segments = compile_to_json(raw_script_lines, output_file)
        
        actual_minutes = round(final_duration / 60.0, 1)
        print(f"✅ Table {table_num} Complete: ~{actual_minutes} minutes generated ({total_segments} segments). Saved to {output_file}")