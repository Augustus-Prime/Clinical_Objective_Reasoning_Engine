import os
import json
from anthropic import Anthropic

def generate_cohort_transcripts(block_name, master_case, logistics_issue, total_tables):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing ANTHROPIC_API_KEY! Please set it in your terminal first.")
    
    client = Anthropic(api_key=api_key)
    os.makedirs("data/synthetic_transcripts", exist_ok=True)
    
    table_dynamics = [
        "One student anchors heavily on an initial piece of evidence and makes a critical diagnostic misstep. The group has to actively debate to correct it.",
        "The group works through the logic, but gets deeply derailed talking about off-topic weekend plans for a few turns before getting back on track.",
        "The students are highly confused by the case logic, leading to circular reasoning, stumbles, hesitations, and a call for the instructor's help.",
        "The team splits into a heated disagreement over two competing diagnoses, arguing back and forth using clinical evidence.",
        "The table struggles significantly with the infrastructure/logistics problem, letting tech frustration completely hijack the first half of their discussion."
    ]
    
    print(f"\n🚀 Launching Cohort Simulation for {total_tables} tables in the [{block_name}] block...")

    for table_num in range(1, total_tables + 1):
        dynamic_profile = table_dynamics[(table_num - 1) % len(table_dynamics)]
        simulated_audio_file = f"Table_{table_num:02d}_{block_name.replace(' ', '_')}_Session.m4a"
        output_filename = f"data/synthetic_transcripts/{block_name}_synthetic_table_{table_num:02d}.json"
        
        print(f"🔄 Simulating Table {table_num}/{total_tables} -> {output_filename}...")
        
        # ADVANCED XML PROMPT (WITH CASE COHORT STABILIZATION)
        master_prompt = f"""
        <system_role>
        You are a clinical education data simulation engine. Your sole task is to generate hyper-realistic, raw, un-sanitized conversational audio transcripts of medical students working in small groups.
        </system_role>

        <behavioral_guidelines>
        1. NO AI PERFECTION: Real human dialogue is messy. Use false starts, interruptions, and vocal fillers ('uh', 'like', 'um', 'wait, hang on').
        2. COGNITIVE STRUGGLE: Students must show real confusion, stuttering logic, or stubbornness when corrected.
        3. INTENTIONAL PRIVACY VIOLATIONS: You must naturally plant real un-anonymized first names or professor mentions (e.g., "Marcus", "Sarah", "Dr. Patel") so our downstream security scripts can test removing them.
        4. COHORT CONSTRAINTS (CRITICAL): Every table in this classroom is reviewing the EXACT same case file. Do not invent a different case or change the patient's presentation. They must all debate the exact case detailed in the current session variables.
        </behavioral_guidelines>

        <gold_standard_few_shot_example>
        {{
            "audio_metadata": {{
                "file_name": "Table_00_Example_Session.m4a",
                "detected_language": "en",
                "total_duration_seconds": 0.0
            }},
            "conversation_analytics": {{
                "total_speakers": 0,
                "speaker_talk_time_seconds": {{}}
            }},
            "segments": [
                {{
                    "id": 0,
                    "start": 1.2,
                    "end": 4.4,
                    "speaker": "SPEAKER_00",
                    "text": "Wait... look at the... uh, the mean corpuscular volume here. It's like 115."
                }},
                {{
                    "id": 1,
                    "start": 4.2,
                    "end": 8.8,
                    "speaker": "SPEAKER_01",
                    "text": "Hold up, Chris, let me see. Yeah, that's macrocytic."
                }}
            ]
        }}
        </gold_standard_few_shot_example>

        <current_session_variables>
        - Academic Module Block: {block_name}
        - THE ABSOLUTE MASTER CASE PARAMETERS (All tables read this identical file): {master_case}
        - Environmental/Logistical Friction: {logistics_issue}
        - Assigned Table Group Dynamic: {dynamic_profile}
        - Current Target Table Number: Table {table_num}
        </current_session_variables>

        <output_instructions>
        Generate a completely original conversation tracking the current session variables. You must output between 14 and 20 conversational segments. 
        Focus heavily on realistic dialogue text. For the metadata and analytics blocks, you can output 0 or dummy values, as a localized Python script will override your math with perfect calculations.
        Respond with ONLY valid, raw JSON. Do not include markdown ticks (```json).
        </output_instructions>
        """

        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=4000,
            temperature=0.85,
            messages=[{"role": "user", "content": master_prompt}]
        )
        
        raw_text = response.content[0].text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
            
        try:
            json_data = json.loads(raw_text)
            
            # ======================================================================
            # 🧮 PRIORITY 1: AUTOMATED PYTHON MATH POST-PROCESSING ENGINE
            # ======================================================================
            segments = json_data.get("segments", [])
            if segments:
                # 1. Calculate absolute total duration based on the final timestamp
                total_duration = max(seg["end"] for seg in segments)
                
                # 2. Programmatically calculate individual speaker talk times
                speaker_times = {}
                for seg in segments:
                    speaker = seg["speaker"]
                    duration = round(seg["end"] - seg["start"], 1)
                    if duration < 0: 
                        duration = 0.0 # Handle accidental inverted timestamps
                    speaker_times[speaker] = round(speaker_times.get(speaker, 0.0) + duration, 1)
                
                # 3. Force overwrite the JSON metadata block with perfect math
                json_data["audio_metadata"]["total_duration_seconds"] = round(total_duration, 1)
                json_data["conversation_analytics"]["total_speakers"] = len(speaker_times)
                json_data["conversation_analytics"]["speaker_talk_time_seconds"] = speaker_times
            # ======================================================================
            
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            print(f"✅ Table {table_num} generated and programmatically math-corrected.")
            
        except json.JSONDecodeError:
            print(f"❌ Structural JSON crash on Table {table_num}.")

if __name__ == "__main__":
    print("📋 --- CLINICAL COHORT GENERATOR (MATH & CASE SYNCHRONIZED) ---")
    
    user_block = input("Enter Course Block (e.g., Neurology): ")
    user_case = input("Enter Specific Case Details (e.g., 58yo male, acute lower back pain, T12-L1 compression on MRI, preserved knee reflexes, no bladder symptoms): ")
    user_logistics = input("Enter Classroom Logistics issue (e.g., none): ")
    user_tables = int(input("How many classroom tables do you want to simulate? (e.g., 3): "))
            
    generate_cohort_transcripts(user_block, user_case, user_logistics, user_tables)