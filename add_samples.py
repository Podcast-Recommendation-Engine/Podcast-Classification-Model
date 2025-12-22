import pandas as pd
import uuid
import random
import re

# --- CONFIGURATION ---
NUM_SAMPLES = 996 
OUTPUT_FILE = r'c:\Users\Ibrahim\Documents\WORK\Faculty-Projects\5eme\Podcast-Classification\data\annotated\podcasts_annotated.csv'

# --- NON-KIDS DATA POOLS ---
adult_pools = [
    {
        "category": "Adult Comedy/Roast",
        "templates": ["{host}'s Late Night Roast", "Unfiltered: {topic}", "The {topic} Hangover", "No Filter with {host}"],
        "topic": ["Dating Disasters", "Office Grinds", "Bad Decisions", "Modern Life", "Social Media Fails"],
        "host": ["Dave", "The Crew", "Sarah J.", "The Anonymous Comic"],
        "base_keywords": ["comedy", "unfiltered", "mature", "humor", "sarcasm", "stories", "laugh"],
        "kid_friendly": 0
    },
    {
        "category": "Intense True Crime",
        "templates": ["Autopsy of {victim}", "The {crime} Serial", "Cold Case: {victim}", "Evidence of {crime}"],
        "victim": ["The Stranger", "The Night Stalker", "Room 402", "The Silent Witness"],
        "crime": ["Murder", "Heist", "Abduction", "Conspiracy", "Betrayal"],
        "base_keywords": ["crime", "mystery", "blood", "violence", "investigation", "forensics", "dark"],
        "kid_friendly": 0
    },
    {
        "category": "Politics/Current Affairs",
        "templates": ["The {region} Conflict", "{topic} Debate 2025", "Inside the {institution}", "The {topic} Crisis"],
        "region": ["Middle East", "Balkans", "Global South", "European Union"],
        "topic": ["Geopolitics", "Economic Collapse", "Tax Reform", "Military Spending", "Election Fraud"],
        "institution": ["Pentagon", "Central Bank", "Senate", "White House"],
        "base_keywords": ["politics", "government", "policy", "war", "economy", "debate", "analysis"],
        "kid_friendly": 0
    },
    {
        "category": "Mature Health/Psychology",
        "templates": ["Dealing with {issue}", "The {issue} Addiction", "Therapy Sessions: {issue}", "Inside {issue}"],
        "issue": ["Depression", "Trauma", "Toxic Relationships", "Substance Abuse", "Chronic Stress"],
        "base_keywords": ["health", "psychology", "mental health", "therapy", "adult", "wellness", "struggle"],
        "kid_friendly": 0
    }
]

generated_data = []

for _ in range(NUM_SAMPLES):
    pool = random.choice(adult_pools)
    template = random.choice(pool["templates"])
    
    # Extract placeholders like {topic} or {host}
    keys_needed = re.findall(r'\{(.*?)\}', template)
    format_dict = {key: random.choice(pool[key]) for key in keys_needed}
    
    title = template.format(**format_dict)
    
    # Keyword generation logic
    raw_keywords = list(set(pool["base_keywords"] + title.lower().replace(":", "").split()))
    clean_keywords = [w for w in raw_keywords if len(w) > 2]
    random.shuffle(clean_keywords)
    
    generated_data.append({
        "id": str(uuid.uuid4()),
        "title": title,
        "keywords": str(clean_keywords),
        "n_keywords": len(clean_keywords),
        "keywords_clean": str(clean_keywords).lower(),
        "is_kid_friendly": pool["kid_friendly"]
    })

# --- APPEND TO CSV ---
df_adult = pd.DataFrame(generated_data)
df_adult.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

print(f"Success! Added {NUM_SAMPLES} non-kid-friendly samples to your dataset.")