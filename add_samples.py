import pandas as pd
import uuid
import random

# --- CONFIGURATION ---
NUM_SAMPLES = 996  # Adding to your existing 4 to reach 1000
OUTPUT_FILE = r'c:\Users\Ibrahim\Documents\WORK\Faculty-Projects\5eme\Podcast-Classification\data\annotated\podcasts_annotated.csv'

# --- DATA POOLS ---
data_pools = [
    {
        "category": "Science/Nature",
        "templates": ["The Secret Life of {item}", "Deep Dive: {item}", "Understanding {item}", "{item} Explained"],
        "item": ["Black Holes", "Ant Colonies", "Quantum Physics", "Global Warming", "The Human Brain", "Honeybees", "Mars Rover"],
        "base_keywords": ["science", "nature", "discovery", "education", "research", "facts"],
        "kid_friendly": 1
    },
    {
        "category": "True Crime",
        "templates": ["Murder in {item}", "The {item} Mystery", "Case File: {item}", "Shadows of {item}"],
        "item": ["The Silent Woods", "Downstairs", "The Midnight Alley", "Suburbia", "The Broken Window"],
        "base_keywords": ["crime", "mystery", "murder", "investigation", "police", "forensics", "suspense"],
        "kid_friendly": 0
    },
    {
        "category": "Bedtime/Kids",
        "templates": ["The {adj} {animal}'s Adventure", "Stories for {item}", "The Tale of the {adj} {animal}"],
        "adj": ["Sparkly", "Grumpy", "Magical", "Tiny", "Golden", "Sleepy"],
        "animal": ["Dragon", "Bunny", "Owl", "Penguin", "Cat", "Squirrel"],
        "item": ["Bedtime", "Little Ears", "Dreamers", "Quiet Time"],
        "base_keywords": ["kids", "bedtime", "story", "imagination", "fairy tale", "family", "sleep"],
        "kid_friendly": 1
    },
    {
        "category": "Politics/Finance",
        "templates": ["{item} Report", "The Future of {item}", "Inside {item}", "The {item} Crisis"],
        "item": ["Wall Street", "The Senate", "Global Trade", "Crypto", "Inflation", "Tax Reform", "The Election"],
        "base_keywords": ["politics", "economy", "money", "government", "policy", "finance", "debate"],
        "kid_friendly": 0
    },
    {
        "category": "Tech/AI",
        "templates": ["Coding {item}", "The {item} Revolution", "Is {item} taking over?", "Mastering {item}"],
        "item": ["Python", "Artificial Intelligence", "Robotics", "Web3", "SaaS", "Cybersecurity", "The Metaverse"],
        "base_keywords": ["tech", "software", "future", "innovation", "computers", "code", "digital"],
        "kid_friendly": 1
    },
    {
        "category": "Sports",
        "templates": ["{item} Weekly", "The Art of {item}", "Behind the {item}", "History of {item}"],
        "item": ["The NBA", "Formula 1", "Tennis", "World Cup Soccer", "Olympic Gold", "The Superbowl"],
        "base_keywords": ["sports", "athlete", "game", "competition", "team", "score", "fitness"],
        "kid_friendly": 1
    }
]

# --- GENERATION LOGIC ---
generated_data = []

for _ in range(NUM_SAMPLES):
    pool = random.choice(data_pools)
    template = random.choice(pool["templates"])
    
    # Fill template slots
    title_data = {}
    for key in ["item", "adj", "animal"]:
        if key in pool:
            title_data[key] = random.choice(pool[key])
            
    title = template.format(**title_data)
    
    # Keyword construction
    # Mix base keywords + words from the generated title
    keywords_list = list(set(pool["base_keywords"] + title.lower().split()))
    # Remove short words like 'the', 'of', 'in'
    keywords_list = [w for w in keywords_list if len(w) > 2]
    random.shuffle(keywords_list)
    
    generated_data.append({
        "id": str(uuid.uuid4()),
        "title": title,
        "keywords": str(keywords_list),
        "n_keywords": len(keywords_list),
        "keywords_clean": str(keywords_list).lower(),
        "is_kid_friendly": pool["kid_friendly"]
    })

# --- SAVE ---
df_new = pd.DataFrame(generated_data)
df_new.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

print(f"Dataset expanded! Added {NUM_SAMPLES} samples to reach 1,000.")