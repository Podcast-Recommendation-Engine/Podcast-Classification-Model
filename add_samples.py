import pandas as pd
import uuid
import random
import re

# --- CONFIGURATION ---
NUM_SAMPLES = 996 
OUTPUT_FILE = r'c:\Users\Ibrahim\Documents\WORK\Faculty-Projects\5eme\Podcast-Classification\data\annotated\podcasts_annotated.csv'

# --- KID-FRIENDLY DATA POOLS ---
kids_pools = [
    {
        "category": "Space & Science",
        "templates": ["Why is the {item} {color}?", "Journey to {item}", "The {item} Discovery for Kids", "Exploring {item} with Science"],
        "item": ["Mars", "The Moon", "Saturn's Rings", "The Sun", "Black Hole", "Milky Way", "Jupiter"],
        "color": ["Red", "Blue", "Glowing", "Bright", "Huge"],
        "base_keywords": ["science", "space", "astronomy", "education", "learning", "planets", "nasa"],
        "kid_friendly": 1
    },
    {
        "category": "Animal Kingdom",
        "templates": ["Meet the {adj} {animal}", "The Secret Language of {animal}s", "Life of a {adj} {animal}", "Amazing {animal} Facts"],
        "adj": ["Brave", "Cuddly", "Speedy", "Giant", "Tiny", "Wild"],
        "animal": ["Dolphin", "Elephant", "Penguin", "Lion", "Hamster", "Cheetah", "Whale"],
        "base_keywords": ["animals", "nature", "zoology", "wildlife", "fun facts", "creatures"],
        "kid_friendly": 1
    },
    {
        "category": "Fairy Tales & Myths",
        "templates": ["The {hero} and the {creature}", "The Legend of the {adj} {place}", "A Story of {hero} in {place}", "The {adj} {creature}"],
        "hero": ["Princess", "Knight", "Young Explorer", "Magic Cat", "Brave Boy"],
        "creature": ["Dragon", "Unicorn", "Talking Tree", "Friendly Giant", "Phoenix"],
        "adj": ["Golden", "Hidden", "Enchanted", "Lost", "Sparkling"],
        "place": ["Castle", "Magic Mountain", "Crystal Cave", "Rainbow Forest"],
        "base_keywords": ["stories", "fairy tale", "magic", "adventure", "imagination", "bedtime", "fantasy"],
        "kid_friendly": 1
    },
    {
        "category": "School & Learning",
        "templates": ["Fun with {subject}", "Mastering {subject} for Kids", "{subject} Riddles", "The History of {subject} for Beginners"],
        "subject": ["Math", "English", "History", "Painting", "Piano", "Coding", "Geography"],
        "base_keywords": ["education", "school", "learning", "teacher", "student", "brain power", "knowledge"],
        "kid_friendly": 1
    }
]

generated_data = []

for _ in range(NUM_SAMPLES):
    pool = random.choice(kids_pools)
    template = random.choice(pool["templates"])
    
    # Extract keys like {item}, {adj}, {hero}, etc.
    keys_needed = re.findall(r'\{(.*?)\}', template)
    format_dict = {key: random.choice(pool[key]) for key in keys_needed}
    
    title = template.format(**format_dict)
    
    # Keyword generation logic
    raw_keywords = list(set(pool["base_keywords"] + title.lower().replace("?", "").split()))
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
df_kids = pd.DataFrame(generated_data)
df_kids.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

print(f"Success! Added {NUM_SAMPLES} kid-friendly samples. Your dataset is now balanced and massive.")