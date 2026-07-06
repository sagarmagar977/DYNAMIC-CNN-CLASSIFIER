import os
import re
import io
import time
import requests
from PIL import Image

# Configuration
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
PLAYERS = {
    "messi": "Lionel Messi Barcelona",
    "yamal": "Lamine Yamal Barcelona",
    "lewandowski": "Robert Lewandowski Barcelona"
}

# Curated fallback URLs in case scraping fails or to guarantee base set of high-quality images
FALLBACK_URLS = {
    "messi": [
        "https://upload.wikimedia.org/wikipedia/commons/c/c1/Lionel_Messi_20180626.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/b/b4/Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/0/0c/Lionel_Messi_vs_Nigeria_2018.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/e/e0/Lionel_Messi_facing_Nigeria_2018.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/4/4b/Leo_Messi_2018.jpg"
    ],
    "yamal": [
        "https://upload.wikimedia.org/wikipedia/commons/4/47/Lamine_Yamal_2024.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/d/de/Lamine_Yamal_-_03_%28cropped%29.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/0/07/Lamine_Yamal_2023.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/c/cd/Lamine_Yamal_-_02_%28cropped%29.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/b/bd/Lamine_Yamal_cropped.jpg"
    ],
    "lewandowski": [
        "https://upload.wikimedia.org/wikipedia/commons/0/03/Robert_Lewandowski%2C_FC_Bayern_M%C3%BCnchen_%28cropped%29.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/2/23/Robert_Lewandowski_2018.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/0/07/Robert_Lewandowski_2020.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/e/e1/Robert_Lewandowski_CL_2021.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/a/a2/Robert_Lewandowski_Barcelona_2023.jpg"
    ]
}

def get_duckduckgo_image_urls(query, max_images=15):
    print(f"Searching DuckDuckGo for: '{query}'...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    }
    try:
        # Step 1: Get VQD token
        url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}"
        res = requests.get(url, headers=headers, timeout=10)
        vqd_match = re.search(r'vqd=([^&\'"]+)', res.text)
        if not vqd_match:
            print(f"Could not find VQD for query: {query}")
            return []
        vqd = vqd_match.group(1)
        
        # Step 2: Query image API
        image_url = f"https://duckduckgo.com/i.js?q={query.replace(' ', '+')}&o=json&vqd={vqd}"
        res = requests.get(image_url, headers=headers, timeout=10)
        data = res.json()
        results = data.get('results', [])
        urls = [r['image'] for r in results if 'image' in r]
        print(f"Found {len(urls)} image URLs for query: {query}")
        return urls[:max_images]
    except Exception as e:
        print(f"Error scraping DuckDuckGo for {query}: {e}")
        return []

def download_and_save_image(url, folder, file_prefix, idx):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    }
    try:
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code != 200:
            return False
        
        # Verify it's a valid image using PIL
        img_data = res.content
        img = Image.open(io.BytesIO(img_data))
        img.verify()  # Verify image integrity
        
        # Re-open for actual processing (since verify() breaks the img object)
        img = Image.open(io.BytesIO(img_data))
        img = img.convert('RGB')
        
        # Save as JPEG
        filename = f"{file_prefix}_{idx}.jpg"
        filepath = os.path.join(folder, filename)
        img.save(filepath, "JPEG")
        print(f" Saved {filename}")
        return True
    except Exception as e:
        # Ignore download/format errors
        return False

def prepare_dataset(max_images_per_player=15):
    print("Starting dataset download and preparation...")
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    summary = {}
    
    for player, query in PLAYERS.items():
        player_folder = os.path.join(DATASET_DIR, player)
        os.makedirs(player_folder, exist_ok=True)
        
        print(f"\nPreparing dataset for {player.upper()} in {player_folder}")
        
        # Check existing images
        existing_files = [f for f in os.listdir(player_folder) if f.endswith('.jpg')]
        if len(existing_files) >= max_images_per_player:
            print(f"Already have {len(existing_files)} images for {player}. Skipping download.")
            summary[player] = len(existing_files)
            continue
            
        downloaded_count = len(existing_files)
        
        # Try to scrape first
        urls = get_duckduckgo_image_urls(query, max_images=max_images_per_player * 2)
        
        # Append fallback URLs at the front to ensure we get some high-quality ones first
        curated_urls = FALLBACK_URLS.get(player, [])
        all_urls = curated_urls + [u for u in urls if u not in curated_urls]
        
        idx = downloaded_count
        for url in all_urls:
            if downloaded_count >= max_images_per_player:
                break
            
            print(f"Downloading [{downloaded_count + 1}/{max_images_per_player}]: {url}")
            success = download_and_save_image(url, player_folder, player, idx)
            if success:
                downloaded_count += 1
                idx += 1
                time.sleep(0.5)  # Tiny rate-limiting helper
                
        summary[player] = downloaded_count
        print(f"Completed {player.upper()}. Total images: {downloaded_count}")
        
    print("\nDataset preparation completed summary:")
    for player, count in summary.items():
        print(f" - {player.upper()}: {count} images")
    return summary

if __name__ == "__main__":
    prepare_dataset(max_images_per_player=15)
