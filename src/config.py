from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Live Data Sources
HF_SPOTIFY_DATASET = "maharshipandya/spotify-tracks-dataset"
BILLBOARD_ALL_JSON_URL = "https://raw.githubusercontent.com/mhollingshead/billboard-hot-100/main/all.json"

# Local Cache & Output Files
BILLBOARD_CACHE_FILE = RAW_DATA_DIR / "billboard_all.json"
OUTPUT_PARQUET = PROCESSED_DATA_DIR / "hit_songs_master.parquet"
OUTPUT_CSV = PROCESSED_DATA_DIR / "hit_songs_master.csv"

# Hit Definition Parameters (Top 40 within first 12 weeks on chart)
HIT_RANK_THRESHOLD = 40
HIT_WEEKS_WINDOW = 12
