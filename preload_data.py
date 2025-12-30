import kagglehub
import os
import shutil
import glob

# Define paths
TARGET_DIR = "data"
# We normalize the name to 'books.csv' for the app
TARGET_FILE = os.path.join(TARGET_DIR, "books.csv") 

def load_data():
    if os.path.exists(TARGET_FILE):
        print(f"✅ Data found at {TARGET_FILE}. Skipping download.")
        return

    print("⬇️ Data not found. Downloading 'saurabhbagchi/books-dataset'...")
    
    try:
        # 1. Download from Kaggle
        download_path = kagglehub.dataset_download("saurabhbagchi/books-dataset")
        print(f"   Downloaded to cache: {download_path}")

        # 2. Find 'Books.csv' (Case sensitive search)
        # We look recursively because kagglehub might create subfolders
        search_pattern = os.path.join(download_path, "**", "Books.csv")
        found_files = glob.glob(search_pattern, recursive=True)
        
        if not found_files:
            # Fallback: try case-insensitive search
            all_csvs = glob.glob(os.path.join(download_path, "**", "*.csv"), recursive=True)
            found_files = [f for f in all_csvs if "books" in os.path.basename(f).lower()]

        if not found_files:
            print("❌ Error: Could not find 'Books.csv' in the downloaded files.")
            return

        source_csv = found_files[0]
        
        # 3. Move and Rename
        os.makedirs(TARGET_DIR, exist_ok=True)
        print(f"📦 Found {os.path.basename(source_csv)}. Moving to {TARGET_FILE}...")
        shutil.copy(source_csv, TARGET_FILE)
        
        print("✅ Setup complete.")

    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")

if __name__ == "__main__":
    load_data()