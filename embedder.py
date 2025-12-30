import os
import time
import torch
import torchaudio
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from PIL import Image
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from moviepy import VideoFileClip

# Local imports
from utils import connect_milvus, save_json, VECTOR_DIM, COLLECTION_NAME
from searcher import setup_collection

console = Console()

# --- 1. INITIALIZE MODELS ---
# We use CLIP for unified Text & Image embeddings (512 dimensions)
# This aligns text queries with image content perfectly.
console.print("[cyan]Loading CLIP model (Text & Image)...[/cyan]")
CLIP_MODEL = SentenceTransformer('clip-ViT-B-32')

# Audio Setup (Wav2Vec)
# Note: Audio lives in its own vector space unless you use a CLAP model.
# We fix the random seed so the random projection is consistent across restarts.
torch.manual_seed(42) 
sound_projector = torch.nn.Linear(768, VECTOR_DIM)
sound_projector.eval() # Set to eval mode

def embed_text(text: str):
    """Embed text using CLIP."""
    try:
        # CLIP encodes text into the same 512-dim space as images
        embedding = CLIP_MODEL.encode(text).tolist()
        return embedding
    except Exception as e:
        console.print(f"[red]Text embedding failed: {e}[/red]")
        return None

def embed_image(image_path: str):
    """Embed image using CLIP."""
    try:
        if not os.path.exists(image_path):
            return None
        
        # CLIP can encode PIL images directly
        img = Image.open(image_path).convert("RGB")
        embedding = CLIP_MODEL.encode(img).tolist()
        return embedding
    except Exception as e:
        console.print(f"[red]Image embedding failed for {image_path}: {e}[/red]")
        return None

def embed_sound(sound_path: str):
    """
    Embed audio using Wav2Vec + Linear Projection.
    NOTE: Audio search will only match other audio files, not text/images,
    unless you train a specific alignment adapter.
    """
    try:
        if not os.path.exists(sound_path):
            return None
            
        waveform, sample_rate = torchaudio.load(sound_path)
        
        # Resample to 16k if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Average channels if stereo
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Load Wav2Vec bundle
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        model = bundle.get_model()
        model.eval()

        with torch.no_grad():
            features, _ = model(waveform)
            # Mean pooling over time
            embedding = features.mean(dim=1).squeeze()
            # Project to VECTOR_DIM (512)
            embedding = sound_projector(embedding)
            
            # Normalize
            norm = torch.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
        return embedding.tolist()
    except Exception as e:
        console.print(f"[red]Sound embedding failed for {sound_path}: {e}[/red]")
        return None

def embed_video(video_path: str):
    """Extract frames and audio from video."""
    embeddings = []
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        
        # 1. Extract Frames (1 per second)
        for t in range(0, int(duration), 1):
            frame_path = f"temp_frame_{t}.jpg"
            clip.save_frame(frame_path, t=t)
            
            vec = embed_image(frame_path)
            if vec:
                embeddings.append({
                    "id": f"{os.path.basename(video_path)}_frame_{t}",
                    "vector": vec,
                    "type": "video-frame",
                    "content": f"{video_path} @ {t}s"
                })
            
            if os.path.exists(frame_path):
                os.remove(frame_path)

        # 2. Extract Audio
        if clip.audio:
            audio_path = "temp_audio.wav"
            clip.audio.write_audiofile(audio_path, fps=16000, verbose=False, logger=None)
            vec = embed_sound(audio_path)
            if vec:
                embeddings.append({
                    "id": f"{os.path.basename(video_path)}_audio",
                    "vector": vec,
                    "type": "video-audio",
                    "content": video_path
                })
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
        clip.close()
    except Exception as e:
        console.print(f"[red]Video processing failed for {video_path}: {e}[/red]")
    
    return embeddings

def embed_books_csv(csv_path: str):
    """Load Kaggle Books.csv, embed text, and insert into Milvus."""
    try:
        # KAGGLE FIX 1: Try reading with different encodings and separators
        try:
            # Standard attempt
            df = pd.read_csv(csv_path)
        except:
            try:
                # Common Kaggle format (Latin-1 and semi-colon separator)
                df = pd.read_csv(csv_path, encoding="latin-1", sep=";", on_bad_lines='skip')
                console.print("[yellow]Detected Latin-1 encoding with semicolon separator.[/yellow]")
            except Exception as e:
                console.print(f"[red]Failed to read CSV. Error: {e}[/red]")
                return

        # KAGGLE FIX 2: Normalize Column Names
        # Map 'Book-Title' -> 'title', 'Book-Author' -> 'author'
        df.columns = [c.strip().lower().replace('-', '_').replace(' ', '_') for c in df.columns]
        # Now columns look like: 'book_title', 'book_author', 'year_of_publication'
        
    except Exception as e:
        console.print(f"[red]Failed to process {csv_path}: {e}[/red]")
        return

    # Identify valid title column
    col_title = next((c for c in df.columns if 'title' in c), None)
    
    if not col_title:
        console.print(f"[red]CSV must contain a title column. Found: {df.columns.tolist()}[/red]")
        return

    docs = []
    metas = []
    
    # Process rows (limit to 2000 for speed if running locally, remove .head(2000) for full)
    console.print("[cyan]Processing rows...[/cyan]")
    for _, row in df.head(2000).iterrows():
        try:
            # Construct text: "Harry Potter by J.K. Rowling"
            text_parts = [str(row[col_title])]
            
            # Check for author column variants
            col_author = next((c for c in df.columns if 'author' in c), None)
            if col_author and pd.notna(row[col_author]):
                text_parts.append(f"by {row[col_author]}")
            
            # Use Publisher as Genre proxy if Genre is missing (common in this dataset)
            col_pub = next((c for c in df.columns if 'publisher' in c), None)
            if col_pub and pd.notna(row[col_pub]):
                text_parts.append(f"| Publisher: {row[col_pub]}")

            full_text = " ".join(text_parts)
            
            # Simple validity check
            if len(full_text) > 5:
                docs.append(full_text)
                metas.append(full_text)
        except Exception:
            continue

    if not docs:
        console.print("[red]No valid documents extracted.[/red]")
        return

    # Embed batch
    console.print(f"[cyan]Embedding {len(docs)} books with CLIP...[/cyan]")
    vectors = CLIP_MODEL.encode(docs, show_progress_bar=True)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    connect_milvus()
    collection = setup_collection(index_type="HNSW")

    try:
        collection.insert([vectors.tolist(), metas])
        collection.flush()
        console.print(f"[green]Inserted {len(docs)} books into '{COLLECTION_NAME}'[/green]")
    except Exception as e:
        console.print(f"[red]Failed to insert books: {e}[/red]")

def generate_embeddings(data_dir: str, output_file: str = "data_embeddings.json"):
    """Walk directory, embed supported files, save to JSON."""
    embeddings = []
    file_id = 0
    supported = {'.txt', '.jpg', '.png', '.jpeg', '.wav', '.mp3', '.mp4', '.mov'}
    
    files_to_process = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in supported:
                files_to_process.append(os.path.join(root, f))

    if not files_to_process:
        console.print(f"[red]No supported files in {data_dir}[/red]")
        return []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Processing files...", total=len(files_to_process))
        
        for fpath in files_to_process:
            ext = os.path.splitext(fpath)[1].lower()
            vec = None
            type_tag = "unknown"
            
            if ext == '.txt':
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        vec = embed_text(f.read().strip())
                    type_tag = "text"
                except: pass
            
            elif ext in ['.jpg', '.png', '.jpeg']:
                vec = embed_image(fpath)
                type_tag = "image"
            
            elif ext in ['.wav', '.mp3']:
                vec = embed_sound(fpath)
                type_tag = "sound"
                
            elif ext in ['.mp4', '.mov']:
                # Videos return a list of frame embeddings
                v_embeddings = embed_video(fpath)
                for v in v_embeddings:
                    v['id'] = file_id
                    embeddings.append(v)
                    file_id += 1
                progress.advance(task)
                continue # Skip standard append

            if vec is not None:
                embeddings.append({
                    "id": file_id,
                    "vector": vec,
                    "type": type_tag,
                    "content": fpath
                })
                file_id += 1
            
            progress.advance(task)

    save_json(embeddings, output_file)
    console.print(f"[green]Saved {len(embeddings)} embeddings to {output_file}[/green]")
    return embeddings