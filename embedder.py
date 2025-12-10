import os
import json
import time
import torch
from sentence_transformers import SentenceTransformer
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchaudio.models import wav2vec2_base
import torchaudio
from PIL import Image
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from moviepy import VideoFileClip
from utils import VECTOR_DIM, save_json
import subprocess
import soundfile as sf
import numpy as np
import pandas as pd
from searcher import setup_collection
from utils import connect_milvus, VECTOR_DIM, COLLECTION_NAME
from pymilvus import Collection

console = Console()
image_projector = torch.nn.Linear(2048, VECTOR_DIM)
sound_projector = torch.nn.Linear(768, VECTOR_DIM)

def embed_books_csv(csv_path: str):
    """Load a Kaggle books CSV, embed titles/authors/genres, and insert into Milvus."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        console.print(f"[red]Failed to read {csv_path}: {e}[/red]")
        return

    # Detect columns
    col_title = next((c for c in df.columns if 'title' in c.lower()), None)
    col_author = next((c for c in df.columns if 'author' in c.lower()), None)
    col_genre = next((c for c in df.columns if 'genre' in c.lower()), None)
    col_id = next((c for c in df.columns if 'id' in c.lower() or 'Unnamed' in c), None)

    if not col_title:
        console.print("[red]CSV must contain a title column[/red]")
        return

    # Build docs
    if col_author:
        docs = (df[col_title].astype(str) + " by " + df[col_author].astype(str)).tolist()
    else:
        docs = df[col_title].astype(str).tolist()
    if col_genre:
        docs = [f"{d} | genre: {g}" for d, g in zip(docs, df[col_genre].astype(str).tolist())]

    ids = df[col_id].astype(int).tolist() if col_id else list(range(len(docs)))
    metas = docs

    # Embed using SentenceTransformers
    console.print(f"[cyan]Embedding {len(docs)} book records with SentenceTransformers...[/cyan]")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vectors = model.encode(docs, show_progress_bar=True)

    # Normalize for COSINE
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Connect Milvus
    connect_milvus(host="127.0.0.1", port="19530")

    # Setup collection
    collection = setup_collection(index_type="IVF_FLAT")

    try:
        insert_data = [ids, vectors.tolist(), metas]
        collection.insert(insert_data)
        collection.flush()
        console.print(f"[green]Flushed collection, entities: {collection.num_entities}[/green]")
        collection.load()
        console.print(f"[green]Loaded collection, entities: {collection.num_entities}[/green]")
        console.print(f"[green]Inserted {collection.num_entities} entities into '{COLLECTION_NAME}'[/green]")
    except Exception as e:
        console.print(f"[red]Failed to insert books: {e}[/red]")

def embed_text(text):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(text).tolist()
        norm = np.linalg.norm(embedding)
        if norm == 0:
            console.print(f"[red]Warning: Zero norm embedding for text[/red]")
            return None
        return embedding
    except Exception as e:
        console.print(f"[red]Text embedding failed: {e}[/red]")
        return None
    finally:
        if 'model' in locals():
            del model

def embed_image(image_path):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = torch.nn.Sequential(*list(resnet50(weights=weights).children())[:-1]).eval()
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            embedding = model(input_tensor).squeeze()
            embedding = image_projector(embedding)
            norm = torch.norm(embedding)
            if norm == 0:
                console.print(f"[red]Warning: Zero norm embedding for {image_path}[/red]")
                return None
            embedding = embedding / norm
            embedding = embedding.tolist()
        return embedding
    except Exception as e:
        console.print(f"[red]Image embedding failed for {image_path}: {e}[/red]")
        return None
    finally:
        if 'image' in locals():
            del image
        if 'input_tensor' in locals():
            del input_tensor
        if 'model' in locals():
            del model

def embed_video(video_path):
    embeddings = []
    audio_path = "tmp_audio.wav"
    try:
        clip = VideoFileClip(video_path)
        fps_interval = 1
        num_frames = int(clip.duration // fps_interval)
        for i in range(num_frames):
            frame_time = i * fps_interval
            frame = clip.get_frame(frame_time)
            frame_image = Image.fromarray(frame)
            frame_file = f"tmp_frame_{i}.jpg"
            frame_image.save(frame_file)
            vec = embed_image(frame_file)
            if vec is not None:
                embeddings.append({
                    "id": f"{video_path}_frame_{i}",
                    "vector": vec,
                    "type": "video-frame",
                    "content": f"{video_path} @ {frame_time:.1f}s"
                })
            if os.path.exists(frame_file):
                os.remove(frame_file)
            del frame, frame_image
        if clip.audio:
            clip.audio.write_audiofile(audio_path, fps=16000, logger=None)
            vec = embed_sound(audio_path)
            if vec is not None:
                embeddings.append({
                    "id": f"{video_path}_audio",
                    "vector": vec,
                    "type": "video-audio",
                    "content": video_path
                })
            if os.path.exists(audio_path):
                os.remove(audio_path)
        clip.close()
    except Exception as e:
        console.print(f"[red]Video embedding failed for {video_path}: {e}[/red]")
    return embeddings

def load_wav_ffmpeg(path: str):
    try:
        data, samplerate = sf.read(path, dtype="float32")
        waveform = torch.tensor(data).unsqueeze(0)
        return waveform, samplerate
    except Exception as e:
        console.print(f"[red]Failed to load audio {path}: {e}[/red]")
        raise

def embed_sound(sound_path):
    try:
        if not os.path.exists(sound_path):
            raise FileNotFoundError(f"Audio file not found: {sound_path}")
        waveform, sample_rate = load_wav_ffmpeg(sound_path)
        if waveform.ndim > 2:
            waveform = waveform.mean(dim=-1)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        model = bundle.get_model().eval()
        with torch.no_grad():
            features, _ = model(waveform)
            embedding = features.mean(dim=1).squeeze()
            embedding = sound_projector(embedding)
            norm = torch.norm(embedding)
            if norm == 0:
                console.print(f"[red]Warning: Zero norm embedding for {sound_path}[/red]")
                return None
            embedding = embedding / norm
            embedding = embedding.tolist()
        return embedding
    except Exception as e:
        console.print(f"[red]Sound embedding failed for {sound_path}: {e}[/red]")
        ffmpeg_path = "D:\\ffmpeg-2025-09-10-git-c1dc2e2b7c-full_build\\bin\\ffmpeg.exe"
        if not os.path.exists(ffmpeg_path):
            console.print(f"[red]FFmpeg not found at: {ffmpeg_path}[/red]")
            return None
        try:
            subprocess.run([
                ffmpeg_path, "-i", sound_path, "-f", "wav", "-ar", "16000",
                "-acodec", "pcm_s16le", "-y", "temp.wav"
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            waveform, sample_rate = load_wav_ffmpeg("temp.wav")
            if waveform.ndim > 2:
                waveform = waveform.mean(dim=-1)
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
            model = bundle.get_model().eval()
            with torch.no_grad():
                features, _ = model(waveform)
                embedding = features.mean(dim=1).squeeze()
                embedding = sound_projector(embedding)
                norm = torch.norm(embedding)
                if norm == 0:
                    console.print(f"[red]Warning: Zero norm embedding for {sound_path} (fallback)[/red]")
                    return None
                embedding = embedding / norm
                embedding = embedding.tolist()
            os.remove("temp.wav")
            return embedding
        except Exception as e2:
            console.print(f"[red]Fallback failed for {sound_path}: {e2}[/red]")
            return None
    finally:
        if 'waveform' in locals():
            del waveform
        if 'model' in locals():
            del model

def generate_embeddings(data_dir: str, output_file: str = "data_embeddings.json"):
    """Generate embeddings for all supported files in data_dir and save to output_file"""
    embeddings = []
    file_id = 0
    supported_extensions = {'.txt', '.jpg', '.png', '.jpeg', '.bmp', '.wav', '.mp3', '.mp4', '.avi', '.mov'}
    total_files = sum(1 for root, _, files in os.walk(data_dir)
                      for fname in files if os.path.splitext(fname.lower())[1] in supported_extensions)
    
    if total_files == 0:
        console.print(f"[red]No supported files found in {data_dir}[/red]")
        save_json([], output_file)
        return []

    invalid_count = 0
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[progress.completed]{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Embedding files (ETM: calculating)...", total=total_files)
        start_time = time.time()
        processed_files = 0

        for root, _, files in os.walk(data_dir):
            for fname in files:
                if os.path.splitext(fname.lower())[1] not in supported_extensions:
                    continue
                fpath = os.path.join(root, fname)
                ext = os.path.splitext(fname.lower())[1]
                if ext == '.txt':
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            text = f.read().strip()
                        vec = embed_text(text)
                        if vec is not None:
                            embeddings.append({"id": file_id, "vector": vec, "type": "text", "content": fpath})
                            file_id += 1
                            processed_files += 1
                        else:
                            invalid_count += 1
                    except Exception as e:
                        console.print(f"[red]Failed reading {fpath}: {e}[/red]")
                        invalid_count += 1
                elif ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                    vec = embed_image(fpath)
                    if vec is not None:
                        embeddings.append({"id": file_id, "vector": vec, "type": "image", "content": fpath})
                        file_id += 1
                        processed_files += 1
                    else:
                        invalid_count += 1
                elif ext in ['.wav', '.mp3']:
                    vec = embed_sound(fpath)
                    if vec is not None:
                        embeddings.append({"id": file_id, "vector": vec, "type": "sound", "content": fpath})
                        file_id += 1
                        processed_files += 1
                    else:
                        invalid_count += 1
                elif ext in ['.mp4', '.avi', '.mov']:
                    vids = embed_video(fpath)
                    for v in vids:
                        v["id"] = file_id
                        embeddings.append(v)
                        file_id += 1
                        processed_files += 1
                    if not vids:
                        invalid_count += 1
                progress.update(task, advance=1)
                if processed_files % 100 == 0 and processed_files > 0:
                    elapsed = time.time() - start_time
                    time_per_file = elapsed / processed_files
                    remaining_files = total_files - processed_files
                    etm = remaining_files * time_per_file
                    progress.update(task, description=f"[cyan]Embedding files (ETM: {etm:.0f}s)...")
        
        elapsed = time.time() - start_time
        progress.update(task, description=f"[cyan]Embedding files (ETM: 0s, Total: {elapsed:.0f}s)...")
    
    if invalid_count > 0:
        console.print(f"[red]Processed {processed_files}/{total_files} files, {invalid_count} invalid embeddings[/red]")
    try:
        save_json(embeddings, output_file)
    except Exception as e:
        console.print(f"[red]Failed to save embeddings to {output_file}: {e}[/red]")
        raise
    return []