from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import io
import os
import uuid
import soundfile as sf
import traceback
import shutil  # Neu für das einfache Löschen von Ordnern

app = FastAPI()

OUTPUT_DIR = "outputs"
# Sicherstellen, dass der Ordner existiert
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

SAMPLERATE = 44100

def cleanup_outputs():
    """Löscht alle Dateien im Output-Ordner."""
    for filename in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path) # Löscht Datei oder Link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path) # Löscht Unterordner
        except Exception as e:
            print(f'Fehler beim Löschen von {file_path}: {e}')

@app.get("/")
async def main():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    # --- SCHRITT 0: ALLES ALTE LÖSCHEN ---
    cleanup_outputs()

    try:
        file_content = await file.read()
        
        # --- FALL A: AUDIO ZU BILD ---
        if file.content_type.startswith("audio") or file.filename.endswith((".wav", ".mp3")):
            audio_data, sr = sf.read(io.BytesIO(file_content))
            if len(audio_data.shape) > 1: audio_data = audio_data.mean(axis=1)

            # Header auslesen (W, H, Blockgröße, Skalierung)
            w = int(round((audio_data[0] + 1.0) * 10000))
            h = int(round((audio_data[1] + 1.0) * 10000))
            samples_per_row = int(round((audio_data[2] + 1.0) * 50000))
            max_val = (audio_data[3] + 1.0) * 100.0
            
            current_idx = 4
            img_rows = []
            
            for r in range(h):
                block = audio_data[current_idx : current_idx + samples_per_row]
                if len(block) < samples_per_row:
                    block = np.pad(block, (0, samples_per_row - len(block)))
                
                spectrum = np.fft.rfft(block)
                magnitudes = np.abs(spectrum)
                
                line_data = magnitudes[:w*3]
                pixel_line = (line_data * max_val).clip(0, 255).astype(np.uint8)
                img_rows.append(pixel_line.reshape(w, 3))
                current_idx += samples_per_row

            img_array = np.array(img_rows)
            img = Image.fromarray(img_array, 'RGB')
            
            fn = f"rec_{uuid.uuid4()}.png"
            img.save(os.path.join(OUTPUT_DIR, fn))
            return {"type": "image", "url": f"/static/{fn}", "msg": "Ordner geleert & Bild rekonstruiert!"}

        # --- FALL B: BILD ZU AUDIO ---
        elif file.content_type.startswith("image"):
            img = Image.open(io.BytesIO(file_content)).convert('RGB')
            w, h = img.size
            img_array = np.array(img)
            
            needed_bins = w * 3
            samples_per_row = (needed_bins) * 2
            
            audio_blocks = []
            for r in range(h):
                line_pixels = img_array[r].flatten().astype(np.float32)
                spectrum = np.zeros(samples_per_row // 2 + 1, dtype=np.complex64)
                spectrum[:len(line_pixels)] = line_pixels
                line_audio = np.fft.irfft(spectrum, n=samples_per_row)
                audio_blocks.append(line_audio)
            
            audio_signal = np.concatenate(audio_blocks).astype(np.float32)
            
            actual_max = np.max(np.abs(audio_signal))
            if actual_max > 0:
                audio_signal = audio_signal / actual_max
            
            header = np.zeros(4)
            header[0] = (w / 10000.0) - 1.0
            header[1] = (h / 10000.0) - 1.0
            header[2] = (samples_per_row / 50000.0) - 1.0
            header[3] = (actual_max / 100.0) - 1.0
            
            final_signal = np.concatenate([header, audio_signal])

            fn = f"turbo_{uuid.uuid4()}.wav"
            sf.write(os.path.join(OUTPUT_DIR, fn), final_signal, SAMPLERATE)
            
            return {"type": "audio", "url": f"/static/{fn}", "msg": "Ordner geleert & Audio neu erzeugt!"}

    except Exception:
        print(traceback.format_exc())
        return {"error": "Ein Fehler ist aufgetreten."}