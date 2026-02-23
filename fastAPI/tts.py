import torch
import numpy as np
import soundfile as sf
import re
from IPython.display import Audio as IPyAudio, display
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model_inf = SpeechT5ForTextToSpeech.from_pretrained("../model_tts_finetuning/speecht5_tts_colombian_final").to(DEVICE)
processor_inf = SpeechT5Processor.from_pretrained("../model_tts_finetuning/speecht5_tts_colombian_final")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)
model_inf.eval()
spk_emb_avg = torch.tensor(np.load("../model_tts_finetuning/speaker_embedding.npy"), dtype=torch.float32).unsqueeze(0).to(DEVICE)


def preparar_texto(texto):
    texto = texto.replace("¿", "").replace("¡", "")
    texto = texto.replace(":", " ")
    texto = texto.replace(";", "")
    texto = texto.replace(",", "")
    return re.sub(r'\s+', ' ', texto).strip()


def dividir_en_frases(texto, max_tokens=50):

    PATRONES_CORTE = [
        r',\s+',
        r'\s+(pero|sino|aunque|porque|cuando|donde|como|mientras|después|antes|entonces|ya que)\s+',
        r'\s+(y|e|o|u)\s+(?=\w{4,})',
    ]
    
    n_tokens = processor_inf(text=texto, return_tensors="pt")["input_ids"].shape[1]
    if n_tokens <= max_tokens:
        return [texto]

    for patron in PATRONES_CORTE:
        segmentos = re.split(f'({patron})', texto, flags=re.IGNORECASE)
        partes_candidatas = []
        chunk = ""

        for seg in segmentos:
            chunk += seg
            tok = processor_inf(text=chunk.strip(), return_tensors="pt")["input_ids"].shape[1]

            if tok >= max_tokens:
                anterior = chunk[:-(len(seg))].strip()
                if anterior:
                    partes_candidatas.append(anterior)
                chunk = seg.strip()

        if chunk.strip():
            partes_candidatas.append(chunk.strip())
        if all(processor_inf(text=p, return_tensors="pt")["input_ids"].shape[1] <= max_tokens for p in partes_candidatas) and len(partes_candidatas) > 1:
            return partes_candidatas
        
    palabras = texto.split()
    partes, chunk_actual = [], []
    for palabra in palabras:
        chunk_actual.append(palabra)

        if processor_inf(text=" ".join(chunk_actual), return_tensors="pt")["input_ids"].shape[1] >= max_tokens:
            partes.append(" ".join(chunk_actual))
            chunk_actual = []

    if chunk_actual:
        partes.append(" ".join(chunk_actual))
        
    return partes


def recortar_silencio(audio, sr=16000, margen_ms=150):
    umbral = max(audio.abs().max().item() * 0.05, 0.001)
    margen = int(sr * margen_ms / 1000)
    indices = (audio.abs() > umbral).nonzero(as_tuple=True)[0]
    if len(indices) == 0:
        return audio

    return audio[max(0, indices[0].item() - margen):min(len(audio), indices[-1].item() + margen)]


def generar_chunk(texto, threshold=0.5, minlenratio=0.1):
    if texto.strip()[-1] not in ".,":
        texto = texto.strip() + "."

    inputs = processor_inf(text=texto, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        speech = model_inf.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=spk_emb_avg,
            vocoder=vocoder,
            threshold=threshold,
            minlenratio=minlenratio,
            maxlenratio=20.0,
        )

    return speech.cpu()


def intentar_generar(parte, amp_minima=0.05, profundidad=0):
    for params in [
        {"threshold": 0.5, "minlenratio": 0.1},
        {"threshold": 0.4, "minlenratio": 0.2},
        {"threshold": 0.6, "minlenratio": 0.0},
        {"threshold": 0.35, "minlenratio": 0.3},
    ]:
        speech = generar_chunk(parte, params["threshold"], params["minlenratio"])
        if speech.abs().max().item() >= amp_minima:
            return [recortar_silencio(speech)]

    palabras = parte.strip().split()
    if len(palabras) <= 2 or profundidad >= 3:
        return [torch.zeros(int(16000 * 0.5))]

    mitad = len(palabras) // 2
    return (
        intentar_generar(" ".join(palabras[:mitad]), amp_minima, profundidad + 1)
        + intentar_generar(" ".join(palabras[mitad:]), amp_minima, profundidad + 1)
    )
    
    
def crossfade(audio1: torch.Tensor, audio2: torch.Tensor, sr=16000, fade_ms=60, pausa_ms=80) -> torch.Tensor:

    fade_samples  = int(sr * fade_ms / 1000)
    pausa_samples = int(sr * pausa_ms / 1000)

    if len(audio1) >= fade_samples:
        fade_out = torch.linspace(1.0, 0.0, fade_samples)
        audio1 = audio1.clone()
        audio1[-fade_samples:] *= fade_out

    if len(audio2) >= fade_samples:
        fade_in = torch.linspace(0.0, 1.0, fade_samples)
        audio2 = audio2.clone()
        audio2[:fade_samples] *= fade_in

    pausa = torch.zeros(pausa_samples)
    return torch.cat([audio1, pausa, audio2])


def ajustar_velocidad(audio: torch.Tensor, sr: int = 16000, velocidad: float = 0.96) -> torch.Tensor:
    import torchaudio.functional as F
    return F.resample(audio, int(sr * velocidad), sr)


def generar_audio(texto, nombre="output.wav", max_tokens=50):
    
    texto_limpio = preparar_texto(texto)
    partes = dividir_en_frases(texto_limpio, max_tokens)
    partes_filtradas = []
    
    for parte in partes:
        n_tok = processor_inf(text=parte, return_tensors="pt")["input_ids"].shape[1]
        if n_tok < 15 and partes_filtradas:
            partes_filtradas[-1] += " " + parte
        else:
            partes_filtradas.append(parte)

    partes = partes_filtradas
    audios_partes = []
    for parte in partes:
        chunks = intentar_generar(parte)
        if chunks:
            audios_partes.append(torch.cat(chunks))

    if not audios_partes:
        print("No se generó audio")
        return

    audio_final = audios_partes[0]
    for i in range(1, len(audios_partes)):
        parte_anterior = partes[i - 1]
        if parte_anterior.strip()[-1] == ".":
            pausa_ms = 180
        else:
            pausa_ms = 80
        audio_final = crossfade(audio_final, audios_partes[i], pausa_ms=pausa_ms)

    audio_final = ajustar_velocidad(audio_final, sr=16000, velocidad=0.96)
    if audio_final.abs().max().item() > 0.001:
        sf.write(nombre, audio_final.numpy(), samplerate=16000)
        display(IPyAudio(nombre))
    else:
        print("Audio silencioso")