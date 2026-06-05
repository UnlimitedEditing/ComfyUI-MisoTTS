import io
import os
import urllib.request

import torch
import torchaudio.functional as AF

# Generator is imported lazily (inside functions) so ComfyUI can register
# the node classes even before MisoTTS finishes installing.
# Runtime dep: git+https://github.com/MisoLabsAI/MisoTTS.git

_MODEL_CACHE: dict = {}


def _get_generator():
    """Lazy import of MisoTTS generator — raises clearly if not installed."""
    try:
        from generator import load_miso_8b
        return load_miso_8b
    except ImportError:
        raise ImportError(
            "MisoTTS generator not found. "
            "Ensure git+https://github.com/MisoLabsAI/MisoTTS.git is in pip requirements."
        )


def _get_segment_class():
    try:
        from generator import Segment
        return Segment
    except ImportError:
        raise ImportError(
            "MisoTTS Segment not found. "
            "Ensure git+https://github.com/MisoLabsAI/MisoTTS.git is in pip requirements."
        )


class MisoTTSModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_repo": ("STRING", {"default": "MisoLabs/MisoTTS"}),
            }
        }

    RETURN_TYPES = ("MISO_TTS",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "MisoTTS"

    def load(self, model_repo):
        if model_repo not in _MODEL_CACHE:
            load_miso_8b = _get_generator()
            os.environ.setdefault("NO_TORCH_COMPILE", "1")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _MODEL_CACHE[model_repo] = load_miso_8b(
                device=device,
                model_path_or_repo_id=model_repo,
            )
        return (_MODEL_CACHE[model_repo],)


class LoadAudioFromURL:
    """Download an audio file from a URL and return it as a ComfyUI AUDIO tensor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load"
    CATEGORY = "MisoTTS"

    def load(self, url):
        import torchaudio

        with urllib.request.urlopen(url) as resp:
            data = resp.read()

        waveform, sample_rate = torchaudio.load(io.BytesIO(data))
        # ComfyUI AUDIO type: waveform shape [batch, channels, samples]
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)


class MisoTTSGenerate:
    """
    speaker IDs: 0 = friend (warm/conversational)
                 1 = teacher (clear/instructional)
                 2 = voiceover (professional)

    Connect context_audio to clone a reference voice.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MISO_TTS",),
                "text": ("STRING", {"multiline": True, "default": "Hello from Miso."}),
                "speaker": ("INT", {"default": 0, "min": 0, "max": 2, "step": 1}),
                "max_audio_length_ms": (
                    "FLOAT",
                    {"default": 10000.0, "min": 1000.0, "max": 90000.0, "step": 500.0},
                ),
            },
            "optional": {
                "context_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "MisoTTS"

    def generate(self, model, text, speaker, max_audio_length_ms, context_audio=None):
        context = []
        if context_audio is not None:
            Segment = _get_segment_class()
            waveform = context_audio["waveform"]  # [B, C, S]
            src_sr = context_audio["sample_rate"]
            target_sr = model.sample_rate  # 24 000 Hz (Mimi codec)

            wav = waveform[0].mean(dim=0)  # mono [S]
            if src_sr != target_sr:
                wav = AF.resample(wav, src_sr, target_sr)

            context = [Segment(speaker=int(speaker), text="", audio=wav.cpu())]

        audio = model.generate(
            text=text,
            speaker=int(speaker),
            context=context,
            max_audio_length_ms=float(max_audio_length_ms),
        )
        waveform_out = audio.unsqueeze(0).unsqueeze(0).cpu()
        return ({"waveform": waveform_out, "sample_rate": model.sample_rate},)


NODE_CLASS_MAPPINGS = {
    "MisoTTSModelLoader": MisoTTSModelLoader,
    "LoadAudioFromURL": LoadAudioFromURL,
    "MisoTTSGenerate": MisoTTSGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MisoTTSModelLoader": "MisoTTS Model Loader",
    "LoadAudioFromURL": "Load Audio From URL",
    "MisoTTSGenerate": "MisoTTS Generate",
}
