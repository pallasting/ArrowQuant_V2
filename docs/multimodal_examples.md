# Multimodal Encoder System - Usage Examples

## Example 1: Image Encoding

Encode images using the Vision Encoder (CLIP ViT).

```python
import numpy as np
from PIL import Image
from llm_compression.multimodal import get_multimodal_provider

# Initialize provider
provider = get_multimodal_provider(
    vision_model_path="D:/ai-models/clip-vit-b32",
    device="cpu"
)

# Load and preprocess image
image = Image.open("photo.jpg").resize((224, 224))
image_array = np.array(image)

# Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
image_batch = image_array[np.newaxis, ...]

# Encode image
embedding = provider.encode_image(image_batch, normalize=True)
print(f"Image embedding shape: {embedding.shape}")  # (1, 512)
print(f"Embedding norm: {np.linalg.norm(embedding)}")  # ~1.0 (normalized)
```

## Example 2: Audio Encoding

Encode audio using the Audio Encoder (Whisper).

```python
import numpy as np
import librosa
from llm_compression.multimodal import get_multimodal_provider

# Initialize provider
provider = get_multimodal_provider(
    audio_model_path="D:/ai-models/whisper-base",
    device="cpu"
)

# Load audio file (16kHz sample rate required)
audio, sr = librosa.load("speech.wav", sr=16000)

# Add batch dimension if needed
if audio.ndim == 1:
    audio = audio[np.newaxis, ...]

# Encode audio
embedding = provider.encode_audio(audio, normalize=True)
print(f"Audio embedding shape: {embedding.shape}")  # (1, 512)
```

