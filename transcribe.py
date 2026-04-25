from faster_whisper import WhisperModel
from pathlib import Path
import json, os, sys

wav = Path(sys.argv[1])
out = Path(sys.argv[2])
model = WhisperModel('small', device='cpu', compute_type='int8')
segments, info = model.transcribe(str(wav), language='ko', vad_filter=True)
text = '\n'.join(s.text.strip() for s in segments if s.text.strip())
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text + '\n', encoding='utf-8')
print(json.dumps({
    'file': str(wav),
    'language': info.language,
    'duration': info.duration,
    'text': text[:400]
}, ensure_ascii=False))
