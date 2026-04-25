from faster_whisper import WhisperModel
import sys, os
chunk, out = sys.argv[1], sys.argv[2]
model = WhisperModel('small', device='cpu', compute_type='int8')
segments, info = model.transcribe(chunk, language='ko', vad_filter=True)
text = '\n'.join(s.text.strip() for s in segments if s.text.strip())
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, 'w', encoding='utf-8').write(text + ('\n' if text else ''))
print(out, len(text))
