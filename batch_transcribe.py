from faster_whisper import WhisperModel
import sys, os, glob, time
rd=sys.argv[1]
start=int(sys.argv[2]); end=int(sys.argv[3])
model = WhisperModel('small', device='cpu', compute_type='int8')
for i in range(start, end+1):
    chunk=f'{rd}/audio/chunk_{i:06d}.wav'
    out=f'{rd}/transcripts/chunk_{i:06d}.txt'
    if not os.path.exists(chunk):
        print(f'{i}: missing')
        continue
    if os.path.exists(out) and os.path.getsize(out)>0:
        print(f'{i}: exists')
        continue
    t0=time.time()
    segments, info = model.transcribe(chunk, language='ko', vad_filter=True)
    text='\n'.join(s.text.strip() for s in segments if s.text.strip())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out,'w',encoding='utf-8') as f:
        f.write(text + ('\n' if text else ''))
    print(f'{i}: chars={len(text)} sec={time.time()-t0:.1f}')
