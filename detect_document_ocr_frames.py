#!/usr/bin/env python3
from pathlib import Path
import json, re, sys, math
from rapidocr_onnxruntime import RapidOCR

RUN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
CAP_DIR = RUN_DIR / 'captures'
OUT = RUN_DIR / 'document_ocr_detection.json'
ocr = RapidOCR()

def frame_no(p):
    return int(re.search(r'(\d+)', p.stem).group(1))

def is_onair_text(s):
    t = (s or '').upper().replace(' ', '')
    return 'ONAIR' in t or 'ONAR' in t

def analyze(p):
    result, elapse = ocr(str(p))
    boxes=[]
    if result:
        for item in result:
            pts, text, score = item[0], str(item[1]), float(item[2])
            if score < 0.45:
                continue
            xs=[pt[0] for pt in pts]; ys=[pt[1] for pt in pts]
            x1,y1,x2,y2=min(xs),min(ys),max(xs),max(ys)
            w,h=x2-x1,y2-y1
            if w <= 2 or h <= 2:
                continue
            boxes.append({'text':text,'score':round(score,3),'x1':x1,'y1':y1,'x2':x2,'y2':y2,'w':w,'h':h,'onair':is_onair_text(text)})
    # Exclude ON AIR and tiny corner labels from document score.
    content=[b for b in boxes if not b['onair']]
    # Group rows by y center.
    rows=[]
    for b in sorted(content, key=lambda b:(b['y1']+b['y2'])/2):
        cy=(b['y1']+b['y2'])/2
        placed=False
        for r in rows:
            if abs(cy-r['cy']) <= max(14, r['h']*0.8, b['h']*0.8):
                r['items'].append(b)
                ys=[i['y1'] for i in r['items']]; y2s=[i['y2'] for i in r['items']]
                r['cy']=(min(ys)+max(y2s))/2; r['h']=max(y2s)-min(ys)
                placed=True; break
        if not placed:
            rows.append({'cy':cy,'h':b['h'],'items':[b]})
    good_rows=[]
    for r in rows:
        xs=[i['x1'] for i in r['items']]; x2s=[i['x2'] for i in r['items']]
        width=max(x2s)-min(xs) if xs else 0
        # OCR may detect one word per row for a slide title, so accept rows with any nontrivial text.
        texts=' '.join(i['text'] for i in r['items'])
        if len(texts.strip()) >= 1 and width >= 15:
            good_rows.append(r)
    line_count=len(good_rows)
    content_count=len(content)
    # User rule: PPT/document or images with 2+ lines of text. OCR boxes/rows are used as code-only signal.
    is_doc = line_count >= 2 or content_count >= 4
    return {
        'file': p.name,
        'frame': frame_no(p),
        'is_document': bool(is_doc),
        'line_count': line_count,
        'ocr_box_count': len(boxes),
        'content_box_count': content_count,
        'texts': [b['text'] for b in content[:12]],
        'scores': [b['score'] for b in content[:12]],
    }

# Scan every captured frame with OCR; this is code-based, no LLM/vision model calls.
files=sorted(CAP_DIR.glob('frame_*.jpg'), key=frame_no)
results=[]
for idx,p in enumerate(files,1):
    r=analyze(p)
    results.append(r)
    if idx % 50 == 0:
        print(f'scanned {idx}/{len(files)}', file=sys.stderr)

docs=[r for r in results if r['is_document']]
# Group detections close in time; choose row/box richest representative.
groups=[]; cur=[]; prev=None
for r in docs:
    if prev is None or r['frame']-prev <= 6:
        cur.append(r)
    else:
        groups.append(cur); cur=[r]
    prev=r['frame']
if cur: groups.append(cur)
reps=[]
for g in groups:
    best=max(g, key=lambda r:(r['line_count'], r['content_box_count'], r['ocr_box_count']))
    d=dict(best); d['group_start']=g[0]['frame']; d['group_end']=g[-1]['frame']; d['group_size']=len(g)
    reps.append(d)
OUT.write_text(json.dumps({'total_frames':len(files),'document_frames':len(docs),'groups':len(groups),'representatives':reps,'all_document_frames':docs},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'total_frames':len(files),'document_frames':len(docs),'groups':len(groups),'representatives':reps},ensure_ascii=False,indent=2))
