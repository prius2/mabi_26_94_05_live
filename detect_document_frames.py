#!/usr/bin/env python3
from pathlib import Path
import json, re, sys
import cv2
import numpy as np

RUN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
CAP_DIR = RUN_DIR / 'captures'
OUT = RUN_DIR / 'document_frame_detection.json'

# Code-only document/text-heavy frame detector.
# Heuristic focuses on text-like connected components grouped into 2+ readable horizontal lines.
# It intentionally rejects speaker-only scenes where sparse labels/face/clothes edges create noise.

def analyze(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h0, w0 = img.shape[:2]
    # Remove thin borders/player overlays; keep slide/studio content.
    img = img[:int(h0*0.92), :]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Make text strokes prominent while suppressing large objects.
    # Adaptive threshold handles both white text on dark slides and dark text on light slides.
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th1 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,11)
    th2 = cv2.bitwise_not(th1)
    # Use whichever binary has more small connected components.
    candidates = []
    for th in (th1, th2):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(th, 8)
        comps=[]
        for i in range(1,n):
            x,y,ww,hh,area=stats[i]
            if area < 8: continue
            if hh < 4 or ww < 2: continue
            if hh > h*0.06 or ww > w*0.35: continue
            aspect=ww/max(hh,1)
            # character/word fragments; Korean glyphs and Latin chars tend to be compact.
            if 0.08 <= aspect <= 12 and area/(ww*hh) > 0.08:
                comps.append((x,y,ww,hh,area))
        candidates.append(comps)
    comps = max(candidates, key=len)

    # Merge nearby components into word/line blobs.
    mask = np.zeros((h,w), dtype=np.uint8)
    for x,y,ww,hh,area in comps:
        cv2.rectangle(mask,(x,y),(x+ww,y+hh),255,-1)
    # Horizontal join: characters -> words/short phrases, not whole screen.
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT,(9,2)), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    blobs=[]
    for i in range(1,n):
        x,y,ww,hh,area=stats[i]
        if area < 20: continue
        if hh < 5 or ww < 8: continue
        if hh > h*0.09 or ww > w*0.65: continue
        aspect=ww/max(hh,1)
        if 0.8 <= aspect <= 35:
            blobs.append((x,y,ww,hh,area))

    # Group blobs into horizontal text rows.
    rows=[]
    for b in sorted(blobs, key=lambda b: b[1]+b[3]/2):
        x,y,ww,hh,area=b; cy=y+hh/2
        placed=False
        for row in rows:
            if abs(cy-row['cy']) <= max(8, row['h']*0.7, hh*0.7):
                row['items'].append(b)
                ys=[i[1] for i in row['items']]; y2=[i[1]+i[3] for i in row['items']]
                row['cy']=(min(ys)+max(y2))/2; row['h']=max(y2)-min(ys)
                placed=True; break
        if not placed:
            rows.append({'cy':cy,'h':hh,'items':[b]})

    good=[]
    for row in rows:
        items=row['items']
        x1=min(i[0] for i in items); x2=max(i[0]+i[2] for i in items)
        y1=min(i[1] for i in items); y2=max(i[1]+i[3] for i in items)
        row_w=x2-x1; row_h=y2-y1
        # Require either multiple blobs on same line or one long phrase.
        if (len(items)>=2 and row_w>=w*0.12) or row_w>=w*0.22:
            good.append((x1,y1,x2,y2,len(items)))

    if good:
        x1=min(r[0] for r in good); x2=max(r[2] for r in good)
        y1=min(r[1] for r in good); y2=max(r[3] for r in good)
        width_cov=(x2-x1)/w; height_cov=(y2-y1)/h
    else:
        width_cov=height_cov=0.0
    line_count=len(good)
    comp_count=len(comps)
    blob_count=len(blobs)

    # Text-heavy/document-like: at least 2 rows of broad text, or many rows/components.
    # Reject tiny overlays: require vertical spread or many lines.
    is_doc = (
        (line_count >= 2 and width_cov >= 0.22 and height_cov >= 0.08 and blob_count >= 8) or
        (line_count >= 3 and blob_count >= 10) or
        (line_count >= 5 and comp_count >= 45)
    )
    score=line_count*3 + min(blob_count/4,10) + min(comp_count/25,8) + width_cov*3 + height_cov*2
    frame=int(re.search(r'(\d+)', path.stem).group(1))
    return {'file':path.name,'frame':frame,'is_document':bool(is_doc),'score':round(float(score),3),
            'line_count':line_count,'component_count':comp_count,'blob_count':blob_count,
            'width_cov':round(float(width_cov),3),'height_cov':round(float(height_cov),3)}

results=[]
for p in sorted(CAP_DIR.glob('frame_*.jpg')):
    r=analyze(p)
    if r: results.append(r)

docs=[r for r in results if r['is_document']]
groups=[]; cur=[]; prev=None
for r in docs:
    if prev is None or r['frame']-prev <= 4:
        cur.append(r)
    else:
        groups.append(cur); cur=[r]
    prev=r['frame']
if cur: groups.append(cur)
reps=[]
for g in groups:
    best=max(g,key=lambda r:(r['score'],r['line_count'],r['blob_count']))
    d=dict(best); d['group_start']=g[0]['frame']; d['group_end']=g[-1]['frame']; d['group_size']=len(g)
    reps.append(d)
OUT.write_text(json.dumps({'total_frames':len(results),'document_frames':len(docs),'groups':len(groups),'representatives':reps,'all_document_frames':docs},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'total_frames':len(results),'document_frames':len(docs),'groups':len(groups),'representatives':reps},ensure_ascii=False,indent=2))
