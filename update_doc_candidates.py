from pathlib import Path
from datetime import datetime, timedelta
import json, re
run=Path('/home/tjseh0091/youtube-live-summary-20260425-142308')
summary=run/'summary.md'
data=json.loads((run/'document_ocr_detection.json').read_text(encoding='utf-8'))
text=summary.read_text(encoding='utf-8')
existing=set(int(m.group(1)) for m in re.finditer(r'frame_(\d{6})\.jpg', text))
selected=[]
for r in data['representatives']:
    group_frames=set(range(r['group_start'], r['group_end']+1))
    if group_frames & existing:
        continue
    if r['content_box_count'] >= 10 and r['line_count'] >= 4:
        selected.append(r)
selected=selected[:3]
start=datetime(2026,4,25,14,23,8)
def time_for_frame(n):
    return (start + timedelta(seconds=(n-1)*10)).strftime('%H:%M')
rows=[]
for r in selected:
    path=f"captures/{r['file']}"
    if path in text:
        continue
    rows.append(f"| {time_for_frame(r['frame'])} | 화면/문서 후보 | 코드 OCR 기반 자동 감지: 캡처 화면에서 텍스트 {r['line_count']}줄, 콘텐츠 박스 {r['content_box_count']}개가 감지되어 문서/PPT/글자 많은 화면 후보로 추가. 기존 타임라인의 동일 프레임/동일 구간과 중복되지 않는 신규 후보만 반영 | `{path}`<br><img src=\"{path}\" alt=\"코드 OCR 자동 감지 문서/텍스트 화면 후보 frame {r['frame']}\" width=\"240\"> |")
if rows and '## Q&A 주제별 정리' in text:
    text=text.replace('\n\n## Q&A 주제별 정리', '\n'+'\n'.join(rows)+'\n\n## Q&A 주제별 정리', 1)
report_lines=['## 코드 기반 문서 이미지 점검','']
report_lines.append(f"- 검사 방식: RapidOCR 코드 실행으로 `captures/frame_*.jpg` 전체 {data['total_frames']}장을 스캔하고, ON AIR 등 단순 오버레이를 제외한 OCR 텍스트 줄/박스 수로 문서·PPT·글자 많은 화면 후보를 판정.")
report_lines.append(f"- 감지 결과: 문서 후보 프레임 {data['document_frames']}장, 연속 구간 {data['groups']}개.")
if selected:
    for r in selected:
        path=f"captures/{r['file']}"
        report_lines.append(f"- 신규 반영 후보: `{path}` — OCR 기준 {r['line_count']}줄 / 콘텐츠 박스 {r['content_box_count']}개.")
else:
    report_lines.append('- 신규 반영 후보: 기존 타임라인/캡처 섹션과 중복되지 않는 고신뢰 신규 후보 없음.')
report_lines.append('- 주의: 이 점검은 LLM 시각 분석이 아니라 코드 OCR/레이아웃 휴리스틱 기반이라, 실제 텍스트 의미는 요약하지 않고 후보 화면만 증거 이미지로 추가했다.')
report='\n'.join(report_lines)+'\n\n'
if '## 코드 기반 문서 이미지 점검' not in text:
    text=text.replace('\n## 주요 화면 캡처', '\n'+report+'## 주요 화면 캡처', 1)
if selected:
    insert=''
    for r in selected:
        path=f"captures/{r['file']}"
        if f'### 코드 OCR 자동 감지 문서 후보 {r["frame"]}' not in text:
            insert += f"\n### 코드 OCR 자동 감지 문서 후보 {r['frame']}\n<img src=\"{path}\" alt=\"코드 OCR 자동 감지 문서/텍스트 화면 후보 frame {r['frame']}\" width=\"720\">\n"
    if insert:
        text=text.replace('\n## 최종 정리', insert+'\n## 최종 정리', 1)
summary.write_text(text, encoding='utf-8')
print(json.dumps({'selected': selected}, ensure_ascii=False, indent=2))
