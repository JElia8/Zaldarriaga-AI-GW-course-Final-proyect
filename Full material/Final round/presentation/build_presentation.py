"""Build presentation.html from deck_source.html: embeds the figures of ../figures as base64 PNG and the
retardance data (pol_data.json), so the output is a single self-contained file (KaTeX is loaded from a CDN).
Edit deck_source.html and re-run:  python build_presentation.py"""
import base64
import io
import os
import re
import shutil
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, '..', 'figures')


def data_uri(name, maxw=1500):
    im = Image.open(os.path.join(FIGS, name + '.png')).convert('RGB')
    if im.width > maxw:
        im = im.resize((maxw, int(im.height * maxw / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


html = open(os.path.join(HERE, 'deck_source.html'), encoding='utf8').read()
names = sorted(set(re.findall(r'\{\{IMG:([A-Za-z0-9_]+)\}\}', html)))
for n in names:
    html = html.replace('{{IMG:%s}}' % n, data_uri(n))
html = html.replace('{{DATA:pol}}', open(os.path.join(HERE, 'pol_data.json'), encoding='utf8').read())
out = os.path.join(HERE, 'presentation.html')
open(out, 'w', encoding='utf8').write(html)
shutil.copy(out, os.path.join(HERE, '..', '3_presentation.html'))
n_slides = len(re.findall(r'<section class="slide"', html))
print('%d slides, %d figures; %s (%.1f MB)' % (n_slides, len(names), out, os.path.getsize(out) / 1e6))
