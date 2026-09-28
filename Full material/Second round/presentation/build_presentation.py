"""Build presentation.html from presentation_template.html, embedding figures as base64 PNG
(downscaled to at most 1500 px wide) so that the HTML file is self-contained."""
import base64
import io
import os
import re
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = [os.path.join(HERE, '..', '..', 'First round', 'figures'), os.path.join(HERE, '..', 'figures_round2')]


def data_uri(name, maxw=1500):
    for d in SRC:
        p = os.path.join(d, name + '.png')
        if os.path.exists(p):
            im = Image.open(p).convert('RGB')
            if im.width > maxw:
                im = im.resize((maxw, int(im.height*maxw/im.width)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format='PNG', optimize=True)
            return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    raise FileNotFoundError(name)


tpl = open(os.path.join(HERE, 'presentation_template.html'), encoding='utf8').read()
names = sorted(set(re.findall(r'\{\{IMG:([A-Za-z0-9_]+)\}\}', tpl)))
for n in names:
    tpl = tpl.replace('{{IMG:%s}}' % n, data_uri(n))
out = os.path.join(HERE, 'presentation.html')
open(out, 'w', encoding='utf8').write(tpl)
print('embedded %d figures; %s (%.1f MB)' % (len(names), out, os.path.getsize(out)/1e6))
