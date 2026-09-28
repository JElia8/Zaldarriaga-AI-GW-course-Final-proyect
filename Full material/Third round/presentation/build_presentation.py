"""Build presentation.html (self-contained, figures embedded as base64 PNG).

Layout, style and the interactive simulations (collapse animation, model timeline, live wave packet) are taken from
the second-round template (read only); the slide content is in slides_new.html; figures come from ../figures."""
import base64
import io
import os
import re
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, '..', '..', 'Second round', 'presentation', 'presentation_template.html')
FIGS = os.path.join(HERE, '..', 'figures')

old = open(OLD, encoding='utf8').read()
new = open(os.path.join(HERE, 'slides_new.html'), encoding='utf8').read()

head = old[:old.index('<!-- ============================================================== 1 -->')]
tail = old[old.index('<div id="bar"></div>'):]
old_sec = {m.group(1): m.group(0) for m in re.finditer(r'<section class="slide" data-title="([^"]+)">.*?</section>', old, re.S)}
new_sec = {m.group(1): m.group(0) for m in re.finditer(r'<section class="slide" data-title="([^"]+)">.*?</section>', new, re.S)}

live = old_sec['Live scattering'].replace(
    'Check made offline: for R = 2.2M the late tail decays like\n  exp(-0.035 t), exactly the imaginary part of the least-damped QNM 0.4211 - 0.0350i.',
    'For R = 2.2M the late signal decays like exp(-0.035 t), the imaginary part of the trapped QNM 0.4211 - 0.0350i.')
options = old_sec['Four options'].replace('<h2>How to get a background for scattering <span class="tag">the options</span></h2>',
                                          '<h2>Which shell? <span class="tag">the options for a static background</span></h2>')

order = [new_sec['Title'], new_sec['Line of thought'], old_sec['The setup'], old_sec['Thin shells in GR'],
         old_sec['The collapse problem'], options, new_sec['Master equations'], new_sec['Junction conditions'],
         new_sec['Reflection and transmission'], new_sec['Two limits'], new_sec['Quasinormal modes'],
         new_sec['Echoes in the time domain'], live, new_sec['The instability'], new_sec['Back to the background'],
         new_sec['Validation'], new_sec['Take-home'], new_sec['Backup: multipoles and EOS']]
html = head + '\n\n'.join(order) + '\n\n' + tail


def data_uri(name, maxw=1500):
    im = Image.open(os.path.join(FIGS, name + '.png')).convert('RGB')
    if im.width > maxw:
        im = im.resize((maxw, int(im.height * maxw / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


names = sorted(set(re.findall(r'\{\{IMG:([A-Za-z0-9_]+)\}\}', html)))
for n in names:
    html = html.replace('{{IMG:%s}}' % n, data_uri(n))
out = os.path.join(HERE, 'presentation.html')
open(out, 'w', encoding='utf8').write(html)
print('%d slides, %d figures; %s (%.1f MB)' % (len(order), len(names), out, os.path.getsize(out) / 1e6))
