import re
import sys

BS = chr(92)


def fix_file_macro(s):
    return re.sub(BS * 2 + r"file\{([^{}]*)\}", lambda m: BS + "file{" + m.group(1).replace(BS + "_", "_") + "}", s)


for fn in sys.argv[1:]:
    s = open(fn, encoding='utf8').read()
    old = BS + r"newcommand{" + BS + r"file}[1]{" + BS + r"texttt{" + BS + "small #1}}"
    new = (BS + "usepackage{url}\n" + BS + "def" + BS + "UrlBreaks{" + BS + "do" + BS + "/" + BS + "do" + BS + "_" + BS
           + "do" + BS + "." + BS + "do" + BS + "-}\n" + BS + "urlstyle{tt}\n" + BS + "newcommand{" + BS
           + "file}[1]{{" + BS + "small" + BS + "path{#1}}}")
    s = s.replace(old, new)
    s = fix_file_macro(s)
    a = BS + "begin{center}" + BS + "small\n" + BS + "begin{tabular}{lcccc}"
    s = s.replace(a, BS + "begin{center}" + BS + "small\n" + BS + "resizebox{" + BS + "textwidth}{!}{" + BS
                  + "begin{tabular}{lcccc}")
    b = "0.16818-0.19045i$" + BS * 2 + BS + "bottomrule\n" + BS + "end{tabular}" + BS + "end{center}"
    s = s.replace(b, "0.16818-0.19045i$" + BS * 2 + BS + "bottomrule\n" + BS + "end{tabular}}" + BS + "end{center}")
    if BS + "sloppy" not in s and BS + "begin{document}" in s:
        s = s.replace(BS + "begin{document}", BS + "begin{document}\n" + BS + "sloppy", 1)
    open(fn, 'w', encoding='utf8').write(s)
    print('fixed', fn)
