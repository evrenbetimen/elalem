# Elalem için "ortaya karışık" bir KOD veri seti hazırlar: HTML/CSS/JS + Python, Java, C, SQL vb.
# Yalnızca kod cevabı olan örnekler alınır; düz yazı açıklamalar atlanır.
# Yalnızca Python standart kütüphanesi kullanır; kaynağı GitHub'dan indirip Elalem sohbet formatına çevirir.
# Kullanım: python3 karisik_veri_hazirla.py            -> veri/elle/karisik/ altına 2 dosya yazar
#           python3 karisik_veri_hazirla.py hedef/klasor
import json
import os
import random
import re
import sys
import urllib.request

KAYNAK = "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json"
WEB = re.compile(r"\b(HTML|CSS|JavaScript|web ?page|website|DOM|jQuery)\b", re.I)
DUZ_YAZI = re.compile(r"^(The|This|It|A|An|Yes|No|One)\b")


def indir(url):
    print("indiriliyor:", url, file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as y:
        return json.loads(y.read().decode("utf-8-sig"))


def ornek(istek, cevap):
    return f"<|kullanici|> {' '.join(istek.split())}\n<|elalem|>\n{cevap.strip()}\n<|son|>\n"


def kod_mu(cevap):
    """Cevap kod mu, düz yazı mı? Kodda parantez, süslü parantez, = ve ; gibi işaretler yoğundur."""
    c = cevap.strip()
    if len(c) < 15 or DUZ_YAZI.match(c):
        return False
    return sum(ch in "(){}[]=;<>:_" for ch in c) / len(c) > 0.03


def ayir(veri):
    """CodeAlpaca'yı web kodu ve diğer diller olarak ikiye ayırır; düz yazı cevapları atar."""
    web, genel = [], []
    for x in veri:
        if not kod_mu(x["output"]):
            continue
        (web if WEB.search(x["instruction"]) else genel).append(ornek(x["instruction"] + " " + x["input"], x["output"]))
    return web, genel


if __name__ == "__main__":
    hedef = sys.argv[1] if len(sys.argv) > 1 else os.path.join("veri", "elle", "karisik")
    os.makedirs(hedef, exist_ok=True)
    r = random.Random(1)
    web, genel = ayir(indir(KAYNAK))
    for ad, ornekler in {"web_kod_codealpaca.txt": web, "genel_kod_codealpaca.txt": genel}.items():
        r.shuffle(ornekler)
        yol = os.path.join(hedef, ad)
        with open(yol, "w", encoding="utf-8") as f:
            f.write("".join(ornekler))
        print(f"{yol}: {len(ornekler)} örnek, {os.path.getsize(yol) / 1e6:.2f} MB", file=sys.stderr)
