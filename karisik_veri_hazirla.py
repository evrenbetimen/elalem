# Elalem için "ortaya karışık" bir veri seti hazırlar: web kodu + Türkçe soru-cevap + genel İngilizce talimat.
# Yalnızca Python standart kütüphanesi kullanır; kaynakları GitHub'dan indirip Elalem sohbet formatına çevirir.
# Kullanım: python3 karisik_veri_hazirla.py            -> veri/elle/karisik/ altına 3 dosya yazar
#           python3 karisik_veri_hazirla.py hedef/klasor
import json
import os
import random
import re
import sys
import urllib.request

KAYNAKLAR = {
    "codealpaca": "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json",
    "tquad_train": "https://raw.githubusercontent.com/TQuad/turkish-nlp-qa-dataset/master/train-v0.1.json",
    "tquad_dev": "https://raw.githubusercontent.com/TQuad/turkish-nlp-qa-dataset/master/dev-v0.1.json",
    "alpaca": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
}
WEB = re.compile(r"\b(HTML|CSS|JavaScript|web ?page|website|DOM|jQuery)\b", re.I)


def indir(url):
    print("indiriliyor:", url, file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as y:
        return json.loads(y.read().decode("utf-8-sig"))


def ornek(istek, cevap):
    return f"<|kullanici|> {' '.join(istek.split())}\n<|elalem|>\n{cevap.strip()}\n<|son|>\n"


def codealpaca_web(veri):
    """CodeAlpaca'dan HTML/CSS/JS ile ilgili istek -> kod örnekleri (İngilizce)."""
    out = []
    for x in veri:
        if WEB.search(x["instruction"]) and x["output"].strip():
            out.append(ornek(x["instruction"] + " " + x["input"], x["output"]))
    return out


def cumle_bul(metin, bas):
    """Cevabın geçtiği cümleyi döndürür (düz, kodsuz anlatım cevabı)."""
    sol = max(metin.rfind(". ", 0, bas), metin.rfind("? ", 0, bas), metin.rfind("! ", 0, bas))
    sol = 0 if sol < 0 else sol + 2
    sag = min([i for i in (metin.find(". ", bas), metin.find("? ", bas), metin.find("! ", bas)) if i >= 0] or [len(metin) - 1])
    return metin[sol:sag + 1].strip()


def tquad(*veriler):
    """TQuad'dan Türkçe soru -> cevabın geçtiği tam cümle."""
    out = []
    for veri in veriler:
        for makale in veri["data"]:
            for p in makale["paragraphs"]:
                for qa in p["qas"]:
                    if not qa["answers"]:
                        continue
                    a = qa["answers"][0]
                    bas = int(a["answer_start"])
                    if p["context"][bas:bas + len(a["text"])] != a["text"]:
                        bas = p["context"].find(a["text"])
                        if bas < 0:
                            continue
                    cumle = cumle_bul(p["context"], bas)
                    if 20 <= len(cumle) <= 400:
                        out.append(ornek(qa["question"].replace(" ?", "?"), cumle))
    return out


def alpaca_kisa(veri, adet, r):
    """Stanford Alpaca'dan kodsuz, kısa genel İngilizce talimatlar."""
    secilen = [x for x in veri if not x["input"] and 40 <= len(x["output"]) <= 500 and "```" not in x["output"]]
    return [ornek(x["instruction"], x["output"]) for x in r.sample(secilen, min(adet, len(secilen)))]


if __name__ == "__main__":
    hedef = sys.argv[1] if len(sys.argv) > 1 else os.path.join("veri", "elle", "karisik")
    os.makedirs(hedef, exist_ok=True)
    r = random.Random(1)
    parcalar = {
        "web_kod_codealpaca.txt": codealpaca_web(indir(KAYNAKLAR["codealpaca"])),
        "turkce_soru_cevap_tquad.txt": tquad(indir(KAYNAKLAR["tquad_train"]), indir(KAYNAKLAR["tquad_dev"])),
        "genel_ingilizce_alpaca.txt": alpaca_kisa(indir(KAYNAKLAR["alpaca"]), 4000, r),
    }
    for ad, ornekler in parcalar.items():
        r.shuffle(ornekler)
        yol = os.path.join(hedef, ad)
        with open(yol, "w", encoding="utf-8") as f:
            f.write("".join(ornekler))
        print(f"{yol}: {len(ornekler)} örnek, {os.path.getsize(yol) / 1e6:.2f} MB", file=sys.stderr)
