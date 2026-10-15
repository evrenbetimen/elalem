# Elalem için sentetik form verisi üretir (istek -> form eşleşmesi öğretir).
# Kullanım: python3 sentetik_form_uret.py > veri/elle/sentetik_formlar.txt
import random

ALANLAR = {
    # anahtar: (prompt'taki adlar, id, tip, etiket)
    "kullanici": (["kullanıcı adı"], "kullanici", "text", "Kullanıcı adı"),
    "email": (["e-posta", "email", "e-posta adresi"], "email", "email", "E-posta"),
    "sifre": (["şifre", "parola"], "sifre", "password", "Şifre"),
    "adres": (["adres"], "adres", "textarea", "Adres"),
    "telefon": (["telefon", "telefon numarası"], "telefon", "tel", "Telefon"),
    "ad": (["ad soyad", "isim"], "ad", "text", "Ad Soyad"),
    "dogum": (["doğum tarihi"], "dogum", "date", "Doğum tarihi"),
    "sehir": (["şehir"], "sehir", "text", "Şehir"),
    "mesaj": (["mesaj"], "mesaj", "textarea", "Mesaj"),
    "yas": (["yaş"], "yas", "number", "Yaş"),
    "site": (["web sitesi"], "site", "url", "Web sitesi"),
}
EN = {"kullanici": "username", "email": "email", "sifre": "password", "adres": "address",
      "telefon": "phone", "ad": "full name", "dogum": "birth date", "sehir": "city",
      "mesaj": "message", "yas": "age", "site": "website"}
TURLER = {"giriş": ("giris", "Giriş Yap"), "kayıt": ("kayit", "Kayıt Ol"), "iletişim": ("iletisim", "Gönder"),
          "adres": ("adres", "Kaydet"), "sipariş": ("siparis", "Siparişi Tamamla"), "": ("form", "Gönder")}
SITELER = ["example.com", "google.com", "github.com", "w3schools.com", "elalem.com"]


def liste(adlar):
    return adlar[0] if len(adlar) == 1 else ", ".join(adlar[:-1]) + " ve " + adlar[-1]


def alan_html(k):
    _, i, tip, etiket = ALANLAR[k]
    if tip == "textarea":
        return [f'  <label>{etiket} <textarea name="{i}"></textarea></label>']
    return [f'  <label>{etiket} <input type="{tip}" name="{i}"></label>']


def ornek(r):
    tur = r.choice(list(TURLER))
    sinif, buton = TURLER[tur]
    alanlar = r.sample(list(ALANLAR), r.choice([1, 2, 2, 3, 3, 3, 4]))
    site = r.choice(SITELER) if r.random() < 0.25 else None
    if r.random() < 0.12:
        p = f"Make a {sinif if tur else 'simple'} form with {liste([EN[a] for a in alanlar]).replace(' ve ', ' and ')}."
        if site:
            p += f" Add a link to {site}."
    else:
        adlar = liste([r.choice(ALANLAR[a][0]) for a in alanlar])
        ad = f"{tur} formu" if tur else "form"
        kalip = r.choice([
            "Bir {ad} yap. {l} olsun sadece", "Bir {ad} yap. {l} olsun", "{l} alanları olan bir {ad} yap.",
            "Sadece {l} olan bir {ad} yap.", "{l} soran bir {ad} yap.", "Bir {ad} yap: {l}",
            "Bana {l} içeren bir {ad} yazar mısın?",
        ])
        p = kalip.format(ad=ad, l=adlar)
        p = p[0].upper() + p[1:]
        if site:
            p += f" link de olsun {site}'a"
    html = [f'<form class="{sinif}-form" method="post">']
    for a in alanlar:
        html += alan_html(a)
    html.append(f'  <button type="submit">{buton}</button>')
    html.append("</form>")
    if site:
        html.append(f'<p><a href="https://{site}">{site}</a></p>')
    return f"<|kullanici|> {p}\n<|elalem|>\n" + "\n".join(html) + "\n<|son|>\n"


if __name__ == "__main__":
    r = random.Random(42)
    print("".join(ornek(r) for _ in range(4000)), end="")
