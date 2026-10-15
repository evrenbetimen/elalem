# Elalem için sentetik sayfa verisi üretir (istek -> sayfa türü eşleşmesi öğretir).
# Form verisinin yanına konur ki model her isteğe form ile cevap vermesin.
# Kullanım: python3 sentetik_sayfa_uret.py > veri/elle/sentetik_sayfalar.txt
#           python3 sentetik_sayfa_uret.py 2000 > ...   (örnek sayısı, varsayılan 3000)
import random
import sys

ADLAR = ["Ayşe Yılmaz", "Mehmet Demir", "Zeynep Kaya", "Ali Çelik", "Elif Şahin", "Can Öztürk",
         "Deniz Aydın", "Selin Arslan", "Emre Koç", "Merve Doğan", "Evren Kurt", "Ece Polat"]
MESLEKLER = ["Web Geliştirici", "Grafik Tasarımcı", "Öğretmen", "Fotoğrafçı", "Yazılım Mühendisi",
             "Mimar", "Öğrenci", "Doktor", "Müzisyen", "Yazar"]
SEHIRLER = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Eskişehir", "Trabzon"]
ILGILER = ["HTML", "CSS", "JavaScript", "Python", "fotoğrafçılık", "kitap okumak", "yürüyüş",
           "müzik", "satranç", "resim", "yemek yapmak", "seyahat"]
URUNLER = [("Kablosuz Kulaklık", "799 TL"), ("Akıllı Saat", "1.499 TL"), ("Sırt Çantası", "349 TL"),
           ("Kahve Makinesi", "2.199 TL"), ("Masa Lambası", "259 TL"), ("Klavye", "649 TL")]
MENU = ["Ana Sayfa", "Hakkımda", "Projeler", "Blog", "İletişim", "Hizmetler", "Ürünler"]
KONULAR = ["HTML öğrenmeye nereden başlamalı?", "CSS ile renkler", "İlk JavaScript kodum",
           "Neden Python?", "Duyarlı tasarım nedir?", "Flexbox ile hizalama"]


def kimlik(r):
    ad = r.choice(ADLAR)
    return ad, r.choice(MESLEKLER), r.choice(SEHIRLER), r.sample(ILGILER, 3)


def sayfa(baslik, govde, stil=None):
    s = ["<!DOCTYPE html>", '<html lang="tr">', "<head>", '  <meta charset="UTF-8">',
         f"  <title>{baslik}</title>"]
    if stil:
        s.append(f"  <style>{stil}</style>")
    return "\n".join(s + ["</head>", "<body>"] + govde + ["</body>", "</html>"])


def kisisel(r):
    ad, meslek, sehir, ilgi = kimlik(r)
    p = r.choice(["Örnek bir kişisel sayfa tasarla", "Bir kişisel sayfa yap", "Kişisel web sayfası oluştur",
                  "Benim için kişisel bir sayfa yap", "Bir kişisel sayfa tasarla, adım {ad}",
                  "{ad} için kişisel sayfa yap, mesleği {meslek}", "Make a personal page for {ad}"])
    p = p.replace("{ad}", ad).replace("{meslek}", meslek.lower())
    govde = ["  <header>", f"    <h1>{ad}</h1>", f"    <p>{meslek} - {sehir}</p>", "  </header>",
             "  <section>", "    <h2>Hakkımda</h2>",
             f"    <p>Merhaba, ben {ad.split()[0]}. {sehir} şehrinde {meslek.lower()} olarak çalışıyorum.</p>",
             "  </section>", "  <section>", "    <h2>İlgi Alanlarım</h2>", "    <ul>"]
    govde += [f"      <li>{i}</li>" for i in ilgi] + ["    </ul>", "  </section>"]
    if r.random() < 0.6:
        govde += ["  <footer>", '    <a href="mailto:bilgi@example.com">Bana yaz</a>', "  </footer>"]
    stil = "body{font-family:sans-serif;max-width:600px;margin:auto}" if r.random() < 0.5 else None
    return p, sayfa(ad, govde, stil)


def hakkimda(r):
    ad, meslek, sehir, ilgi = kimlik(r)
    p = r.choice(["Bir hakkımda bölümü yap", "Hakkımda kısmı yaz", "Kendimi tanıtan bir bölüm yap"])
    return p, "\n".join(['<section class="hakkimda">', "  <h2>Hakkımda</h2>",
                         f"  <p>Ben {ad}. {sehir} şehrinde yaşıyorum, mesleğim {meslek.lower()}.</p>",
                         f"  <p>Boş zamanlarımda {ilgi[0]} ve {ilgi[1]} ile ilgileniyorum.</p>", "</section>"])


def portfoy(r):
    ad, meslek, _, ilgi = kimlik(r)
    p = r.choice(["Bir portföy sayfası yap", "Portföy sayfası tasarla", "Projelerimi gösteren bir sayfa yap"])
    govde = [f"  <h1>{ad} - Portföy</h1>", f"  <p>{meslek}</p>", '  <div class="projeler">']
    for i, konu in enumerate(r.sample(ILGILER, r.choice([2, 3])), 1):
        govde += ['    <div class="proje">', f"      <h3>Proje {i}</h3>",
                  f"      <p>Konusu: {konu}.</p>", "    </div>"]
    govde.append("  </div>")
    return p, sayfa(f"{ad} Portföy", govde, ".projeler{display:flex;gap:16px}")


def navbar(r):
    p = r.choice(["Bir menü yap", "Navigasyon çubuğu yap", "Bir navbar oluştur", "Üst menü tasarla"])
    ogeler = r.sample(MENU, r.choice([3, 4]))
    return p, "\n".join(['<nav class="navbar">', "  <ul>"] +
                        [f'    <li><a href="#">{o}</a></li>' for o in ogeler] + ["  </ul>", "</nav>"])


def urun_karti(r):
    ad, fiyat = r.choice(URUNLER)
    p = r.choice(["Bir ürün kartı yap", "Ürün kartı tasarla: {urun}", "{urun} için bir kart yap"]).replace("{urun}", ad)
    return p, "\n".join(['<div class="kart">', f"  <h3>{ad}</h3>", f'  <p class="fiyat">{fiyat}</p>',
                         '  <button type="button">Sepete Ekle</button>', "</div>"])


def blog(r):
    konu = r.choice(KONULAR)
    p = r.choice(["Bir blog yazısı sayfası yap", "Blog yazısı tasarla: {k}", "{k} başlıklı bir blog yazısı yap"]).replace("{k}", konu)
    return p, "\n".join(["<article>", f"  <h1>{konu}</h1>", f'  <p class="tarih">{r.randint(1, 28)} Ekim 2026</p>',
                         "  <p>Bu yazıda konuya kısa bir giriş yapıyoruz.</p>", "</article>"])


def footer(r):
    ad = r.choice(ADLAR)
    p = r.choice(["Bir alt bilgi yap", "Footer yap", "Sayfa sonu bölümü tasarla"])
    return p, "\n".join(["<footer>", f"  <p>&copy; 2026 {ad}</p>", '  <a href="mailto:bilgi@example.com">İletişim</a>',
                         "</footer>"])


TURLER = [(kisisel, 4), (hakkimda, 2), (portfoy, 2), (navbar, 2), (urun_karti, 2), (blog, 2), (footer, 1)]


def ornek(r):
    f = r.choices([t for t, _ in TURLER], [w for _, w in TURLER])[0]
    p, html = f(r)
    if r.random() < 0.5:
        p += "."
    return f"<|kullanici|> {p}\n<|elalem|>\n{html}\n<|son|>\n"


if __name__ == "__main__":
    adet = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    r = random.Random(7)
    print("".join(ornek(r) for _ in range(adet)), end="")
