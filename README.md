# Elalem

Elalem, hiçbir hazır yapay zeka kütüphanesi kullanmadan **sıfırdan, saf PyTorch matris matematiğiyle** yazılmış, karakter seviyesinde çalışan küçük bir GPT dil modelidir. Modeli eğitmek, ona kod yazdırmak ve ürettiği kodu denetlemek için kendi web arayüzüyle birlikte gelir. Arayüz bilgisayarında `127.0.0.1:8080` adresinde çalışır.

Her şey tek bir dosyadadır: [`main.py`](main.py) (~2.200 satır). İhtiyaç duyduğu tek dış paket `torch`'tur. Geri kalan her şey Python standart kütüphanesidir.

---

## İçindekiler

1. [Hızlı başlangıç](#hızlı-başlangıç)
2. [Neler yapabilir, neler yapamaz](#neler-yapabilir-neler-yapamaz)
3. [Teknik özellikler (özet)](#teknik-özellikler-özet)
4. [Mimari](#mimari)
5. [Model boyutları](#model-boyutları)
6. [Eğitim](#eğitim)
7. [Eğitim verisi](#eğitim-verisi)
8. [Metin üretimi](#metin-üretimi)
9. [QA/QC kod doğrulayıcı](#qaqc-kod-doğrulayıcı)
10. [Kontrol paneli](#kontrol-paneli)
11. [HTTP API](#http-api)
12. [Komut satırı seçenekleri](#komut-satırı-seçenekleri)
13. [Dosyalar](#dosyalar)
14. [Güvenlik](#güvenlik)
15. [Parametreleri büyütmek](#parametreleri-büyütmek)
16. [Sınırlamalar](#sınırlamalar)

---

## Hızlı başlangıç

```bash
pip3 install torch
python3 main.py --open                 # küçük model, paneli tarayıcıda açar
python3 main.py --boyut orta --open    # daha büyük model (önerilen)
python3 main.py --selftest             # uçtan uca kısa test
```

Panel açıldıktan sonra:

1. **Eğitim Verisi** bölümünden veri ekle (isteğe bağlı).
2. **Adım sayısı** kutusuna 3000-5000 yaz ve **▶ Eğitimi Başlat**'a bas.
3. Loss grafiği düştükten sonra sağdaki kutuya bir istek yaz, örneğin "Bir giriş formu yap.", ve **✨ Kod Üret**'e bas.

---

## Neler yapabilir, neler yapamaz

| Yapabilir | Yapamaz |
|---|---|
| Eğitim verisindeki kalıplara benzeyen HTML, CSS, JS, Python kodu ve kısa Türkçe/İngilizce cevaplar üretmek | ChatGPT gibi hiç görmediği istekleri anlayıp yorumlamak |
| Ürettiği kodu otomatik olarak doğrulayıp hata raporlamak | Ezberlediği örneklerin dışına çıkan karmaşık istekleri doğru birleştirmek (yeterli veri ve eğitim olmadan) |
| Kendi verinle (dosya, web sayfası, yapıştırılan metin) eğitilmek | Büyük modellerin bilgisine sahip olmak: 2-14 milyon parametre vs. yüz milyarlarca |

Elalem bir **öğrenme ve deney projesidir**. Gördüğü karakter dizilerini taklit eder. Ne kadar çok ve ne kadar istediğin biçimde veri görürse o kadar iyi sonuç verir.

---

## Teknik özellikler (özet)

| Özellik | Değer |
|---|---|
| Model türü | Decoder-only Transformer (GPT), karakter seviyesi |
| Bağımlılık | Sadece `torch` (`torch`, `torch.nn`, `torch.nn.functional`) + Python standart kütüphanesi |
| Yasaklı/kullanılmayan | Hugging Face, Transformers, Tiktoken, LangChain, OpenAI, Ollama vb. hiçbiri |
| Tokenizer | Kendi karakter tokenizer'ı, **131 karakterlik sözlük** |
| Dikkat | Causal (üçgen maskeli) Multi-Head Self-Attention, elle yazılmış |
| Blok yapısı | Pre-LayerNorm + residual bağlantılar, FFN: Linear → GELU → Linear (4× genişlik) |
| Parametre | Küçük **1,83 M** · Orta **4,90 M** · Büyük **14,43 M** · Dev **38,14 M** · ya da istediğin kadar (`--n-embd`, `--n-layer`…) |
| Bağlam (block_size) | Küçük 128 · Orta/Büyük 512 karakter |
| Optimizer / kayıp | AdamW + `nn.CrossEntropyLoss` |
| Cihaz önceliği | CUDA → Apple `mps` (M serisi GPU) → CPU |
| Sunucu | Python `http.server.ThreadingHTTPServer`, sadece `127.0.0.1` |
| Arayüz | Tek sayfa, koyu tema, responsive, harici kütüphane yok (grafik `<canvas>` ile elle çiziliyor) |
| Kod doğrulama | Kural tabanlı QA/QC: HTML etiket yığını, CSS bildirimleri, Python `ast.parse`, parantez/tırnak dengesi |
| Kalıcılık | Ağırlıklar `.pt` dosyasına kaydedilir, açılışta otomatik yüklenir |

---

## Mimari

```
metin ──► ElalemTokenizer.encode ──► idx [B, T]
                                          │
               token_emb [V, C] ──────────┤
               pos_emb [block_size, C] ───┤  (toplama)
                                          ▼
                                     x [B, T, C]
                                          │
                ┌─────────────── N × TransformerBlock ───────────────┐
                │  x = x + CausalSelfAttention(LayerNorm(x))          │
                │  x = x + FeedForward(LayerNorm(x))                  │
                └─────────────────────────────────────────────────────┘
                                          │
                                   LayerNorm (ln_f)
                                          │
                        lm_head (token_emb ile ağırlık paylaşımı)
                                          ▼
                                   logits [B, T, V]
```

Boyut kısaltmaları (kodda her satırda yorum olarak yazılıdır): `B` batch, `T` dizi uzunluğu, `C` gömme boyutu (`n_embd`), `H` kafa sayısı, `hs = C / H` kafa boyutu, `V` sözlük boyutu.

### 1. ElalemTokenizer

- **Karakter tabanlıdır.** Her karakter bir token'dır.
- **Alfabe:** ASCII yazdırılabilir karakterler (32-126: HTML `< > /`, CSS `{ } : ;`, kod sembolleri `= ( ) [ ]` vb.) + `\n`, `\t` + Türkçe harfler (`çğıöşüÇĞİÖŞÜâîûÂÎÛ`) + sık semboller (`© € ₺ “ ” ‘ ’ – — … • → ← ✓ ✗`) + gömülü veride geçen diğer karakterler. Toplam **131 karakter**.
- **id 0 = UNK.** Bilinmeyen karakterler buna eşlenir. `decode` bunları `?` olarak yazar.
- **Sözlük sabittir.** Sadece gömülü veriden türetilir. Dışarıdan eklenen verideki sözlük dışı karakterler `temizle()` ile eşdeğerine çevrilir (ör. bölünmez boşluk → boşluk, `×` → `x`) ya da atılır. Böylece veri eklemek modelin boyutunu değiştirmez ve eğitilmiş ağırlıklar geçerli kalır.
- `encode(metin) -> [int]`, `decode([int]) -> metin`.

### 2. CausalSelfAttention

- Tek bir `Linear(C, 3C)` ile Q, K, V birlikte hesaplanır, `H` kafaya bölünür: `[B, H, T, hs]`.
- Dikkat skoru: `softmax(Q·Kᵀ / √hs)`.
- Alt üçgen maske (`torch.tril`) gelecekteki token'ları `-inf` ile kapatır, yani model sadece geçmişe bakar.
- Attention dropout + çıkış projeksiyonu `Linear(C, C)` + residual dropout.

### 3. FeedForward

`Linear(C, 4C) → GELU → Linear(4C, C) → Dropout`

### 4. TransformerBlock

Pre-LayerNorm düzeni: `x = x + Attn(LN(x))`, ardından `x = x + FFN(LN(x))`.

### 5. ElalemGPT

- Token embedding `[V, C]` + öğrenilen pozisyon embedding `[block_size, C]`.
- `N` adet `TransformerBlock`, son `LayerNorm`, `lm_head: Linear(C, V)`.
- **Weight tying:** `lm_head.weight` ile `token_emb.weight` aynı tensördür (daha az parametre, daha iyi genelleme).
- Ağırlık başlatma: `Linear` ve `Embedding` için `N(0, 0.02)`, bias'lar sıfır.
- `targets` verilirse `nn.CrossEntropyLoss` ile kayıp hesaplanır (`[B*T, V]` ile `[B*T]`).

---

## Model boyutları

`--boyut` ile seçilir. Her boyutun ağırlıkları ayrı dosyaya kaydedilir, yani birbirinin üzerine yazmaz.

| Boyut | `n_embd` | `n_head` | `n_layer` | `block_size` | `batch_size` | Öğrenme oranı | Parametre | Ağırlık dosyası |
|---|---|---|---|---|---|---|---|---|
| `kucuk` (varsayılan) | 192 | 6 | 4 | 128 | 32 | 3e-3 | **1.827.264** | `elalem_checkpoint.pt` |
| `orta` | 256 | 8 | 6 | 512 | 24 | 1e-3 | **4.899.072** | `elalem_checkpoint_256x6x512.pt` |
| `buyuk` | 384 | 6 | 8 | 512 | 32 | 6e-4 | **14.434.176** | `elalem_checkpoint_384x8x512.pt` |
| `dev` | 512 | 8 | 12 | 512 | 16 | 3e-4 | **38.140.416** | `elalem_checkpoint_512x12x512.pt` |

Ortak ayarlar: dropout 0.1, `eval_interval` 50 adım, `eval_iters` 10 batch, seed 1337.

**Hangisini seçmeli?** Sadece denemek için `kucuk`. Web'den veri indirdiysen ya da form gibi uzun çıktılar istiyorsan `orta`. Küçük model aynı anda sadece 128 karakter görür, uzun bir form yazarken isteği unutur. Çok veri varsa `buyuk`, güçlü bir Mac ve bol veriyle `dev`.

---

## Eğitim

| Ayar | Değer |
|---|---|
| Optimizer | `torch.optim.AdamW`, `betas=(0.9, 0.99)`, `weight_decay=0.01` |
| Kayıp | `nn.CrossEntropyLoss` (bir sonraki karakteri tahmin) |
| Gradyan kırpma | `clip_grad_norm_(..., 1.0)` |
| Batch üretimi | `get_batch`: veriden rastgele `block_size` uzunluğunda `X`, bir karakter kaydırılmış `Y` |
| Train/val bölünmesi | Her kaynak için %90 / %10 |
| Değerlendirme | Her 50 adımda ve son adımda, 10 batch ortalaması (train + val) |
| Çalışma şekli | Arka plan thread'i, panel her saniye durumu çeker |
| Durdurma | Panelden **■ Durdur**, mevcut adım bitince durur |
| Kayıt | Eğitim bitince ya da durdurulunca ağırlıklar otomatik kaydedilir, sonraki açılışta kaldığı adımdan devam eder |

Öğrenme oranı her eğitim başlatılışında panelden değiştirilebilir. Eğitim sırasında üretim yapılabilir; model kilidi her adım arasında serbest kalır.

---

## Eğitim verisi

### Gömülü veri

`main.py` içinde **36 elle yazılmış istek → cevap örneği** vardır: HTML sayfaları, kart, navbar, form, tablo, CSS (koyu tema, buton, grid, responsive), Python fonksiyonları ve basit API'ler, JavaScript, Türkçe/İngilizce sohbet. Örnekler üç kez (ileri, ters, ileri sırayla) tekrarlanarak ~25.000 karakterlik bir metne dönüştürülür.

### Sohbet biçimi

Model, istekle cevabı şu işaretlerle ayırt etmeyi öğrenir. Bunlar özel token değil, düz karakter dizileridir:

```
<|kullanici|> Bir giriş formu yap.
<|elalem|>
<form ...> ... </form>
<|son|>
```

Kendi örneklerini aynı biçimde yazarsan modele "şu isteğe şu cevabı ver" demeyi öğretmiş olursun. Bu, modelin davranışını değiştirmenin en etkili yoludur.

### Dış veri (`veri/` klasörü)

- `main.py`'nin yanındaki `veri/` klasörü ve alt klasörleri taranır.
- Okunan uzantılar: `.html .htm .css .scss .js .jsx .ts .tsx .vue .py .txt .md .json`.
- Atlanan klasörler: `node_modules`, `__pycache__`, `venv`, `.venv`, `dist`, `build`, `.git` ve `.` ile başlayanlar. Adında `contributors` geçen dosyalar da atlanır.
- Toplam sınır: 30 milyon karakter.
- **Dengeleme:** Dış veri büyüdükçe gömülü veri tekrarlanır (en fazla 50 kez), böylece sohbet biçimi eğitimin yaklaşık %20'sini oluşturur ve unutulmaz.
- Yeni dosya ekledikten sonra panelde **↻ Veriyi Yenile**'ye bas. Eğitim sürerken veri yenilenemez.

### Veri ekleme yolları (panelden)

| Yol | Ne olur |
|---|---|
| **⬇ İndir** (web adresi) | Sayfadaki kod örnekleri (`<pre>`, çok satırlı `<code>`, `<textarea>`, `w3-code`/`highlight`/`codehilite` sınıflı bloklar) çıkarılır, `veri/web/<site>_<yol>.txt` olarak kaydedilir. Kod bloğu yoksa görünür metin alınır. |
| **＋ Veriye Ekle** (yapıştırma) | Metin `veri/elle/<ad>.txt` olarak kaydedilir ve veri hemen yenilenir. |
| Klasöre dosya koymak | `veri/` altına istediğin dosyayı koy, **↻ Veriyi Yenile**'ye bas. |

**Web indiricinin kuralları:**

- Sadece aynı site ve başlangıç adresiyle aynı klasör altındaki bağlantıları gezer (en fazla 300 sayfa).
- `robots.txt` kurallarına uyar ve istekler arasında 1 saniye bekler.
- Resim, PDF, video, font gibi dosyaları ve MDN `contributors.txt` sayfalarını atlar.
- Tek satırlık satır içi `<code>` parçalarını (menü öğeleri) almaz.
- Kod bloklarının başındaki `html`, `css`, `js` gibi dil etiketlerini siler.
- Sayfa başına en fazla 3 MB indirir.
- `veri/web/` git'e eklenmez. İndirdiğin içeriğin lisansına sen dikkat etmelisin.

### Hazır örnek veriler

| Dosya | İçerik |
|---|---|
| `veri/elle/anlatim_ornekleri.txt` | 33 kodsuz, düz Türkçe/İngilizce açıklama (HTML nedir, flexbox, API, döngü, dil modeli, loss…) |
| `veri/elle/link_ornekleri.txt` | 28 link / form + link / buton / menü / kart örneği |
| `sentetik_form_uret.py` | İstenen alanları içeren form örnekleri üretir: 11 alan türü × 6 form türü × 7 istek kalıbı |
| `sentetik_sayfa_uret.py` | Form dışı sayfa örnekleri üretir: kişisel sayfa, hakkımda, portföy, menü, ürün kartı, blog yazısı, alt bilgi |

Sentetik veriyi üretmek için (sayı vermezseniz form 4000, sayfa 3000 örnek):

```bash
python3 sentetik_form_uret.py 1500 > veri/elle/sentetik_formlar.txt
python3 sentetik_sayfa_uret.py 3000 > veri/elle/sentetik_sayfalar.txt
```

Yalnızca form verisiyle eğitilen model her isteğe form ile cevap verir ("kişisel sayfa tasarla" deyince bile form yazar). İki üreteci birlikte kullanın ki model istek türünü ayırt etmeyi öğrensin.

---

## Metin üretimi

- İstek `<|kullanici|> {istek}\n<|elalem|>\n` biçimine sarılır ve model karakter karakter devam ettirir.
- Her adımda bağlam son `block_size` karaktere kırpılır. Logits sıcaklığa bölünür, isteğe bağlı **top-k** filtresi uygulanır ve `torch.multinomial` ile örneklenir.
- `<|son|>` üretildiğinde ya da karakter sınırına ulaşıldığında durur.
- Panel ayarları: **Sıcaklık** 0.1-1.5 (varsayılan 0.6; düşük = daha tutarlı), **Maks. karakter** 20-1200, **Top-k** 1-60.

---

## QA/QC kod doğrulayıcı

Her üretilen çıktı otomatik olarak denetlenir. Kod alanında yaptığın düzenlemeler de anında yeniden denetlenir. Sonuç yeşil (geçti) ya da kırmızı (hata var) bir panel, 0-100 arası bir skor ve kontrol listesi olarak gösterilir.

**Dil tespiti:** HTML → JavaScript → Python → CSS → düz metin sırasıyla kural tabanlı tespit.

| Dil | Kontroller |
|---|---|
| Tümü | Çıktı boş değil · yazdırılamayan kontrol karakteri yok |
| HTML | `html.parser` ile ayrıştırma · **etiket yığını**: kapanmamış, açılmadan kapatılmış, yanlış iç içe geçmiş etiketler · yarım kalmış etiket (`<` açılmış, `>` yok) · öznitelikte kapanmamış tırnak · gömülü `<style>` CSS denetimi · gömülü `<script>` parantez denetimi · uyarılar: bilinmeyen etiket, kapatılmış void etiket, DOCTYPE eksikliği |
| CSS | Süslü parantez dengesi · her bildirimde `özellik: değer` biçimi · geçerli özellik adı · boş değer |
| Python | `ast.parse` ile sözdizimi (satır numarasıyla hata) · parantez/tırnak dengesi · uyarılar: gövdesi boş fonksiyon, sekme/boşluk karışık girinti |
| JavaScript | Parantez / tırnak / şablon dizesi dengesi (yorumlar atlanır) · uyarı: son ifade `;` veya `}` ile bitmiyor |
| Düz metin | Parantez dengesi |

Skor = geçen kontrol oranı × 100, her uyarı için −3 (en fazla −15).

---

## Kontrol paneli

`http://127.0.0.1:8080`. Koyu tema, masaüstü ve telefonda çalışır, internet bağlantısı gerektirmez.

| Bölüm | İçerik |
|---|---|
| Üst çubuk | Durum (Hazır / Eğitiliyor), cihaz, parametre sayısı, sözlük boyutu, veri token sayısı |
| 🧠 Elalem LLM Eğitim Paneli | Adım sayısı ve öğrenme oranı · Başlat/Durdur · adım, eğitim loss, doğrulama loss, ms/adım kutuları · ilerleme çubuğu · **canlı loss grafiği** (eğitim + doğrulama) · log alanı |
| 📚 Eğitim Verisi | Dış dosya ve karakter sayısı · web adresinden indirme (sayfa sınırıyla) · metin yapıştırma · veriyi yenileme · kaynak listesi |
| ⚡ Prompt / Kod Üretim Alanı | İstek kutusu · hazır örnek istekler · sıcaklık / maks. karakter / top-k ayarları · düzenlenebilir kod çıktısı · Kopyala · Yeniden Doğrula |
| 👁️ Canlı Önizleme | HTML olduğu gibi, CSS örnek bir sayfaya uygulanarak, JS örnek sayfada çalıştırılarak gösterilir · açık/koyu zemin · masaüstü/mobil genişlik · yeni sekmede açma · JS hataları kırmızı kutuda |
| 🛡️ QA/QC Raporu | Yeşil/kırmızı durum, skor, dil, kontrol listesi, uyarılar |

---

## HTTP API

Tüm cevaplar JSON'dur. Sunucu sadece `127.0.0.1`'e bağlanır.

| Yöntem | Yol | Gövde | Açıklama |
|---|---|---|---|
| GET | `/` | — | Kontrol paneli |
| GET | `/api/status` | — | Eğitim durumu, loss geçmişi, loglar, veri kaynakları, web indirme durumu |
| POST | `/api/train` | `{"steps": 1500, "lr": 0.003}` | Eğitimi başlatır (1-100.000 adım) |
| POST | `/api/stop` | `{}` | Eğitimi durdurur |
| POST | `/api/generate` | `{"prompt": "...", "max_new_tokens": 400, "temperature": 0.6, "top_k": 10}` | Üretim + QA/QC raporu |
| POST | `/api/qc` | `{"code": "..."}` | Verilen kodu doğrular |
| POST | `/api/data/url` | `{"url": "https://...", "max_sayfa": 1}` | Web'den kod örneği indirir (1-300 sayfa) |
| POST | `/api/data/stop` | `{}` | İndirmeyi durdurur |
| POST | `/api/data/text` | `{"ad": "...", "metin": "..."}` | Metni `veri/elle/` altına kaydeder |
| POST | `/api/data/reload` | `{}` | `veri/` klasörünü yeniden yükler |

---

## Komut satırı seçenekleri

| Seçenek | Varsayılan | Açıklama |
|---|---|---|
| `--host` | `127.0.0.1` | Sunucu adresi |
| `--port` | `8080` | Sunucu portu |
| `--open` | kapalı | Paneli tarayıcıda otomatik aç |
| `--boyut` | `kucuk` | Hazır model boyutu: `kucuk`, `orta`, `buyuk`, `dev` |
| `--n-embd` | boyuta göre | Gömme boyutu; `--n-head`'e tam bölünmeli |
| `--n-layer` | boyuta göre | Transformer blok sayısı |
| `--n-head` | boyuta göre | Dikkat kafası sayısı |
| `--block-size` | boyuta göre | Bağlam uzunluğu (karakter) |
| `--batch-size` | boyuta göre | Batch boyutu |
| `--lr` | boyuta göre | Varsayılan öğrenme oranı (panele de yansır) |
| `--selftest` | kapalı | Tokenizer, batch şekilleri, model çıktısı, eğitim, üretim ve 6 QC vakasını test edip çıkar (çıkış kodu 0/1) |
| `--selftest-steps` | `30` | Self-test sırasında eğitim adımı |

---

## Dosyalar

```
elalem/
├── main.py                  # tüm model, eğitim, QC, sunucu ve arayüz
├── sentetik_form_uret.py    # sentetik form verisi üreteci
├── sentetik_sayfa_uret.py   # sentetik sayfa verisi üreteci
├── README.md
├── veri/
│   ├── elle/                # elle eklenen / hazır örnek veriler (git'te)
│   └── web/                 # web'den indirilenler (git'e eklenmez)
└── elalem_checkpoint*.pt    # eğitilmiş ağırlıklar (git'e eklenmez)
```

---

## Güvenlik

- Sunucu varsayılan olarak sadece `127.0.0.1`'i dinler, ağdaki başka cihazlar erişemez.
- Önizleme, `allow-same-origin` olmadan sandbox'lı bir `iframe` içinde çalışır; üretilen JS paneline ya da API'ye erişemez.
- Üretilen Python kodu **hiçbir zaman çalıştırılmaz**, sadece `ast.parse` ile ayrıştırılır.
- İstek gövdesi 1 MB ile, prompt 1.000 karakterle, üretim 2.000 karakterle sınırlıdır.
- Ağırlıklar `torch.load(..., weights_only=True)` ile yüklenir.

---

## Parametreleri büyütmek

Kodu açmadan komut satırından ayarlanır. Önce bir hazır boyut seçilir, verdiğin ayarlar onun üzerine yazılır:

```bash
python3 main.py --boyut dev --open                                  # hazır ~38M
python3 main.py --boyut orta --n-layer 10 --open                    # orta, ama 10 blok
python3 main.py --n-embd 640 --n-head 10 --n-layer 12 --block-size 512 --batch-size 12 --lr 2e-4 --open   # ~59M
```

Açılışta terminale seçilen ayarlar ve tahmini parametre sayısı yazılır, panelin üst çubuğunda da gerçek sayı görünür.

| Ayar | Etkisi |
|---|---|
| `--n-embd` | Her karakterin temsil genişliği. Parametre sayısı bunun **karesiyle** artar. `--n-head`'e tam bölünmeli. |
| `--n-layer` | Transformer blok sayısı. Parametre sayısı bununla **doğru orantılı** artar. |
| `--n-head` | Dikkat kafası sayısı. Parametre sayısını değiştirmez. |
| `--block-size` | Modelin aynı anda görebildiği karakter sayısı. Parametre değil, bellek ve hız maliyeti. |
| `--batch-size` | Her adımda işlenen örnek sayısı. Bellek yetmezse düşür. |
| `--lr` | Öğrenme oranı. Model büyüdükçe düşürülmeli (ör. 3e-4, 2e-4). |

Kaba hesap: **parametre ≈ 12 × n_layer × n_embd²**.

| n_embd | n_layer | Yaklaşık parametre |
|---|---|---|
| 384 | 8 | 14 M |
| 512 | 12 | 38 M |
| 640 | 12 | 59 M |
| 768 | 12 | 85 M |

Her ayar kombinasyonu ağırlıklarını kendi dosyasına (`elalem_checkpoint_<n_embd>x<n_layer>x<block_size>.pt`) kaydeder; mimari değişince model sıfırdan başlar. Daha büyük model daha çok veri, daha uzun eğitim ve daha fazla bellek ister. Bellek hatası alırsan önce `--batch-size`'ı düşür.

## Sınırlamalar

- **Ölçek:** 2-14 milyon parametre ve birkaç megabayt veriyle model gördüğü kalıpları taklit eder. Hiç görmediği istekleri anlamaz.
- **İsteği takip etme:** Küçük modeller isteği okuyup cevaba kelime kopyalamakta zorlanır. Test ortamında 0,7 M parametreli bir model 1.500 adımda geçerli form yapısını öğrendi ama alanları istekten seçmeyi henüz öğrenmedi. Bu beceri genelde model büyüdükçe ve eğitim uzadıkça gelir.
- **Karakter seviyesi:** Kelime seviyesindeki modellere göre aynı metni öğrenmek daha çok adım gerektirir.
- **Eğitim hızı:** CPU'da yavaştır. Apple M serisi Mac'lerde `mps` ile çok daha hızlı çalışır.
