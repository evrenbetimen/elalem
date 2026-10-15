# Elalem

Sıfırdan, saf PyTorch ile yazılmış karakter seviyesi bir GPT ve onu eğitip kullanmak için 127.0.0.1 üzerinde çalışan koyu temalı bir kontrol paneli. Her şey tek dosyada: `main.py`.

```bash
pip install torch
python main.py            # http://127.0.0.1:8080
python main.py --open     # paneli tarayıcıda otomatik açar
python main.py --selftest # kısa uçtan uca test (eğitim, üretim, QA/QC)
```

Panelde eğitimi başlat, loss grafiğini izle, sonra prompt yazıp kod üret. Üretilen kod kural tabanlı QA/QC katmanından geçer (HTML etiket eşleşmesi, CSS bildirimleri, Python `ast.parse`, parantez/tırnak dengesi). Eğitilen ağırlıklar `elalem_checkpoint.pt` dosyasına kaydedilir ve bir sonraki açılışta yüklenir.

Cihaz seçimi: CUDA, yoksa Apple `mps`, o da yoksa CPU.

## Eğitim verisini büyütmek

- `main.py`'nin yanına bir `veri/` klasörü aç ve içine `.html`, `.css`, `.js`, `.py`, `.txt`, `.md` dosyaları koy. Elalem bunları açılışta (ya da paneldeki **Veriyi Yenile** ile) eğitime ekler. Kendi projelerinin kodu en iyi kaynaktır.
- Panelin **Eğitim Verisi** bölümünden bir web adresi girersen sayfadaki `<pre>`/`<code>` kod örnekleri `veri/web/` klasörüne indirilir. "En fazla sayfa" 1'den büyükse aynı klasördeki bağlantılar da gezilir. İndirici `robots.txt` kurallarına uyar ve istekler arasında 1 sn bekler. İndirdiğin içeriğin lisansına dikkat et; `veri/web/` git'e eklenmez.
- Çok veri için daha büyük model: `python main.py --boyut orta` (ya da `buyuk`). Her boyut kendi ağırlık dosyasına kaydedilir.
