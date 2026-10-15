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
