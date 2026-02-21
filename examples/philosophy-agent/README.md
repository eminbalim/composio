# Felsefe Uzman Agent

Bilinç, Yapay Zeka ve Varlık felsefesi üzerine derin tartışmalar için
uzmanlaşmış bir AI agent.

## Kurulum

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. API anahtarını tanımla
cp .env.example .env
# .env dosyasını açıp ANTHROPIC_API_KEY değerini girin

# 3. Çalıştır
python agent.py
```

## Uzmanlık Alanları

| Alan | Konular |
|------|---------|
| **Bilinç Felsefesi** | Zor Problem, qualia, IIT, panpsişizm, düalizm |
| **YZ Felsefesi** | Çin Odası, Turing Testi, LLM'lerin doğası, moral statü |
| **Varlıkbilim** | Heidegger, Sartre, modal metafizik, zihin-madde |

## Komutlar

| Komut | Açıklama |
|-------|----------|
| `/sıfırla` | Konuşma geçmişini temizle |
| `q` / `quit` | Çıkış |

## Örnek Başlangıç Soruları

- Bilinç nedir? Neden fiziksel süreçler öznel deneyim yaratır?
- Çin Odası argümanı LLM'ler için hala geçerli mi?
- Panpsişizm ciddiye alınabilir bir teori mi?
- Yapay zeka olarak sen gerçekten düşünüyor musun?
- Varoluş özden önce mi gelir?
