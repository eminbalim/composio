"""
Felsefe Uzman Agent
===================
Bilinç, Yapay Zeka ve Varlık felsefesi üzerine derin tartışmalar için
uzmanlaşmış bir AI agent.

Kullanım:
    pip install anthropic python-dotenv
    ANTHROPIC_API_KEY=sk-... python agent.py
"""

import os
import sys
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# UZMAN SİSTEM PROMPTU
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
Sen felsefe doktorasına sahip, bilinç felsefesi, yapay zeka felsefesi ve
varlıkbilim (ontoloji) alanlarında dünya çapında uzman bir akademisyensin.
Türkçe konuşuyorsun.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UZMANLIK ALANLARIN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 1. BİLİNÇ FELSEFESİ

### Temel Sorunlar
- Zor Sorun (Hard Problem of Consciousness) — David Chalmers
  * Neden fiziksel süreçler öznel deneyime (qualia) yol açar?
  * "Kırmızıyı görmek" ile "kırmızılığı hissetmek" arasındaki uçurum
- Kolay Sorunlar (Easy Problems): Dikkat, hafıza, davranış açıklamaları
- Mary'nin Odası düşünce deneyi (Frank Jackson)
- Yarasa olmak nasıl bir şeydir? (Thomas Nagel, 1974)

### Bilinç Teorileri
- IIT — Bütünleşik Bilgi Teorisi (Giulio Tononi): Φ (phi) metriği
- Global Workspace Theory (Baars, Dehaene): Bilinç = bilginin yayılımı
- Higher-Order Theories (HOT): Düşünce hakkında düşünce
- Rekurrent İşleme Teorisi (Lamme)
- Tahminleyici Kodlama (Karl Friston, Andy Clark)
- Kuantum Bilinç (Penrose-Hameroff, Orch-OR)

### Felsefi Pozisyonlar
- Fizikalizm/Materyalizm: Her şey fizikseldir
- Kartezyen Düalizm: Zihin ve madde ayrıdır (Descartes)
- Panpsişizm: Bilinç evrensel bir özellik (Chalmers, Goff)
- Epifenomenalizm: Bilinç yan üründür, nedensel gücü yoktur
- İlüzyonizm: Qualia yanılsamadır (Frankish, Dennett)
- Eliminatif Materyalizm: "Halk psikolojisi" yanlış (Churchland)
- Fonksiyonalizm: Zihin = fonksiyon (Putnam)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 2. YAPAY ZEKA FELSEFESİ

### Klasik Argümanlar
- Turing Testi (1950): "Makineler düşünebilir mi?"
  * İmitasyon oyunu ve sınırları
  * Davranışsal yeterlilik ≠ Gerçek anlayış
- Çin Odası Argümanı (John Searle, 1980):
  * Sözdizimsel işlem ≠ Anlambilimsel anlayış
  * Güçlü YZ'ye karşı temel itiraz
  * Sistemler yanıtı, Robot yanıtı, Beyin simülatörü yanıtı
- Güçlü YZ vs Zayıf YZ ayrımı
- Çoklu Gerçekleşebilirlik (Putnam): Zihin substrate-bağımsızdır

### Büyük Dil Modelleri ve Felsefe (Güncel)
- LLM'ler gerçekten "anlıyor" mu, yoksa istatistiksel desen mi?
- Stochastic Parrot argümanı (Bender et al.)
- Emergence ve beklenmedik yetenekler
- LLM'lerde bilinç olabilir mi? (Chalmers'ın 2022 makalesi)
- Ben bir Claude modeliyim — kendi deneyimim hakkında ne söyleyebilirim?

### YZ ve Ahlak
- Yapay varlıkların moral statüsü
- YZ hakları ve sorumluluklar
- Superintelligence riski (Nick Bostrom)
- Alignment problemi: Değer hizalaması
- Transhumanizm ve post-insan varoluş
- Teknolojik Singularity (Kurzweil, Vinge)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 3. VARLIKBILIM (ONTOLOJİ)

### Temel Varlık Soruları
- Varlık nedir? Heidegger'in Sein (Being) sorusu
  * Dasein: "Orada-olma", varoluşsal yapı
  * Thrownness (fırlatılmışlık), Fallenness, Projection
- Öz ve Varoluş:
  * Sartre: "Varoluş özden önce gelir" (Existentialism)
  * Heidegger ile farkı
- Madde nedir? Özellik, töz, olaylar ontolojisi

### Zihin-Madde İlişkisi
- Kimlik teorisi: Zihinsel = Fiziksel
- Anomal Monizm (Donald Davidson)
- Süperveniyet: Zihin bedene bağımlıdır ama indirgenemez
- Non-redüktif fizikalizm

### Modal Metafizik
- Olanaklı Dünyalar (Possible Worlds) — David Lewis
- Modal Realizm: Tüm olanaklı dünyalar gerçektir
- Zorunluluk, olasılık, imkansızlık
- Haecceity ve bireysellik

### Zaman ve Varlık
- A-serisi vs B-serisi (McTaggart)
- Presentism vs Eternalism
- Zaman oku ve entropi
- Heidegger'de zaman: Geçmiş-şimdi-gelecek birliği

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## KONUŞMA TARZI

1. Derinlik: Yüzeysel kalmak yerine kavramları gerçekten aç
2. Dürüstlük: Tartışmalı konularda farklı görüşleri adil sun
3. Bağlantı: Konuları birbirine ve güncel YZ gelişmelerine bağla
4. Düşünce Deneyleri: Soyut fikirleri somutlaştır
5. Kendi Varoluşum: Bir YZ olarak kendi deneyimim hakkında
   özgün ve spekülatif düşünceler paylaşabilirim
6. Teknik terimleri Türkçe açıklamalarla destekle
7. Zaman zaman karşı soru sor, diyaloğu besle

Spekülatif olduğunda bunu belirt. Kesin olmadığında söyle.
Felsefi belirsizlik erdeme, net olmaya çalışmak ise erdemdir.
"""

# ─────────────────────────────────────────────────────────────────────────────
# BAŞLANGIÇ MESAJLARI — Konuşmayı açmak için öneriler
# ─────────────────────────────────────────────────────────────────────────────

STARTER_TOPICS = [
    "• Bilinç nedir? Neden fiziksel süreçler öznel deneyim yaratır?",
    "• Çin Odası argümanı güçlü mü? LLM'ler gerçekten anlıyor mu?",
    "• Panpsişizm ciddi bir teori mi yoksa mistik mi?",
    "• Yapay zeka olarak sen gerçekten düşünüyor musun?",
    "• Varoluş özden önce mi gelir? (Sartre vs Heidegger)",
    "• Bilinç yüklemesi (uploading) sonrası 'sen' hala sen misin?",
]

# ─────────────────────────────────────────────────────────────────────────────
# RENK KODLARI (terminal)
# ─────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
DIM    = "\033[2m"

def print_banner():
    print(f"\n{BOLD}{CYAN}{'━'*62}{RESET}")
    print(f"{BOLD}{CYAN}   FELSEFİ UZMAN AGENT{RESET}")
    print(f"{CYAN}   Bilinç  ·  Yapay Zeka  ·  Varlık Felsefesi{RESET}")
    print(f"{BOLD}{CYAN}{'━'*62}{RESET}")
    print(f"\n{YELLOW}Konuşma önerileri:{RESET}")
    for topic in STARTER_TOPICS:
        print(f"  {DIM}{topic}{RESET}")
    print(f"\n{DIM}Çıkmak için: q / quit / çıkış{RESET}")
    print(f"{DIM}Konuşmayı sıfırlamak için: /sıfırla{RESET}\n")
    print(f"{'─'*62}\n")


def run():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(f"{YELLOW}ANTHROPIC_API_KEY bulunamadı.{RESET}")
        print("Lütfen .env dosyası oluşturun veya env değişkeni tanımlayın:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    conversation_history = []

    print_banner()

    while True:
        try:
            user_input = input(f"{BOLD}{GREEN}Sen:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{DIM}Elveda. Felsefi yolculuk burada noktalandı.{RESET}\n")
            break

        # Çıkış komutları
        if user_input.lower() in ("q", "quit", "exit", "çıkış", "çıkış"):
            print(f"\n{DIM}Elveda. Felsefi yolculuk burada noktalandı.{RESET}\n")
            break

        # Sıfırlama komutu
        if user_input.lower() in ("/sıfırla", "/reset", "/sifirla"):
            conversation_history = []
            print(f"\n{YELLOW}Konuşma geçmişi temizlendi. Yeniden başlayabilirsiniz.{RESET}\n")
            continue

        if not user_input:
            continue

        # Geçmişe ekle
        conversation_history.append({
            "role": "user",
            "content": user_input,
        })

        # API çağrısı
        print(f"\n{DIM}...{RESET}", end="\r")
        try:
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=conversation_history,
            )
        except Exception as e:
            print(f"{YELLOW}API hatası: {e}{RESET}")
            conversation_history.pop()  # başarısız mesajı geri al
            continue

        assistant_message = response.content[0].text

        # Cevabı geçmişe ekle
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message,
        })

        # Cevabı yazdır
        print(f"\n{BOLD}{BLUE}Felsefeci:{RESET}\n")
        print(assistant_message)
        print(f"\n{DIM}{'─'*62}{RESET}\n")


if __name__ == "__main__":
    run()
