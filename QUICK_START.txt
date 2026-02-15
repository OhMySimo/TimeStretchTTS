╔══════════════════════════════════════════════════════════════════════════════╗
║                   LENGTH PENALTY - QUICK START GUIDE                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

📦 HAI SCARICATO 8 FILE:

1. length_penalty_processor.py      → Il codice del processor
2. apply_patch.py                   → Script automatico di installazione ⭐ NUOVO
3. install_length_penalty.sh        → Wrapper bash per apply_patch.py
4. qwen3_tts_model_PATCH.txt       → Le modifiche (per riferimento)
5. PATCH_EXAMPLE.txt                → Esempio dettagliato (per riferimento)
6. test_length_penalty.py           → Script di test
7. README_LENGTH_PENALTY.md         → Documentazione completa
8. QUICK_START.txt                  → Questa guida

════════════════════════════════════════════════════════════════════════════════

🚀 INSTALLAZIONE AUTOMATICA IN 1 STEP:

┌──────────────────────────────────────────────────────────────────────────────┐
│ OPZIONE A: Esegui lo script bash (raccomandato)                             │
└──────────────────────────────────────────────────────────────────────────────┘

chmod +x install_length_penalty.sh
./install_length_penalty.sh

Lo script:
  ✅ Applica TUTTE le modifiche automaticamente
  ✅ Crea backup del file originale
  ✅ Verifica che tutto sia corretto
  ✅ NESSUNA modifica manuale richiesta!


┌──────────────────────────────────────────────────────────────────────────────┐
│ OPZIONE B: Esegui direttamente lo script Python                             │
└──────────────────────────────────────────────────────────────────────────────┘

python3 apply_patch.py

Stesso risultato dell'opzione A.


┌──────────────────────────────────────────────────────────────────────────────┐
│ OPZIONE C: Installazione manuale (se automatica fallisce)                   │
└──────────────────────────────────────────────────────────────────────────────┘

1. Leggi qwen3_tts_model_PATCH.txt
2. Vedi PATCH_EXAMPLE.txt per esempi dettagliati
3. Applica modifiche manualmente con editor di testo

════════════════════════════════════════════════════════════════════════════════

🧪 TESTING:

┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. Modifica test_length_penalty.py                                          │
└──────────────────────────────────────────────────────────────────────────────┘

Apri il file e imposta il path del tuo audio di riferimento:
  REF_AUDIO = "/path/to/your/audio.wav"


┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. Esegui il test                                                            │
└──────────────────────────────────────────────────────────────────────────────┘

python3 test_length_penalty.py

Genera 15 clip (3 per ogni valore di alpha: 0.0, 0.10, 0.15, 0.20, 0.25)
Salva tutto in: length_penalty_test/


┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. Ascolta e scegli                                                          │
└──────────────────────────────────────────────────────────────────────────────┘

Ascolta le clip in ordine e trova il miglior compromesso tra:
  ✓ Durata corretta
  ✓ Qualità vocale
  ✓ Naturalezza

Tipicamente alpha=0.15 è un buon punto di partenza.

════════════════════════════════════════════════════════════════════════════════

💡 USO NEL TUO CODICE:

from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained("simone00/it2")

wavs, sr = tts.generate_voice_clone(
    text="Il tuo testo",
    ref_audio="audio.wav",
    x_vector_only_mode=True,
    temperature=0.72,
    top_p=0.875,
    length_penalty_alpha=0.15,          # ← IL PARAMETRO MAGICO
    frames_per_text_token=8.0,          # ← Regola se necessario
)

════════════════════════════════════════════════════════════════════════════════

🔄 PER RIPRISTINARE IL FILE ORIGINALE:

Lo script crea automaticamente un backup con timestamp.
Per ripristinarlo:

# Trova il backup (nella directory qwen_tts/inference/)
python3 -c "import qwen_tts, os; print(os.path.dirname(qwen_tts.__file__))"
# Vai in quella directory/inference/ e cerca qwen3_tts_model.py.backup_*

# Copia il backup sopra il file modificato
cp qwen3_tts_model.py.backup_YYYYMMDD_HHMMSS qwen3_tts_model.py

════════════════════════════════════════════════════════════════════════════════

📚 PER MAGGIORI DETTAGLI:

Leggi README_LENGTH_PENALTY.md per:
  • Spiegazione dettagliata di come funziona
  • Troubleshooting
  • Note tecniche
  • Parametri avanzati

════════════════════════════════════════════════════════════════════════════════

⚡ TL;DR - COSA FA:

PROBLEMA: Il modello genera speech troppo lento
SOLUZIONE: Forza il modello a fermarsi prima boosting EOS token
COME: Aggiunge "length penalty" che aumenta la probabilità di EOS quando 
      la generazione supera la lunghezza attesa

🎯 INSTALLAZIONE: Completamente automatica! Nessuna modifica manuale!

╔══════════════════════════════════════════════════════════════════════════════╗
║                         BUON TESTING! 🎉                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
