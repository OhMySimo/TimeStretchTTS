# Length Penalty Modification per Qwen3-TTS

Questa modifica aggiunge il controllo della durata all'inferenza di Qwen3-TTS per prevenire speech troppo lento.

## 📋 Cosa Fa

**Problema:** Il modello genera speech più lento del normale durante l'inferenza, anche se i dati di training sono a velocità corretta.

**Soluzione:** Aggiunge un "length penalty" che aumenta la probabilità del token EOS quando la generazione supera la lunghezza attesa, forzando il modello a concludere più velocemente.

## 📦 File Inclusi

1. **length_penalty_processor.py** - Il LogitsProcessor che implementa la penalty
2. **qwen3_tts_model_PATCH.txt** - Modifiche da applicare al file della libreria
3. **test_length_penalty.py** - Script di test per trovare il valore ottimale
4. **install_length_penalty.sh** - Script di installazione guidata

## 🚀 Installazione Rapida

```bash
# 1. Assicurati di avere qwen-tts installato
pip install qwen-tts

# 2. Scarica tutti i file in una directory

# 3. Rendi eseguibile lo script di installazione
chmod +x install_length_penalty.sh

# 4. Esegui l'installazione
./install_length_penalty.sh
```

Lo script:
- Trova automaticamente dove è installato qwen_tts
- Copia length_penalty_processor.py nella posizione corretta
- Ti guida nell'applicare il patch manualmente

## 🔧 Installazione Manuale

Se preferisci installare manualmente:

### Step 1: Trova la directory di qwen_tts

```bash
python3 -c "import qwen_tts; import os; print(os.path.dirname(qwen_tts.__file__))"
# Output esempio: /usr/local/lib/python3.10/site-packages/qwen_tts
```

### Step 2: Copia il processor

```bash
cp length_penalty_processor.py /percorso/a/qwen_tts/inference/
```

### Step 3: Modifica qwen3_tts_model.py

Apri il file:
```bash
nano /percorso/a/qwen_tts/inference/qwen3_tts_model.py
```

Applica le modifiche descritte in `qwen3_tts_model_PATCH.txt`:

1. **Aggiungi import** (dopo riga 29):
```python
from .length_penalty_processor import LengthPenaltyLogitsProcessor
```

2. **Modifica signature del metodo** `generate_voice_clone` (circa riga 489):
```python
def generate_voice_clone(
    self,
    text: Union[str, List[str]],
    language: Union[str, List[str]] = None,
    ref_audio: Optional[Union[AudioLike, List[AudioLike]]] = None,
    ref_text: Optional[Union[str, List[Optional[str]]]] = None,
    x_vector_only_mode: Union[bool, List[bool]] = False,
    voice_clone_prompt: Optional[Union[List[VoiceClonePromptItem], Dict]] = None,
    non_streaming_mode: bool = True,
    length_penalty_alpha: float = 0.0,          # ← NUOVO
    frames_per_text_token: float = 8.0,         # ← NUOVO
    **kwargs,
) -> Tuple[List[np.ndarray], int]:
```

3. **Aggiungi documentazione** nella docstring (dopo riga 534):
```python
        length_penalty_alpha:
            Strength of length penalty (0.0 = disabled, 0.15 = medium, 0.20 = strong).
            Prevents generation from being excessively long by boosting EOS probability.
        frames_per_text_token:
            Expected codec frames per text token (default: 8.0).
            Lower values = expects faster speech. Adjust based on speaking rate.
```

4. **Aggiungi logica penalty** prima di `self.model.generate()` (circa riga 601):
```python
gen_kwargs = self._merge_generate_kwargs(**kwargs)

# Apply length penalty if enabled
if length_penalty_alpha > 0:
    try:
        eos_id = self.model.config.talker_config.codec_eos_token_id
        
        processors = []
        for ids in input_ids:
            processor = LengthPenaltyLogitsProcessor(
                text_length=ids.shape[1],
                frames_per_text_token=frames_per_text_token,
                penalty_alpha=length_penalty_alpha,
                eos_token_id=eos_id,
            )
            processors.append(processor)
        
        if 'logits_processor' not in gen_kwargs:
            gen_kwargs['logits_processor'] = []
        gen_kwargs['logits_processor'].extend(processors[:1])
        
    except Exception as e:
        print(f"Warning: Could not apply length penalty: {e}")

talker_codes_list, _ = self.model.generate(
    # ... resto del codice invariato
```

### Step 4: Verifica installazione

```bash
python3 << 'EOF'
from qwen_tts.inference.length_penalty_processor import LengthPenaltyLogitsProcessor
from qwen_tts import Qwen3TTSModel
import inspect

sig = inspect.signature(Qwen3TTSModel.generate_voice_clone)
params = list(sig.parameters.keys())

if 'length_penalty_alpha' in params and 'frames_per_text_token' in params:
    print("✅ Installation successful!")
else:
    print("❌ Patch not applied correctly")
EOF
```

## 🧪 Testing

### Preparazione

1. Modifica `test_length_penalty.py` e imposta il path del tuo reference audio:
```python
REF_AUDIO = "/path/to/your/audio.wav"  # ← MODIFICA QUESTO
```

2. Esegui il test:
```bash
python3 test_length_penalty.py
```

### Cosa Fa il Test

- Genera 3 clip per ogni valore di `length_penalty_alpha` (0.0, 0.10, 0.15, 0.20, 0.25)
- Misura durata e speaking rate per ogni clip
- Salva tutto in `length_penalty_test/`
- Crea un report comparativo in `results.csv`

### Output Esempio

```
📈 FINAL RESULTS
========================================
Alpha    Frames/Tok   Duration         Rate (w/s)   Samples
------------------------------------------------------------------------
0.00     8.0          4.52s ±0.08      2.65         3
0.10     8.0          4.18s ±0.12      2.87         3
0.15     8.0          3.89s ±0.09      3.08         3
0.20     8.0          3.62s ±0.11      3.31         3
0.25     8.0          3.45s ±0.14      3.48         3
```

### Come Scegliere il Valore

1. Ascolta le clip in ordine (0.0 → 0.25)
2. Trova il punto dove:
   - La durata è accettabile
   - La qualità vocale non è degradata
   - Il parlato suona naturale

**Tipicamente:**
- `alpha=0.15` è un buon punto di partenza
- `alpha=0.20` per parlato molto lento che richiede correzione forte
- Se anche 0.25 è troppo lento, riduci `frames_per_text_token` a 7.0 o 6.5

## 📊 Parametri Tunabili

### `length_penalty_alpha`
- **Cosa fa:** Forza della penalty
- **Range:** 0.0 (disabled) - 0.30 (molto forte)
- **Default:** 0.0 (disabilitato)
- **Raccomandato:** Inizia con 0.15

### `frames_per_text_token`
- **Cosa fa:** Quanti frame codec ci aspettiamo per token di testo
- **Range:** 6.0 (parlato veloce) - 10.0 (parlato lento)
- **Default:** 8.0
- **Come stimare:** 
  - Misura durata di un audio di training (es. 5 secondi)
  - Conta i token di testo (es. 50 token)
  - fps = 12.5 (tokenizer 12Hz)
  - frames_per_token = (5.0 * 12.5) / 50 = 1.25... NO aspetta
  - Meglio: genera audio senza penalty, misura frames generati / text_tokens

## 🔄 Uso nel Codice

### Esempio Base

```python
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained("simone00/it2")

wavs, sr = tts.generate_voice_clone(
    text="Il tuo testo qui",
    ref_audio="audio.wav",
    x_vector_only_mode=True,
    temperature=0.72,
    top_p=0.875,
    length_penalty_alpha=0.15,          # ← Abilita penalty
    frames_per_text_token=8.0,
)
```

### Esempio con Reference Duration (avanzato)

Se hai accesso alla durata del reference audio:

```python
import librosa
from qwen_tts.inference.length_penalty_processor import AdaptiveLengthPenaltyLogitsProcessor

# Misura durata reference
ref_wav, ref_sr = librosa.load("ref_audio.wav")
ref_duration = len(ref_wav) / ref_sr

# Usa AdaptiveLengthPenaltyLogitsProcessor manualmente
# (richiede modifiche più profonde al codice di generazione)
```

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'length_penalty_processor'"

Il file non è stato copiato correttamente. Verifica:
```bash
ls -la /percorso/a/qwen_tts/inference/length_penalty_processor.py
```

### "generate_voice_clone() got an unexpected keyword argument 'length_penalty_alpha'"

Il patch non è stato applicato. Verifica che la signature del metodo includa i nuovi parametri.

### I clip generati sono identici indipendentemente da alpha

1. Verifica che alpha > 0.0
2. Controlla se il modello sta effettivamente raggiungendo la lunghezza target (prova con testo più lungo)
3. Verifica che `eos_token_id` sia corretto nel processor

### La qualità peggiora con penalty alta

È normale. La penalty sta forzando il modello a concludere prima. Riduci alpha o aumenta frames_per_text_token.

## 📝 Note Tecniche

### Limitazioni

1. **Batch Processing:** Il LogitsProcessor corrente usa la lunghezza del primo sample per tutto il batch. Per batch con testi di lunghezza molto diversa, genera un sample alla volta.

2. **Non è una soluzione perfetta:** La penalty migliora ma non risolve completamente il problema training-inference mismatch. Per risultati ottimali considera anche Scheduled Sampling durante training.

3. **Trade-off qualità:** Penalty troppo alta può degradare naturalità e prosodia. Trova il bilanciamento giusto.

### Come Funziona Internamente

1. Calcola target frame count: `text_length × frames_per_text_token`
2. Durante generazione autoregressive, ad ogni step:
   - Se step > target: calcola excess_ratio
   - Boost EOS token: `logits[EOS] += alpha × excess_ratio × 10.0`
   - Il modello è più propenso a fermarsi

3. La penalty cresce esponenzialmente con l'eccesso, quindi anche piccoli valori di alpha hanno effetto forte quando si supera molto il target.

## 🔗 Integrazione con Training

Questa modifica è **complementare** al training con duration head. Non sono in conflitto:

- **Duration head (training):** Insegna al modello rappresentazioni temporali migliori
- **Length penalty (inference):** Corregge il bias verso generazione lenta

Idealmente usi entrambi:
1. Training con duration head → modello impara timing migliore
2. Inferenza con length penalty → corregge residuo di lentezza

## 📞 Support

Se incontri problemi:
1. Verifica di aver seguito tutti gli step di installazione
2. Controlla i messaggi di errore specifici
3. Testa prima con `alpha=0.0` (disabilitato) per verificare che il codice base funzioni

## 📄 License

Questo codice è fornito as-is per uso con qwen_tts.
