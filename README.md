# dnz

Ses dosyasından özet çıkaran araç. / Audio summarization tool.

## Kullanım / Usage

Bir ses dosyasından transkript alır ve özetler.  
Transcribes an audio file and produces a concise summary.

### Kurulum / Installation

```bash
pip install -r requirements.txt
```

### Çalıştırma / Running

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# Summarize an audio file (Turkish output by default)
python summarize.py audio.mp3

# Summarize with English output
python summarize.py audio.mp3 --language en

# Also print the full transcript
python summarize.py audio.mp3 --transcript
```

### Desteklenen formatlar / Supported formats

`mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, `webm`, `ogg`, `flac`

## Testler / Tests

```bash
pytest tests/
```