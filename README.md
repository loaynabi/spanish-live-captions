# Spanish Live Captions

Live Spanish captions from YouTube or shared computer audio, using a downloadable local model.

## Recognition

The app uses `faster-whisper` with the downloadable `small` model. It runs locally on CPU with int8 quantization, so it needs no token, cloud account, subscription, or internet after the model is downloaded.

Free accuracy improvements are enabled:

- Spanish transcription is forced with `language="es"`
- 5-second context windows
- Voice-activity detection removes silence
- Audio is normalized and centered before recognition
- Beam search and medical Spanish prompt are enabled

## Install and run

Use Command Prompt:

```bat
cd "%USERPROFILE%\Desktop\spanish live captions"
py -3.13 -m pip install -r requirements.txt
set WHISPER_MODEL=small
py -3.13 server.py
```

The first start downloads the model. Later starts work offline. Open <http://127.0.0.1:8765>.

## YouTube test

1. Click **Start captions**.
2. Select the YouTube tab in Chrome or Edge.
3. Enable **Share audio** and click **Share**.
4. Play clear Spanish speech.
5. Wait about five seconds for each completed caption.

The separate caption window opens automatically and can be moved or resized.

## Accuracy tips

Use clear audio, one speaker at a time, and avoid music or overlapping speech. The `medium` model can be selected with `set WHISPER_MODEL=medium`, but it is much heavier on this 4-core CPU. Do not use captions as a replacement for a qualified interpreter or clinical documentation.
