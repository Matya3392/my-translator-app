import os
import shutil
import imageio_ffmpeg
from deep_translator import GoogleTranslator

# --- ffmpeg.exe の自動準備 ---
project_ffmpeg = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
if not os.path.exists(project_ffmpeg):
    original_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    shutil.copy(original_ffmpeg, project_ffmpeg)

os.environ["PATH"] = os.path.dirname(__file__) + os.pathsep + os.environ.get("PATH", "")

import sounddevice as sd
from scipy.io.wavfile import write
import whisper

# --- 設定 ---
DURATION = 5  # 録音時間（5秒）
SAMPLE_RATE = 44100
AUDIO_FILE = "input_speech.wav"

# 1. マイクから音声を録音
print(f"🎤 {DURATION} 秒間、マイクに向かって喋ってください...")
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()
write(AUDIO_FILE, SAMPLE_RATE, recording)
print("   -> 録音完了！")

# 2. Whisperモデルの読み込みと推論
print("🤖 音声認識中 (Whisper)...")
model = whisper.load_model("base")
result = model.transcribe(AUDIO_FILE)
recognized_text = result["text"]
print(f"【文字起こし結果】: {recognized_text}")

# 3. 日本語に翻訳（deep-translatorを使用）
print("🌐 日本語に翻訳中...")
translated_text = GoogleTranslator(source='auto', target='ja').translate(recognized_text)
print(f"【翻訳結果 (日本語)】: {translated_text}")