import os
import shutil
import asyncio
import urllib.request
import urllib.parse
import json
import subprocess
import time
import threading
import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import edge_tts
import soundfile as sf

# --- 1. ffmpeg.exe の自動準備 ---
project_ffmpeg = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
if not os.path.exists(project_ffmpeg):
    import imageio_ffmpeg
    original_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    shutil.copy(original_ffmpeg, project_ffmpeg)

os.environ["PATH"] = os.path.dirname(__file__) + os.pathsep + os.environ.get("PATH", "")

# --- 2. 設定・パス関連 ---
SAMPLE_RATE = 44100
MAX_DURATION = 30
SILENCE_THRESHOLD = 0.015
SILENCE_DURATION = 1.5

INPUT_SPEECH_WAV = "input_speech.wav"
TTS_OUTPUT_WAV = "tts_output.wav"
RVC_OUTPUT_WAV = "rvc_output.wav"

RVC_DIR = r"C:\Users\bless\Downloads\RVC20260718Nvidia50x0\RVC20260718Nvidia50x0"
RVC_PYTHON = os.path.join(RVC_DIR, "runtime", "python.exe")
if not os.path.exists(RVC_PYTHON):
    RVC_PYTHON = "python"

RVC_MODEL_PATH = r"C:\Users\bless\Downloads\RVC20260718Nvidia50x0\RVC20260718Nvidia50x0\assets\weights\G_756.pth"
RVC_INDEX_PATH = r"C:\Users\bless\Downloads\RVC20260718Nvidia50x0\RVC20260718Nvidia50x0\logs\my test voice\added_IVF170_Flat_nprobe_1_my test voice_v2.index"
PITCH_SHIFT = 0

whisper_model = None


def clean_temp_files():
    for f in [INPUT_SPEECH_WAV, TTS_OUTPUT_WAV, RVC_OUTPUT_WAV]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass


def record_audio_vad(status_label):
    status_label.config(text="🎤 録音中... マイクに向かって喋ってください（英語でも日本語でもOK）")
    chunk_size = 1024
    recorded_chunks = []
    
    silent_chunks = 0
    silent_chunk_limit = int((SAMPLE_RATE / chunk_size) * SILENCE_DURATION)
    has_spoken = False
    
    start_time = time.time()
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        while True:
            data, overflowed = stream.read(chunk_size)
            recorded_chunks.append(data.copy())
            
            rms = np.sqrt(np.mean(data**2))
            
            if rms > SILENCE_THRESHOLD:
                has_spoken = True
                silent_chunks = 0
            else:
                if has_spoken:
                    silent_chunks += 1
            
            if has_spoken and silent_chunks >= silent_chunk_limit:
                break
                
            if time.time() - start_time > MAX_DURATION:
                break

    if not recorded_chunks or not has_spoken:
        return False

    audio_data = np.concatenate(recorded_chunks, axis=0)
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    write(INPUT_SPEECH_WAV, SAMPLE_RATE, audio_data_int16)
    return True


# --- 多言語対応 翻訳関数 ---
def translate_text(text, target_lang):
    if not text.strip():
        return ""
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={target_lang}&dt=t&q=" + urllib.parse.quote(text)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return "".join([item[0] for item in result[0] if item[0]])
    except Exception:
        return text


# --- 多言語対応 音声合成 (edge-tts) ---
async def generate_tts(text, voice_name, output_file):
    communicate = edge_tts.Communicate(text, voice_name)
    await communicate.save(output_file)


def convert_voice_rvc(input_wav, output_wav):
    if not os.path.exists(RVC_MODEL_PATH):
        shutil.copy(input_wav, output_wav)
        return

    cli_script = os.path.join(RVC_DIR, "infer", "rtrvc.py")
    if not os.path.exists(cli_script):
        shutil.copy(input_wav, output_wav)
        return

    cmd = [
        RVC_PYTHON, cli_script,
        "--f0up_key", str(PITCH_SHIFT),
        "--input_path", os.path.abspath(input_wav),
        "--opt_path", os.path.abspath(output_wav),
        "--model_name", os.path.basename(RVC_MODEL_PATH),
        "--index_path", RVC_INDEX_PATH,
        "--f0method", "rmvpe"
    ]

    try:
        subprocess.run(cmd, cwd=RVC_DIR, capture_output=True, text=True, check=True)
    except Exception:
        shutil.copy(input_wav, output_wav)


# --- 双方向通訳パイプライン ---
def run_translation_pipeline(status_label, detected_lang_var, src_text_var, target_text_var, record_btn):
    record_btn.config(state="disabled")
    clean_temp_files()

    if not record_audio_vad(status_label):
        status_label.config(text="⚠️ 音声が検知されませんでした。")
        record_btn.config(state="normal")
        return

    # 1. 音声認識 & 言語自動検知 (language を指定しない)
    status_label.config(text="🤖 言語判定・音声認識中 (Whisper)...")
    result = whisper_model.transcribe(INPUT_SPEECH_WAV)
    
    recognized_text = result["text"].strip()
    detected_lang = result.get("language", "en")  # 'en' か 'ja' などが自動検出される
    
    if not recognized_text:
        status_label.config(text="⚠️ 音声を認識できませんでした。")
        record_btn.config(state="normal")
        return

    src_text_var.set(recognized_text)

    # 2. 言語判定による分岐処理
    if detected_lang == "ja":
        detected_lang_var.set("検出言語: 日本語 ➔ 英語へ通訳")
        target_lang = "en"
        tts_voice = "en-US-JennyNeural"  # 英語読み上げ用音声
        use_rvc = False                 # 英語音声にRVCを使う場合は True に変更可能
    else:
        detected_lang_var.set("検出言語: 英語 (またはその他) ➔ 日本語へ通訳")
        target_lang = "ja"
        tts_voice = "ja-JP-NanamiNeural" # 日本語読み上げ用音声
        use_rvc = True                  # 日本語モデルへのRVCボイス変換を適用

    # 3. 翻訳
    status_label.config(text=f"🌐 翻訳中 ({detected_lang} ➔ {target_lang})...")
    translated_text = translate_text(recognized_text, target_lang)
    target_text_var.set(translated_text)

    # 4. 音声合成 (edge-tts)
    status_label.config(text="🔊 音声生成中 (edge-tts)...")
    asyncio.run(generate_tts(translated_text, tts_voice, TTS_OUTPUT_WAV))

    # 5. RVC変換（日本語への変換時のみ、または設定に応じて適用）
    play_target = TTS_OUTPUT_WAV
    if use_rvc:
        status_label.config(text="🎤 RVCボイス変換中...")
        convert_voice_rvc(TTS_OUTPUT_WAV, RVC_OUTPUT_WAV)
        if os.path.exists(RVC_OUTPUT_WAV):
            play_target = RVC_OUTPUT_WAV

    # 6. 音声再生
    status_label.config(text="▶️ 変換音声の再生中...")
    data, samplerate = sf.read(play_target)
    sd.play(data, samplerate)
    sd.wait()
    sd.stop()

    status_label.config(text="✨ 完了！「通訳開始」ボタンで次の発言を録音できます。")
    record_btn.config(state="normal")


def load_whisper_bg(status_label, record_btn):
    global whisper_model
    status_label.config(text="🤖 AIモデルを読み込み中... しばらくお待ちください")
    whisper_model = whisper.load_model("base")
    status_label.config(text="👍 準備完了！「通訳開始」ボタンを押してください。")
    record_btn.config(state="normal")


# --- GUI画面 ---
def main():
    root = tk.Tk()
    root.title("ローカルリアルタイム双方向音声通訳アプリ")
    root.geometry("600x460")
    root.resizable(False, False)

    style = ttk.Style()
    style.theme_use('clam')

    title_label = ttk.Label(root, text="🎙️ 自動言語判定・双方向ボイス通訳", font=("Helvetica", 15, "bold"))
    title_label.pack(pady=12)

    status_label = ttk.Label(root, text="初期化中...", font=("Helvetica", 10), foreground="#0055ff")
    status_label.pack(pady=3)

    # 検出言語の表示枠
    detected_lang_var = tk.StringVar(value="検出言語: 待機中")
    lang_label = ttk.Label(root, textvariable=detected_lang_var, font=("Helvetica", 10, "bold"), foreground="#2e7d32")
    lang_label.pack(pady=5)

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill="both", expand=True, padx=20, pady=5)

    # 話した言葉（原文）
    ttk.Label(frame, text="【発言内容 (認識テキスト)】:", font=("Helvetica", 10, "bold")).pack(anchor="w")
    src_text_var = tk.StringVar(value="")
    src_entry = ttk.Entry(frame, textvariable=src_text_var, font=("Helvetica", 11), state="readonly")
    src_entry.pack(fill="x", pady=(2, 15))

    # 通訳された言葉（翻訳文）
    ttk.Label(frame, text="【通訳結果 (翻訳テキスト)】:", font=("Helvetica", 10, "bold")).pack(anchor="w")
    target_text_var = tk.StringVar(value="")
    target_entry = ttk.Entry(frame, textvariable=target_text_var, font=("Helvetica", 11), state="readonly")
    target_entry.pack(fill="x", pady=(2, 10))

    def on_record_click():
        threading.Thread(
            target=run_translation_pipeline,
            args=(status_label, detected_lang_var, src_text_var, target_text_var, record_btn),
            daemon=True
        ).start()

    record_btn = ttk.Button(root, text="🎤 通訳開始 (録音)", command=on_record_click, state="disabled")
    record_btn.pack(pady=15, ipadx=10, ipady=5)

    threading.Thread(target=load_whisper_bg, args=(status_label, record_btn), daemon=True).start()

    root.mainloop()


if __name__ == "__main__":
    main()