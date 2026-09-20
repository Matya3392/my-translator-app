import os
import shutil
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from googletrans import Translator
from gtts import gTTS
from groq import Groq

app = FastAPI()

# CORS（すべての通信を許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイル配信（www フォルダ）
OS_DIR = "www"
os.makedirs(OS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="www"), name="static")

INPUT_SPEECH_WAV = "input_speech.wav"
translator = Translator()

# Groq APIクライアントの初期化（環境変数 GROQ_API_KEY から取得）
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

@app.get("/")
def read_root():
    if os.path.exists("www/index.html"):
        return FileResponse("www/index.html")
    elif os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "ok"}

@app.post("/translate")
async def process_audio(
    file: UploadFile = File(...),
    target_lang: str = Form("ja")
):
    try:
        # 1. アップロードされた音声を保存
        with open(INPUT_SPEECH_WAV, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. 音声認識 (Groq経由Whisper または フォールバック)
        recognized_text = ""
        if groq_client:
            with open(INPUT_SPEECH_WAV, "rb") as audio_file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(INPUT_SPEECH_WAV, audio_file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="json",
                    language="ja"
                )
                recognized_text = transcription.text.strip()
        else:
            return {
                "status": "error",
                "message": "Groq APIキーが設定されていません。環境変数GROQ_API_KEYを設定してください。"
            }

        if not recognized_text:
            return {"status": "error", "message": "音声が認識できませんでした"}

        # 3. Google Translateで翻訳
        translated_res = translator.translate(recognized_text, dest=target_lang)
        translated_text = translated_res.text

        # 4. gTTSで音声合成
        audio_url = None
        try:
            tts_filename = "output_tts.mp3"
            tts_path = os.path.join("www", tts_filename)
            
            tts_lang = target_lang
            if tts_lang == "zh-cn":
                tts_lang = "zh-CN"
            elif tts_lang == "zh-tw":
                tts_lang = "zh-TW"

            tts = gTTS(text=translated_text, lang=tts_lang)
            tts.save(tts_path)
            audio_url = f"/static/{tts_filename}"
        except Exception as tts_err:
            print(f"TTS生成エラー: {tts_err}")

        # 5. 結果返却
        return {
            "status": "success",
            "recognized_text": recognized_text,
            "translated_text": translated_text,
            "audio_url": audio_url
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}