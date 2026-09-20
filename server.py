import os
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from googletrans import Translator
from groq import Groq
from gtts import gTTS

app = FastAPI()

# Groq クライアントの初期化（環境変数 GROQ_API_KEY を使用）
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 翻訳エンジンの初期化
translator = Translator()


@app.get("/", response_class=HTMLResponse)
async def read_index():
    """index.html を返すルート"""
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>index.html が見つかりません</h1>"


@app.post("/translate-audio")
async def translate_audio(file: UploadFile = File(...)):
    """録音データを受け取り、STT -> 翻訳 -> TTS を行って音声ファイルを返す"""
    # 1. 送信された音声データを一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_input:
        content = await file.read()
        temp_input.write(content)
        temp_input_path = temp_input.name

    try:
        # 2. Groq Whisper API で文字起こし (STT)
        with open(temp_input_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(temp_input_path, audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="json",
            )
        recognized_text = transcription.text
        print(f"認識テキスト: {recognized_text}")

        if not recognized_text.strip():
            recognized_text = "音声が聞き取れませんでした。"

        # 3. Google翻訳で日本語から英語へ翻訳
        translated = translator.translate(recognized_text, src="ja", dest="en")
        translated_text = translated.text
        print(f"翻訳テキスト: {translated_text}")

        # 4. gTTS で英語音声を生成 (TTS)
        tts = gTTS(text=translated_text, lang="en")
        temp_output_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp3"
        ).name
        tts.save(temp_output_path)

        # 5. 生成した音声ファイルをレスポンスとして返す
        return FileResponse(
            temp_output_path,
            media_type="audio/mpeg",
            headers={
                "X-Recognized-Text": recognized_text,
                "X-Translated-Text": translated_text,
            },
        )

    finally:
        # 一時ファイルの削除
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)