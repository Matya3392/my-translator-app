import os
import tempfile
from deep_translator import MyMemoryTranslator
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from groq import Groq
from gtts import gTTS

app = FastAPI()

# Groq クライアントの初期化
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>index.html が見つかりません</h1>"


@app.post("/translate")
async def translate_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_input:
        content = await file.read()
        temp_input.write(content)
        temp_input_path = temp_input.name

    try:
        # 1. Groq Whisper API で文字起こし
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

        # 2. MyMemoryTranslator で日本語から英語へ翻訳（Googleの制限回避）
        translated_text = MyMemoryTranslator(
            source="ja-JP", target="en-GB"
        ).translate(recognized_text)
        print(f"翻訳テキスト: {translated_text}")

        # 3. gTTS で英語音声を生成
        tts = gTTS(text=translated_text, lang="en")
        temp_output_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp3"
        ).name
        tts.save(temp_output_path)

        return FileResponse(
            temp_output_path,
            media_type="audio/mpeg",
            headers={
                "X-Recognized-Text": recognized_text,
                "X-Translated-Text": translated_text,
            },
        )

    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)