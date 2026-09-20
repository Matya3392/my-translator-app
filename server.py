import os
import tempfile
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
        # 1. Groq Whisper の Direct Translation 機能を使用
        with open(temp_input_path, "rb") as audio_file:
            translation = client.audio.translations.create(
                file=(temp_input_path, audio_file.read()),
                model="whisper-large-v3",
                response_format="json",
            )

        translated_text = translation.text.strip()
        print(f"翻訳テキスト: {translated_text}")

        if not translated_text:
            translated_text = "Could not hear any audio."

        # 2. gTTS で英語音声を生成 (TTS)
        tts = gTTS(text=translated_text, lang="en")
        temp_output_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp3"
        ).name
        tts.save(temp_output_path)

        # ヘッダーの値を余計なスペースや記号を含まないシンプルな英数字・ASCIIのみにする
        return FileResponse(
            temp_output_path,
            media_type="audio/mpeg",
            headers={
                "X-Recognized-Text": "Translated",
                "X-Translated-Text": translated_text.encode("ascii", "ignore").decode("ascii"),
            },
        )

    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)