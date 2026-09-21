import base64
import os
import tempfile
import traceback
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from groq import Groq

app = FastAPI()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>index.html が見つかりません</h1>"


@app.post("/translate")
async def translate_audio(
    file: UploadFile = File(...), target_lang: str = Form("en")
):
    try:
        content = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".webm"
        ) as temp_input:
            temp_input.write(content)
            temp_input_path = temp_input.name

        try:
            # 1. 日本語の文字起こし (Whisper Transcriptions)
            with open(temp_input_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    file=(temp_input_path, audio_file.read()),
                    model="whisper-large-v3",
                    response_format="json",
                )
            recognized_text = transcript.text.strip()
            print(f"認識テキスト: {recognized_text}")

            # 2. 英語への自動翻訳 (Whisper Translations - GoogleやLLMを使わない)
            with open(temp_input_path, "rb") as audio_file:
                translation = client.audio.translations.create(
                    file=(temp_input_path, audio_file.read()),
                    model="whisper-large-v3",
                    response_format="json",
                )
            translated_text = translation.text.strip()
            print(f"翻訳テキスト: {translated_text}")

            if not recognized_text:
                recognized_text = "音声が聞き取れませんでした"
                translated_text = "-"

            # テキストのみを綺麗に返す (音声合成はブラウザ標準機能を使用)
            return JSONResponse(
                content={
                    "recognized": recognized_text,
                    "translated": translated_text,
                }
            )

        finally:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)

    except Exception as e:
        print("=== エラー発生 ===")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)