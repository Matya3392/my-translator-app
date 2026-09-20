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
        # 1. Groq Whisper API で文字起こし (STT)
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
            translated_text = "Could not hear any audio."
        else:
            # 2. Groq LLM で日本語->英語翻訳
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator. Translate the following Japanese text into natural English. Respond ONLY with the translation, no extra text.",
                    },
                    {"role": "user", "content": recognized_text},
                ],
                temperature=0.3,
            )
            translated_text = response.choices[0].message.content.strip()

        print(f"翻訳テキスト: {translated_text}")

        # 3. gTTS で英語音声を生成 (TTS)
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