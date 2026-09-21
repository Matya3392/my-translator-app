import base64
import os
import tempfile
import traceback
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from groq import Groq
from gtts import gTTS

app = FastAPI()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

LANG_NAMES = {
    "en": "English",
    "zh-cn": "Simplified Chinese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
}


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
            # 1. Groq Whisper で文字起こし
            with open(temp_input_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    file=(temp_input_path, audio_file.read()),
                    model="whisper-large-v3",
                    response_format="json",
                )

            recognized_text = transcript.text.strip()
            print(f"認識テキスト: {recognized_text}")

            if not recognized_text:
                recognized_text = "音声が聞き取れませんでした"
                translated_text = recognized_text
            else:
                # 2. Groq LLM (llama-3.3-70b-versatile) で翻訳
                target_lang_name = LANG_NAMES.get(target_lang, "English")

                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a professional translator. Translate the given Japanese text accurately into {target_lang_name}. Output ONLY the translated text, with no explanations or extra formatting.",
                        },
                        {"role": "user", "content": recognized_text},
                    ],
                    model="llama-3.3-70b-versatile",
                )

                translated_text = (
                    chat_completion.choices[0].message.content.strip()
                )
                print(f"翻訳テキスト: {translated_text}")

            # 3. gTTS で音声を生成
            tts = gTTS(text=translated_text, lang=target_lang)
            temp_output_path = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp3"
            ).name
            tts.save(temp_output_path)

            with open(temp_output_path, "rb") as f:
                audio_bytes = f.read()

            os.remove(temp_output_path)

            # 音声データを Base64 エンコードして JSON で返却
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            return JSONResponse(
                content={
                    "recognized": recognized_text,
                    "translated": translated_text,
                    "audio": audio_b64,
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