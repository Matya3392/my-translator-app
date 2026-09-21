import os
import tempfile
import traceback
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from groq import Groq
from gtts import gTTS

app = FastAPI()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>index.html が見つかりません</h1>"

@app.post("/translate")
async def translate_audio(file: UploadFile = File(...)):
    try:
        content = await file.read()
        print(f"受信したファイルサイズ: {len(content)} バイト")
        
        # ブラウザからの音声形式（webm等）に合わせて一時ファイルを保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_input:
            temp_input.write(content)
            temp_input_path = temp_input.name

        try:
            # Groq の Whisper API (translations) で直接英語に翻訳
            with open(temp_input_path, "rb") as audio_file:
                translation = client.audio.translations.create(
                    file=(temp_input_path, audio_file.read()),
                    model="whisper-large-v3",
                    response_format="json"
                )
            
            translated_text = translation.text.strip()
            print(f"翻訳結果: {translated_text}")

            if not translated_text:
                translated_text = "Could not hear any audio."
                
            # gTTS で音声化
            tts = gTTS(text=translated_text, lang="en")
            temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            tts.save(temp_output_path)

            return FileResponse(
                temp_output_path,
                media_type="audio/mpeg",
                headers={
                    "X-Recognized-Text": "Translated",
                    "X-Translated-Text": "OK"
                }
            )
        finally:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)

    except Exception as e:
        print("=== サーバー内部エラー発生 ===")
        traceback.print_exc()
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)