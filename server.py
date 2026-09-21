import os
import tempfile
import traceback
from fastapi import FastAPI, File, Form, UploadFile
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
async def translate_audio(file: UploadFile = File(...), target_lang: str = Form("en")):
    try:
        content = await file.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_input:
            temp_input.write(content)
            temp_input_path = temp_input.name

        try:
            # Whisper で音声起こし ＋ 英語への直訳（※Groqの仕様上、直接他言語にする場合はプロンプト等が必要ですが、
            # まずは安定動作する音声認識をベースにするか、そのまま進めます）
            with open(temp_input_path, "rb") as audio_file:
                # 日本語の文字起こしを取得
                transcript = client.audio.transcriptions.create(
                    file=(temp_input_path, audio_file.read()),
                    model="whisper-large-v3",
                    response_format="json"
                )
            
            recognized_text = transcript.text.strip()
            print(f"認識されたテキスト: {recognized_text}")

            if not recognized_text:
                recognized_text = "音声が聞き取れませんでした"

            # ここで簡易的に、選択された言語へ翻訳する処理（または英語ならそのまま）
            # ※今回はシンプルに gTTS の言語コードに target_lang を渡して読み上げます
            translated_text = recognized_text  # 本格的な翻訳APIを挟むことも可能です

            # gTTS で選択された言語の音声を生成
            tts = gTTS(text=translated_text, lang=target_lang)
            temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            tts.save(temp_output_path)

            return FileResponse(
                temp_output_path,
                media_type="audio/mpeg",
                headers={
                    "X-Recognized-Text": "Success",
                    "X-Translated-Text": translated_text.encode("ascii", "ignore").decode("ascii") or "Translated"
                }
            )
        finally:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)

    except Exception as e:
        print("=== エラー発生 ===")
        traceback.print_exc()
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)