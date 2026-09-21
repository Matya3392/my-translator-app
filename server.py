import os
import tempfile
import traceback
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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
            # 1. Groq Whisper で日本語の音声を文字起こし
            with open(temp_input_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    file=(temp_input_path, audio_file.read()),
                    model="whisper-large-v3",
                    response_format="json"
                )
            
            recognized_text = transcript.text.strip()
            print(f"認識テキスト: {recognized_text}")

            if not recognized_text:
                recognized_text = "音声が聞き取れませんでした"

            # 2. 翻訳テキスト（今の段階では認識された日本語をそのまま、または必要に応じて）
            translated_text = recognized_text

            # 3. gTTS で音声を生成
            tts = gTTS(text=translated_text, lang=target_lang)
            temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            tts.save(temp_output_path)

            # 音声ファイルをバイナリとして読み込む
            with open(temp_output_path, "rb") as f:
                audio_bytes = f.read()

            # 一時ファイルの削除
            os.remove(temp_output_path)

            # JSON としてテキストと音声（Base64等）を返すか、あるいは別ルートにするのが一番確実ですが、
            # 今回はシンプルに、テキストはカスタムヘッダーに URLEncode して日本語が消えないように送ります！
            import urllib.parse
            encoded_recognized = urllib.parse.quote(recognized_text)
            encoded_translated = urllib.parse.quote(translated_text)

            from fastapi.responses import FileResponse
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_out_res:
                temp_out_res.write(audio_bytes)
                temp_out_path = temp_out_res.name

            return FileResponse(
                temp_out_path,
                media_type="audio/mpeg",
                headers={
                    "X-Recognized-Text": encoded_recognized,
                    "X-Translated-Text": encoded_translated
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