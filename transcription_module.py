import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import tempfile
import time
import whisper


class AudioTranscriber:
    """Класс для транскрипции аудио с разбиением по тишине"""

    def __init__(self, language: str = "ru-RU", min_silence_len: int = 500,
                 silence_thresh_offset: int = 10, keep_silence: int = 300,
                 max_retries: int = 3, retry_delay: float = 1.0):
        self.recognizer = sr.Recognizer()
        self.language = language
        self.min_silence_len = min_silence_len
        self.silence_thresh_offset = silence_thresh_offset
        self.keep_silence = keep_silence
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.whisper_model = whisper.load_model("base")

    def _recognize_with_retry(self, audio_data: sr.AudioData):
        for attempt in range(self.max_retries):
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_data.get_wav_data())
                    tmp_path = tmp.name
                try:
                    result = self.whisper_model.transcribe(tmp_path, language=self.language[:2])
                    text = result["text"]
                finally:
                    os.unlink(tmp_path)
                return text.strip()
            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return None
        return None

    def transcribe_audio(self, audio_path: str, output_txt_path: str = None, delete_chunks: bool = True):
        if output_txt_path is None:
            output_txt_path = audio_path + ".txt"

        sound = AudioSegment.from_file(audio_path)

        silence_thresh = sound.dBFS - self.silence_thresh_offset
        raw_chunks = split_on_silence(
            sound,
            min_silence_len=self.min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=self.keep_silence
        )

        min_chunk_len = 500
        chunks = [chunk for chunk in raw_chunks if len(chunk) >= min_chunk_len]

        temp_dir = tempfile.mkdtemp(prefix="audio_chunks_")
        full_transcript = ""

        with open(output_txt_path, "w", encoding="utf-8") as out_f:
            for i, chunk in enumerate(chunks, start=1):
                chunk_filename = os.path.join(temp_dir, f"chunk_{i:04d}.wav")
                chunk.export(chunk_filename, format="wav")

                with sr.AudioFile(chunk_filename) as source:
                    audio_data = self.recognizer.record(source)
                    text = self._recognize_with_retry(audio_data)

                if text:
                    if not text.endswith(('.', '!', '?')):
                        text += '.'
                    formatted_text = f"{text.capitalize()} "
                    full_transcript += formatted_text
                    out_f.write(formatted_text)

                if delete_chunks:
                    os.remove(chunk_filename)

        os.rmdir(temp_dir)

def main():
    audio_paths = [""]

    transcriber = AudioTranscriber(
        language="ru-RU",
        min_silence_len=500,
        silence_thresh_offset=10,
        keep_silence=300,
        max_retries=3,
        retry_delay=1.0
    )

    for audio_path in audio_paths:
        transcriber.transcribe_audio(audio_path, delete_chunks=True)


if __name__ == "__main__":
    main()