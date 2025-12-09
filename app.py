from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Segment
import subprocess
from env_vars import HUGGINGFACE_TOKEN

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)

# Load models once
asr_model = whisper.load_model("base")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)
os.environ["PATH"] += os.pathsep + "/usr/bin"

# Path to store the audio samples
AUDIO_SAMPLES_PATH = os.path.join(app.root_path, 'static', 'audio_samples')

@app.route('/')
def index():
    # Provide links to the sample audio files
    sample_files = os.listdir(AUDIO_SAMPLES_PATH)
    sample_files = [file for file in sample_files if file.endswith(('.mp3', '.wav'))]
    return render_template('index.html', sample_files=sample_files)

@app.route('/download_sample/<filename>', methods=['GET'])
def download_sample(filename):
    try:
        return send_from_directory(AUDIO_SAMPLES_PATH, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/diarize', methods=['POST'])
def diarize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    input_filename = file.filename
    input_path = os.path.join("uploads", input_filename)
    file.save(input_path)

    filename_no_ext, ext = os.path.splitext(input_filename)
    wav_filename = f"{filename_no_ext}_converted.wav"
    wav_path = os.path.join("uploads", wav_filename)

    try:
        # Convert to WAV (if needed)
        subprocess.run(["ffmpeg", "-y", "-i", input_path, wav_path], check=True)

        # Transcription
        result = asr_model.transcribe(wav_path, language='en', word_timestamps=False)
        segments = result['segments']

        # Diarization
        diarization = diarization_pipeline(wav_path)

        # Match speaker labels
        final_transcript = []

        for segment in segments:
            seg_start = segment['start']
            seg_end = segment['end']
            seg_text = segment['text'].strip()

            speaker_label = "SPEAKER_01"
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= seg_start <= turn.end or turn.start <= seg_end <= turn.end:
                    speaker_label = speaker
                    break

            final_transcript.append(f"{speaker_label}: {seg_text}")

        # Cleanup
        os.remove(input_path)
        os.remove(wav_path)

        return jsonify({"transcript": final_transcript})

    except Exception as e:
        # Cleanup on error
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=9070)
