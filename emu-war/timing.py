#!/usr/bin/env python3
"""
Timing alignment: align script phrases to voiceover audio.
Uses faster-whisper if available, otherwise falls back to silence detection + DP alignment.
"""
import sys
import json
import os
import subprocess
import tempfile
from pathlib import Path

def run_whisper(audio_path, script_path):
    """Try to use faster-whisper for word-level timestamps."""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, word_timestamps=True, language="en")
        return result
    except ImportError:
        pass
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, word_timestamps=True, language="en")
        words = []
        for seg in segments:
            for w in seg.words:
                words.append({"word": w.word, "start": w.start, "end": w.end, "probability": w.probability})
        return {"segments": [{"words": words}]}
    except ImportError:
        pass
    return None

def silence_detect(audio_path):
    """Detect speech segments using ffmpeg silence detection."""
    cmd = [
        "ffmpeg", "-i", audio_path, "-af",
        "silencedetect=noise=-30dB:d=0.1",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = result.stdout

    segments = []
    current_start = None
    for line in output.split('\n'):
        if "silence_start:" in line:
            t = float(line.split("silence_start:")[1].split()[0])
            if current_start is not None:
                segments.append((current_start, t))
            current_start = None
        elif "silence_end:" in line:
            t = float(line.split("silence_end:")[1].split()[0])
            current_start = t
    if current_start is not None:
        cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
        dur = float(subprocess.run(cmd_dur, capture_output=True, text=True).stdout.strip())
        segments.append((current_start, dur))
    return segments

def split_phrases(script_text):
    """Split script into phrases at punctuation."""
    import re
    phrases = re.split(r'(?<=[.!?])\s+', script_text.strip())
    phrases = [p.strip() for p in phrases if p.strip()]
    return phrases

def count_syllables(text):
    """Rough syllable count for English."""
    import re
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    count = 0
    for w in words:
        if not w:
            continue
        vc = len(re.findall(r'[aeiouy]+', w))
        if w.endswith('e') and len(w) > 2 and w[-2] not in 'aeiouy':
            vc -= 1
        count += max(1, vc)
    return count

def align_phrases_to_segments(phrases, segments, audio_duration):
    """Dynamic programming alignment of phrases to speech segments.
    Allows phrases to span multiple segments and skips leading/trailing silence.
    """
    n = len(phrases)
    m = len(segments)

    syl = [count_syllables(p) for p in phrases]
    total_syl = sum(syl)

    seg_dur = [e - s for s, e in segments]
    total_speech = sum(seg_dur)

    if total_speech == 0 or total_syl == 0:
        avg = audio_duration / n
        return [(i * avg, (i + 1) * avg, p) for i, p in enumerate(phrases)]

    sec_per_syl = total_speech / total_syl
    phrase_dur = [s * sec_per_syl for s in syl]

    # DP: dp[i][j] = min cost to align first i phrases using first j segments
    INF = 1e9
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    prev = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    # Allow skipping initial silence segments
    for j in range(1, m + 1):
        dp[0][j] = 0
        prev[0][j] = (0, j - 1, 0)  # skip segment

    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] == INF:
                continue
            # Option 1: skip segment j (silence between phrases)
            if j < m:
                if dp[i][j + 1] > dp[i][j] + 0.1:
                    dp[i][j + 1] = dp[i][j] + 0.1
                    prev[i][j + 1] = (i, j, 0)

            # Option 2: assign phrase i to segment j (1:1)
            if i < n and j < m:
                target = phrase_dur[i]
                actual = seg_dur[j]
                penalty = abs(target - actual) * 10
                if phrases[i].strip()[-1] in '.!?':
                    penalty += 0.1 * (1 - (actual / (target + 0.001))**2)
                if target > actual * 3:
                    penalty += 50
                if dp[i + 1][j + 1] > dp[i][j] + penalty:
                    dp[i + 1][j + 1] = dp[i][j] + penalty
                    prev[i + 1][j + 1] = (i, j, 1)

            # Option 3: phrase i spans segments j and j+1
            if i < n and j + 1 < m:
                target = phrase_dur[i]
                actual = seg_dur[j] + seg_dur[j + 1]
                penalty = abs(target - actual) * 10 + 2
                if target > actual * 2:
                    penalty += 50
                if dp[i + 1][j + 2] > dp[i][j] + penalty:
                    dp[i + 1][j + 2] = dp[i][j] + penalty
                    prev[i + 1][j + 2] = (i, j, 2)

            # Option 4: merge phrases i and i+1 into segment j
            if i + 2 <= n and j < m:
                target = phrase_dur[i] + phrase_dur[i + 1]
                actual = seg_dur[j]
                penalty = abs(target - actual) * 10 + 5
                if dp[i + 2][j + 1] > dp[i][j] + penalty:
                    dp[i + 2][j + 1] = dp[i][j] + penalty
                    prev[i + 2][j + 1] = (i, j, 3)

    # Find best ending (allow trailing silence)
    best_j = m
    best_cost = dp[n][m]
    for j in range(m + 1):
        if dp[n][j] < best_cost:
            best_cost = dp[n][j]
            best_j = j

    if best_cost >= INF:
        # Fallback: even split across speech time
        print("DP failed, using fallback alignment")
        phrase_times = []
        cur = 0
        for i, (dur, phrase) in enumerate(zip(phrase_dur, phrases)):
            start = cur
            end = cur + dur
            phrase_times.append((start, end, phrase))
            cur = end
        return phrase_times

    # Backtrack
    assignments = []
    i, j = n, best_j
    while i > 0 or j > 0:
        p = prev[i][j]
        if p is None:
            break
        pi, pj, k = p
        if k == 0:
            # skipped segment
            j = pj
            continue
        elif k == 1:
            # phrase i-1 -> segment j-1
            assignments.append((pi, i, pj))
        elif k == 2:
            # phrase i-1 -> segments j-2, j-1
            assignments.append((pi, i, pj))
        elif k == 3:
            # phrases i-2, i-1 -> segment j-1
            assignments.append((pi, i, pj))
        i, j = pi, pj
    assignments.reverse()

    # Build phrase timings
    phrase_times = []
    for pi, i_end, seg_idx in assignments:
        if i_end == pi + 1:
            # Single phrase
            s, e = segments[seg_idx]
            phrase_times.append((s, e, phrases[pi]))
        elif i_end == pi + 2:
            # Two phrases in one segment - split by syllable ratio
            total_s = sum(syl[pi:i_end])
            s, e = segments[seg_idx]
            cur = s
            for k in range(pi, i_end):
                dur = (e - s) * syl[k] / total_s
                phrase_times.append((cur, cur + dur, phrases[k]))
                cur += dur
        else:
            # Single phrase spanning multiple segments (k=2)
            s = segments[seg_idx][0]
            e = segments[seg_idx + 1][1]
            phrase_times.append((s, e, phrases[pi]))

    return phrase_times

def main():
    if len(sys.argv) < 5:
        print("Usage: python3 timing.py <voice.mp3> <script.txt> --out <work_dir>")
        sys.exit(1)

    audio_path = sys.argv[1]
    script_path = sys.argv[2]
    out_dir = sys.argv[4]

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(script_path, 'r') as f:
        script_text = f.read().strip()

    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    audio_duration = float(result.stdout.strip())

    whisper_result = run_whisper(audio_path, script_path)

    phrases = split_phrases(script_text)

    if whisper_result and whisper_result.get("segments"):
        words = whisper_result["segments"][0].get("words", [])
        phrase_times = []
        word_idx = 0
        for phrase in phrases:
            phrase_words = phrase.split()
            if not phrase_words:
                continue
            n_words = len(phrase_words)
            if word_idx + n_words <= len(words):
                start = words[word_idx]["start"]
                end = words[word_idx + n_words - 1]["end"]
                phrase_times.append((start, end, phrase))
                word_idx += n_words
            else:
                if word_idx < len(words):
                    start = words[word_idx]["start"]
                    end = words[-1]["end"]
                    phrase_times.append((start, end, phrase))
                    word_idx = len(words)
                else:
                    phrase_times.append((audio_duration * 0.9, audio_duration, phrase))
    else:
        print("Whisper not available, using silence detection + DP alignment...")
        segments = silence_detect(audio_path)
        if not segments:
            segments = [(0, audio_duration)]
        phrase_times = align_phrases_to_segments(phrases, segments, audio_duration)

    total_frames = round(audio_duration * 30)

    phrases_json = []
    for i, (start, end, text) in enumerate(phrase_times):
        phrases_json.append({
            "index": i,
            "start": start,
            "end": end,
            "duration": end - start,
            "text": text,
            "start_frame": round(start * 30),
            "end_frame": round(end * 30)
        })

    with open(os.path.join(out_dir, "phrases.json"), 'w') as f:
        json.dump({
            "audio_duration": audio_duration,
            "total_frames": total_frames,
            "phrases": phrases_json
        }, f, indent=2)

    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Total frames: {total_frames}")
    print(f"Phrases: {len(phrases_json)}")
    for p in phrases_json:
        print(f"  [{p['start']:.2f}-{p['end']:.2f}] {p['text']}")

if __name__ == "__main__":
    main()
