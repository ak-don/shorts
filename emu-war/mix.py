#!/usr/bin/env python3
"""
mix.py — Mix voice, music, and SFX at exact levels
Music: 50% of voice gated RMS (-6 dB)
SFX: 50% of voice gated RMS, peaks capped at 0.7x voice peak
"""
import sys
import os
import numpy as np
import wave

def read_wav(filename):
    with wave.open(filename, 'rb') as wf:
        assert wf.getnchannels() == 1, "Must be mono"
        assert wf.getsampwidth() == 2, "Must be 16-bit"
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return data, rate

def write_wav(filename, data, rate=44100):
    data = np.clip(data, -1, 1)
    int_data = (data * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(int_data.tobytes())

def gated_rms(audio, gate_db=-40):
    gate = 10 ** (gate_db / 20)
    masked = audio[np.abs(audio) > gate]
    if len(masked) == 0:
        return 1e-10
    return np.sqrt(np.mean(masked ** 2))

def peak_level(audio):
    return np.max(np.abs(audio))

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 mix.py <voice.mp3> <music.wav> <sfx.wav> [output.wav]")
        sys.exit(1)

    voice_file = sys.argv[1]
    music_file = sys.argv[2]
    sfx_file = sys.argv[3]
    output_file = sys.argv[4] if len(sys.argv) > 4 else 'mix.wav'

    # Convert voice mp3 to wav if needed
    if voice_file.endswith('.mp3'):
        import subprocess
        temp_wav = '/tmp/voice_temp.wav'
        subprocess.run(['ffmpeg', '-y', '-i', voice_file, temp_wav], check=True, capture_output=True)
        voice_file = temp_wav

    print("Reading files...")
    voice, rate = read_wav(voice_file)
    music, _ = read_wav(music_file)
    sfx, _ = read_wav(sfx_file)

    target_len = len(voice)
    if len(music) != target_len:
        if len(music) > target_len:
            music = music[:target_len]
        else:
            music = np.pad(music, (0, target_len - len(music)))
    if len(sfx) != target_len:
        if len(sfx) > target_len:
            sfx = sfx[:target_len]
        else:
            sfx = np.pad(sfx, (0, target_len - len(sfx)))

    voice_rms = gated_rms(voice)
    voice_peak = peak_level(voice)
    print(f"Voice: gated RMS = {voice_rms:.6f} ({20*np.log10(voice_rms):.2f} dB), peak = {voice_peak:.6f}")

    music_rms = gated_rms(music)
    music_gain = (voice_rms * 0.5) / music_rms if music_rms > 0 else 0
    music = music * music_gain
    print(f"Music: gain = {music_gain:.4f}, resulting RMS = {gated_rms(music):.6f} ({20*np.log10(gated_rms(music)):.2f} dB)")

    sfx_rms = gated_rms(sfx)
    sfx_gain = (voice_rms * 0.5) / sfx_rms if sfx_rms > 0 else 0
    sfx = sfx * sfx_gain

    sfx_peak = peak_level(sfx)
    max_peak = voice_peak * 0.7
    if sfx_peak > max_peak:
        sfx = sfx * (max_peak / sfx_peak)
        print(f"SFX: peak capped from {sfx_peak:.6f} to {max_peak:.6f}")

    print(f"SFX: gain = {sfx_gain:.4f}, resulting RMS = {gated_rms(sfx):.6f} ({20*np.log10(gated_rms(sfx)):.2f} dB), peak = {peak_level(sfx):.6f}")

    mix = voice + music + sfx

    mix_peak = peak_level(mix)
    if mix_peak > 0.95:
        mix = mix * (0.95 / mix_peak)
        print(f"Mix peak limited: {mix_peak:.6f} -> 0.95")

    write_wav(output_file, mix, rate)
    print(f"Mixed output: {output_file}")
    print(f"Final mix RMS: {gated_rms(mix):.6f} ({20*np.log10(gated_rms(mix)):.2f} dB), peak: {peak_level(mix):.6f}")

    print("\n=== LEVEL VERIFICATION ===")
    print(f"Music vs Voice: {20*np.log10(gated_rms(music)/voice_rms):.2f} dB (target: -6.0 dB)")
    print(f"SFX vs Voice: {20*np.log10(gated_rms(sfx)/voice_rms):.2f} dB (target: -6.0 dB or lower)")
    print(f"SFX peak / Voice peak: {peak_level(sfx)/voice_peak:.3f} (target: <= 0.7)")

if __name__ == '__main__':
    main()
