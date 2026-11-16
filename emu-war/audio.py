#!/usr/bin/env python3
"""
audio.py — Synthesize music and SFX stems
Outputs: music.wav, sfx.wav (44.1kHz, 16-bit)
"""
import sys
import os
import numpy as np
import wave

SAMPLE_RATE = 44100
DURATION = 36.44
TOTAL_SAMPLES = int(SAMPLE_RATE * DURATION)

def write_wav(filename, data):
    data = np.clip(data, -1, 1)
    int_data = (data * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(int_data.tobytes())

def gated_rms(audio, gate_db=-40):
    gate = 10 ** (gate_db / 20)
    masked = audio[np.abs(audio) > gate]
    if len(masked) == 0:
        return 1e-10
    return np.sqrt(np.mean(masked ** 2))

def sine(freq, duration, amp=1.0, fade=0.01):
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    wave = amp * np.sin(2 * np.pi * freq * t)
    fade_samples = int(fade * SAMPLE_RATE)
    if fade_samples > 0:
        wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
        wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return wave

def noise(duration, color='white', amp=1.0):
    n = int(SAMPLE_RATE * duration)
    return amp * np.random.randn(n) * (0.5 if color == 'pink' else 1.0)

def envelope(duration, attack=0.01, decay=0.1, sustain=0.7, release=0.1):
    n = int(SAMPLE_RATE * duration)
    a = int(attack * SAMPLE_RATE)
    d = int(decay * SAMPLE_RATE)
    r = int(release * SAMPLE_RATE)
    s = n - a - d - r
    env = np.zeros(n)
    if a > 0: env[:a] = np.linspace(0, 1, a)
    if d > 0: env[a:a+d] = np.linspace(1, sustain, d)
    if s > 0: env[a+d:a+d+s] = sustain
    if r > 0: env[-r:] = np.linspace(sustain, 0, r)
    return env

def kick(duration=0.5):
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    freq = 150 * np.exp(-t * 30) + 40
    wave = np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE)
    wave *= envelope(duration, attack=0.001, decay=0.1, sustain=0, release=0.3)
    return wave * 0.8

def snare(duration=0.3):
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    body = np.sin(2 * np.pi * 200 * t) * np.exp(-t * 20)
    noi = noise(duration, 'white', 0.5) * np.exp(-t * 30)
    return (body + noi) * 0.6

def hihat(duration=0.1, closed=True):
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    noi = noise(duration, 'white', 1.0)
    if closed:
        noi *= np.exp(-t * 200)
    else:
        noi *= np.exp(-t * 30)
    return noi * 0.3

def bass_note(freq, duration):
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    wave = np.sin(2 * np.pi * freq * t)
    wave += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
    wave += 0.1 * np.sin(2 * np.pi * freq * 3 * t)
    wave *= envelope(duration, attack=0.005, decay=0.2, sustain=0.3, release=0.3)
    return wave * 0.4

def brass_pad(chord_freqs, duration):
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    wave = np.zeros_like(t)
    for f in chord_freqs:
        wave += np.sin(2 * np.pi * f * t)
        wave += 0.5 * np.sin(2 * np.pi * f * 2 * t)
        wave += 0.25 * np.sin(2 * np.pi * f * 3 * t)
    wave /= len(chord_freqs)
    wave *= envelope(duration, attack=0.1, decay=0.2, sustain=0.8, release=0.5)
    return wave * 0.3

def generate_music():
    BPM = 100
    beat = 60 / BPM
    bar = beat * 4
    loop_duration = bar * 2
    loops_needed = int(DURATION / loop_duration) + 1

    music = np.zeros(TOTAL_SAMPLES)

    for loop_idx in range(loops_needed):
        offset = int(loop_idx * loop_duration * SAMPLE_RATE)
        if offset >= TOTAL_SAMPLES:
            break

        pattern = np.zeros(int(loop_duration * SAMPLE_RATE))

        for beat_idx in [0, 2]:
            t = beat_idx * beat
            k = kick(0.4)
            start = int(t * SAMPLE_RATE)
            end = min(start + len(k), len(pattern))
            pattern[start:end] += k[:end-start]

        for beat_idx in [1, 3]:
            t = beat_idx * beat
            s = snare(0.25)
            start = int(t * SAMPLE_RATE)
            end = min(start + len(s), len(pattern))
            pattern[start:end] += s[:end-start]

        for i in range(8):
            t = i * beat / 2
            h = hihat(0.08, closed=True)
            start = int(t * SAMPLE_RATE)
            end = min(start + len(h), len(pattern))
            pattern[start:end] += h[:end-start]

        bass_notes = [55, 82.4, 55, 82.4]
        for i, freq in enumerate(bass_notes):
            t = i * beat
            b = bass_note(freq, beat * 0.9)
            start = int(t * SAMPLE_RATE)
            end = min(start + len(b), len(pattern))
            pattern[start:end] += b[:end-start]

        chords = [
            [220, 277, 330],
            [165, 220, 247],
            [196, 247, 294],
            [220, 277, 330]
        ]
        for i, chord in enumerate(chords):
            t = i * beat * 2
            p = brass_pad(chord, beat * 2)
            start = int(t * SAMPLE_RATE)
            end = min(start + len(p), len(pattern))
            pattern[start:end] += p[:end-start]

        end = min(offset + len(pattern), TOTAL_SAMPLES)
        music[offset:end] += pattern[:end-offset]

    twist_sample = int(23.93 * SAMPLE_RATE)
    fade_out = int(0.5 * SAMPLE_RATE)
    music[twist_sample:twist_sample+fade_out] *= np.linspace(1, 0, fade_out)
    music[twist_sample+fade_out:] = 0

    drone = sine(55, DURATION - 23.93, amp=0.1)
    drone *= envelope(DURATION - 23.93, attack=1.0, decay=0, sustain=1, release=2.0)
    music[twist_sample:twist_sample+len(drone)] += drone[:len(music)-twist_sample]

    return music

def generate_sfx():
    sfx = np.zeros(TOTAL_SAMPLES)

    def add_sfx(wave, time_sec):
        if time_sec >= DURATION:
            return
        start = int(time_sec * SAMPLE_RATE)
        end = min(start + len(wave), TOTAL_SAMPLES)
        if start < TOTAL_SAMPLES:
            sfx[start:end] += wave[:end-start]

    def frame_to_time(f):
        return f / 30.0

    cues = {
        0: ('pop', 0.3),
        86: ('whoosh', 0.5),
        135: ('page_flip', 0.2),
        206: ('wind_wheat', 0.3),
        303: ('truck_engine', 0.4),
        303: ('metal_clank', 0.3),
        467: ('gun_chatter', 0.5),
        500: ('impact', 0.6),
        532: ('rattle', 0.4),
        550: ('ting', 0.3),
        615: ('page_turn', 0.2),
        718: ('sting', 0.8),
        1005: ('truck_fade', 0.3),
        1050: ('emu_content', 0.3),
    }

    for frame, (name, amp) in cues.items():
        t = frame_to_time(frame)

        if name == 'pop':
            add_sfx(sine(800, 0.05) * envelope(0.05, 0.001, 0.04, 0, 0) * amp, t)
        elif name == 'whoosh':
            w = noise(0.8, 'white', 0.3) * envelope(0.8, 0.1, 0.2, 0.3, 0.3)
            t_arr = np.arange(len(w)) / SAMPLE_RATE
            w *= np.exp(-t_arr * 2)
            add_sfx(w * amp, t)
        elif name == 'page_flip':
            add_sfx(noise(0.1, 'white', 0.4) * envelope(0.1, 0.001, 0.05, 0, 0.05) * amp, t)
        elif name == 'wind_wheat':
            w = noise(2.0, 'pink', 0.2) * envelope(2.0, 0.5, 0.5, 0.5, 0.5)
            add_sfx(w * amp, t)
        elif name == 'truck_engine':
            w = sine(60, 3.0, amp=0.3) * envelope(3.0, 0.2, 0.5, 0.5, 0.5)
            w += noise(3.0, 'pink', 0.1) * envelope(3.0, 0.2, 0.5, 0.5, 0.5)
            add_sfx(w * amp, t)
        elif name == 'metal_clank':
            add_sfx(sine(1200, 0.1) * envelope(0.1, 0.001, 0.05, 0, 0.05) * amp, t)
            add_sfx(sine(800, 0.15) * envelope(0.15, 0.001, 0.1, 0, 0.05) * amp * 0.5, t)
        elif name == 'gun_chatter':
            for i in range(20):
                shot_t = t + i * 0.15
                if shot_t > 26:
                    break
                add_sfx(noise(0.05, 'white', 0.8) * envelope(0.05, 0.001, 0.02, 0, 0.03) * 0.6, shot_t)
                add_sfx(sine(150, 0.1) * envelope(0.1, 0.001, 0.05, 0, 0.05) * 0.4, shot_t)
        elif name == 'impact':
            add_sfx(kick(0.3) * amp, t)
            add_sfx(noise(0.2, 'white', 0.4) * envelope(0.2, 0.001, 0.1, 0, 0.1), t)
        elif name == 'rattle':
            w = noise(1.0, 'white', 0.2) * envelope(1.0, 0.05, 0.2, 0.5, 0.3)
            w += sine(400, 1.0, amp=0.1) * np.exp(-np.arange(int(SAMPLE_RATE)) / (SAMPLE_RATE * 0.3))
            add_sfx(w * amp, t)
        elif name == 'ting':
            add_sfx(sine(2000, 0.3) * envelope(0.3, 0.001, 0.1, 0, 0.2) * amp, t)
        elif name == 'page_turn':
            add_sfx(noise(0.15, 'white', 0.3) * envelope(0.15, 0.001, 0.1, 0, 0.05) * amp, t)
        elif name == 'sting':
            add_sfx(kick(0.5) * 1.5, t)
            cluster = sine(200, 0.8, amp=0.4) + sine(210, 0.8, amp=0.3) + sine(300, 0.8, amp=0.2)
            cluster *= envelope(0.8, 0.01, 0.2, 0, 0.6)
            add_sfx(cluster * amp, t)
        elif name == 'truck_fade':
            w = sine(60, 2.0, amp=0.3) * envelope(2.0, 0, 0, 0, 2.0)
            add_sfx(w * amp, t)
        elif name == 'emu_content':
            for i in range(5):
                add_sfx(sine(400 + i * 50, 0.2) * envelope(0.2, 0.01, 0.1, 0, 0.1) * 0.2, t + i * 0.5)

    return sfx

def main():
    out_dir = '/sessions/friendly-festive-bardeen/shorts/emu-war'
    os.makedirs(out_dir, exist_ok=True)

    print("Generating music...")
    music = generate_music()
    write_wav(os.path.join(out_dir, 'music.wav'), music)
    print(f"Music RMS: {gated_rms(music):.6f}")

    print("Generating SFX...")
    sfx = generate_sfx()
    write_wav(os.path.join(out_dir, 'sfx.wav'), sfx)
    print(f"SFX RMS: {gated_rms(sfx):.6f}")

    print("Done.")

if __name__ == '__main__':
    main()
