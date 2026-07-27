#!/usr/bin/env python3
"""
Final Benchmark: ParlerTTS vs KokoroTTS (CPU + MPS)
Sequential tests: Parler(MPS) → Kokoro(CPU) → Kokoro(MPS)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

CHAPTER = Path("/tmp/pp_chapter1.txt")
OUT = Path("/tmp/tts_benchmark")

@dataclass
class R:
    engine: str
    device: str
    test: str
    text_chars: int
    text_words: int
    load_t: float
    synth_t: float
    total_t: float
    audio_s: float
    audio_b: int
    rtf: float
    errors: list = field(default_factory=list)

    def d(self):
        return {
            "engine": self.engine,
            "device": self.device,
            "test": self.test,
            "text_chars": self.text_chars,
            "text_words": self.text_words,
            "load_t": round(self.load_t, 3),
            "synth_t": round(self.synth_t, 3),
            "total_t": round(self.total_t, 3),
            "audio_s": round(self.audio_s, 3),
            "audio_b": self.audio_b,
            "rtf": round(self.rtf, 3),
            "errors": self.errors,
        }


def parler(text, device, test_name):
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    import soundfile as sf
    errors = []
    (OUT/"parler").mkdir(parents=True,exist_ok=True)
    op = OUT/"parler"/f"parler_{test_name}.wav"
    t0=time.perf_counter()
    try:
        dt=torch.float16 if device in ("mps","cuda") else torch.float32
        tl=time.perf_counter()
        m=ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1",torch_dtype=dt).to(device)
        tk=AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
        lt=time.perf_counter()-tl
        desc="A middle-aged male speaker with a warm, clear, and measured voice delivers the narration at a comfortable pace in a quiet studio environment. The recording is clean with no background noise."
        ts=time.perf_counter()
        di=tk(desc,return_tensors="pt").input_ids.to(device)
        ti=tk(text,return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            g=m.generate(input_ids=di,prompt_input_ids=ti,do_sample=False)
        a=g.cpu().numpy().squeeze()
        if a.dtype.name == "float16":
            a = a.astype("float32")
        sf.write(str(op),a,m.config.sampling_rate)
        st=time.perf_counter()-ts
        tt=time.perf_counter()-t0
        ad=len(a)/m.config.sampling_rate
        ab=op.stat().st_size
        print(f"  Parler/{test_name} load={lt:.1f}s synth={st:.1f}s audio={ad:.1f}s rtf={st/ad:.2f}x")
    except Exception as e:
        errors.append(str(e))
        return R("parler-tts-mini-v1",device,test_name,len(text),len(text.split()),0,0,0,0,0,0,errors)
    return R("parler-tts-mini-v1",device,test_name,len(text),len(text.split()),lt,st,tt,ad,ab,st/ad if ad>0 else 0,errors)


def kokoro(text, device, test_name):
    from kokoro import KPipeline
    import soundfile as sf
    errors = []
    (OUT/"kokoro").mkdir(parents=True,exist_ok=True)
    op = OUT/"kokoro"/f"kokoro_{test_name}.wav"
    t0=time.perf_counter()
    try:
        tl=time.perf_counter()
        p=KPipeline(lang_code='a',device='cpu')
        if device=='mps':
            p.model.to('mps')
            p.model.eval()
        lt=time.perf_counter()-tl
        ts=time.perf_counter()
        gen=p(text,voice='af_heart',speed=1.0,split_pattern=None)
        segs=[]
        for r in gen:
            s=r.audio
            if torch.is_tensor(s):
                s=s.cpu().numpy().squeeze()
            if s.ndim>=1 and len(s)>0:
                segs.append(s)
        audio=np.concatenate(segs) if segs else np.array([])
        sf.write(str(op),audio,24000)
        st=time.perf_counter()-ts
        tt=time.perf_counter()-t0
        ad=len(audio)/24000
        ab=op.stat().st_size
        print(f"  Kokoro/{test_name}[{device}] load={lt:.1f}s synth={st:.1f}s audio={ad:.1f}s rtf={st/ad:.2f}x")
    except Exception as e:
        import traceback

        traceback.print_exc()
        errors.append(str(e))
        return R("kokoro-82m",device,test_name,len(text),len(text.split()),0,0,0,0,0,0,errors)
    return R("kokoro-82m",device,test_name,len(text),len(text.split()),lt,st,tt,ad,ab,st/ad if ad>0 else 0,errors)


def main():
    print("="*80)
    print("🎙  ParlerTTS vs KokoroTTS — Final Benchmark (Sequential)")
    print("="*80)

    full=CHAPTER.read_text().strip()
    words=full.split()
    short=" ".join(words[:200])
    dev=get_device()
    print(f"\n📖 Pride & Prejudice Ch1: {len(full)} chars, {len(words)} words | Short: {len(short.split())} words")
    print(f"🖥  Device: {dev}")

    results=[]

    # Test order: Parler first (model already loaded from previous imports? nope, fresh)
    print("\n── Short text (200 words) ──")
    results.append(parler(short,dev,"short"))
    results.append(kokoro(short,"cpu","short"))
    results.append(kokoro(short,"mps","short"))

    print("\n── Full Chapter 1 (861 words) ──")
    results.append(parler(full,dev,"full"))
    results.append(kokoro(full,"cpu","full"))
    results.append(kokoro(full,"mps","full"))

    # Table
    print("\n"+"="*90)
    print(f"{'Engine':<22} {'Test':<8} {'Dev':<6} {'Load':>7} {'Synth':>8} {'Total':>8} {'Audio':>8} {'RTF':>7} {'WPS':>7}")
    print("-"*90)
    for r in results:
        wps = r.text_words/(r.audio_s+0.001)
        print(f"{r.engine:<22} {r.test:<8} {r.device:<6} {r.load_t:>6.1f}s {r.synth_t:>7.1f}s {r.total_t:>7.1f}s {r.audio_s:>7.1f}s {r.rtf:>6.2f}x {wps:>6.1f}")

    # Comparison
    tests = ["short","full"]
    for tname in tests:
        pr = [r for r in results if r.engine.startswith("parler") and r.test==tname]
        kc = [r for r in results if r.engine.startswith("kokoro") and r.device=="cpu" and r.test==tname]
        km = [r for r in results if r.engine.startswith("kokoro") and r.device=="mps" and r.test==tname]
        if pr and kc:
            sp = pr[0].total_t/kc[0].total_t if kc[0].total_t>0 else 0
            print(f"\n  [{tname}] Kokoro-CPU vs Parler: {'Kokoro' if sp>1 else 'Parler'} is {max(sp,1/sp):.1f}x faster ({'Kokoro' if sp>1 else 'Parler'} wins)")
        if kc and km:
            sp2 = kc[0].total_t/km[0].total_t if km[0].total_t>0 else 0
            print(f"  [{tname}] Kokoro-MPS vs Kokoro-CPU: {'MPS' if sp2>1 else 'CPU'} is {max(sp2,1/sp2):.1f}x faster")

    # Save
    jp = OUT/"benchmark_final.json"
    jp.parent.mkdir(parents=True,exist_ok=True)
    jp.write_text(json.dumps([r.d() for r in results],indent=2))
    print(f"\n📄 {jp}")
    print("✅ Done!")

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

if __name__=="__main__":
    main()
