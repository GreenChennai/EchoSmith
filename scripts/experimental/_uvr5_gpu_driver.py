# -*- coding: utf-8 -*-
# UVR5 驱动变体:device 可选(ZLUDA GPU 试验)。用法: python _uvr5_gpu_driver.py <in> <vocal_dir> <ins_dir> <device>
import os, sys, time
root = os.path.dirname(os.path.abspath(__file__))
eng = r"E:\平日资料\GitHub\EchoSmith\models\GPT-SoVITS"
sys.path.insert(0, os.path.join(eng, "tools", "uvr5"))
inp, vocal_dir, ins_dir, device = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
W = os.path.join(eng, "tools", "uvr5", "uvr5_weights")
t0 = time.time()
import vr
m = vr.AudioPre(agg=10, model_path=os.path.join(W, "HP5_only_main_vocal.pth"),
                device=device, is_half=False)
m._path_audio_(inp, ins_dir, vocal_dir, "wav", is_hp3=False)
print(f"UVR5_GPU_OK {time.time()-t0:.0f}s")
