import torch

print("avail=", torch.cuda.is_available())
try:
    print("devs=", torch.cuda.device_count())
except Exception as e:
    print("devs ERR:", type(e).__name__, e)
for i in range(max(0, int(torch.cuda.device_count()))):
    try:
        p = torch.cuda.get_device_properties(i)
        print(f"dev{i}: name={p.name} vram_gb={p.total_memory/2**30:.1f}")
    except Exception as e:
        print(f"dev{i} props ERR:", type(e).__name__, e)
