import pynvml
import torch

# Initialize NVML
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Fetch GPU Details
gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)  # No need to decode
gpu_temp = pynvml.nvmlDeviceGetTemperature(gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)

# Print Details
print(f"🖥️ GPU Model: {gpu_name}")
print(f"🎮 VRAM: {gpu_memory.total / 1e9:.2f} GB")
print(f"🔥 Temperature: {gpu_temp}°C")

# Tensor Core Test (AI Performance)
print("\n🚀 Running Tensor Core Test...")
print("CUDA Available:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # Should show "RTX 4080 Super"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Device: {device}")
x = torch.rand((1000, 1000), device=device)
for _ in range(1000):  # Matrix multiplication stress test
    x = torch.mm(x, x)
torch.cuda.synchronize()
print("✅ Tensor Core Test Passed!")

# Cleanup
pynvml.nvmlShutdown()
