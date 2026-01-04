import psutil, os

def log_ram(stage=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"[RAM] {stage}: {mem:.2f} GB")
