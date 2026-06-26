import subprocess


def get_free_gpu():
    """Find the GPU with the most free memory by querying nvidia-smi.

    Returns:
        int | None: Index of the GPU with the largest amount of free memory,
            or None if nvidia-smi could not be accessed.
    """
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        mem_free = [int(x) for x in result.strip().split('\n')]
        gpu_index = mem_free.index(max(mem_free))
        return gpu_index
    except Exception as e:
        print("It was not possible to access nvidia-smi:", e)
        return None
