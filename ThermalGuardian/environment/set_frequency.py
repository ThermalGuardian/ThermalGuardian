import pynvml
def setFrequency(frequency):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # 获取当前频率
    current_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
    # print(f"设置前GPU {0} 当前频率: {current_clock} MHz")

    # 获取显存频率支持范围
    mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
    core_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, mem_clocks[0])
    # print(f"memory 支持频率: {mem_clocks} MHz")
    # print(f"GPU {0} 支持频率: {core_clocks} MHz")

    # 修改GPU频率
    min_clock = frequency
    max_clock = frequency
    pynvml.nvmlDeviceSetGpuLockedClocks(handle, min_clock, max_clock)

    # 获取当前频率
    current_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
    # print(f"设置后GPU {0} 当前频率: {current_clock} MHz")
