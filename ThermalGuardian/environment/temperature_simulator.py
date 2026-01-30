import time
import math


class temperature_simulator:
    """
    牛顿冷却定律温度模拟器

    属性:
        T_env: 环境温度（摄氏度）
        T_0: 初始温度（摄氏度）
        k: 热交换系数（秒的负一次方）
        start_time: 模拟开始时间（时间戳）
    """

    def __init__(self, T_env, T_0, k=0.1, cycle_duration=60):
        """
        初始化温度模拟器

        参数:
            T_env: 环境温度（摄氏度）
            T_0: 初始温度（摄氏度）
            k: 热交换系数（默认0.1秒⁻¹）
        """
        self.T_env = T_env
        self.T_0 = T_0
        self.k = k
        self.cycle_duration = cycle_duration
        self.start_time = time.time()  # 记录模拟开始时间

    def get_temperature(self):
        """
        获取当前温度（基于牛顿冷却定律）

        返回:
            当前温度（摄氏度）
        """
        # 计算经过的时间（秒）
        elapsed_time = time.time() - self.start_time

        # 每cycle_duration为1个周期(默认为60秒)
        elapsed_time = elapsed_time % self.cycle_duration

        # 牛顿冷却定律公式：T(t) = T_0 + (T_env - T_0) * e^(-k*t)
        current_temp = self.T_0 + (self.T_env - self.T_0) * math.exp(-self.k * elapsed_time)

        return current_temp