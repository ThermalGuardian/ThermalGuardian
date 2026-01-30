class temperature_frequency_mapper:
    def __init__(self, f_base, T_nominal=40, T_min=-40, T_max=100, gemma=0.05, alpha=0.15):
        """
        初始化温度频率映射器

        参数:
            f_base: 芯片正常工作基础频率
            T_nominal: 芯片标称工作温度（摄氏度）
            T_min: 芯片最低工作温度（摄氏度）
            T_max: 芯片最高工作温度（摄氏度）
            gemma: 低温补偿系数
            alpha: 高温降频系数
        """
        self.f_base = f_base
        self.T_nominal = T_nominal
        self.T_min = T_min
        self.T_max = T_max
        self.gemma = gemma
        self.alpha = alpha
    def convert_temperature_to_frequency(self,T):
        # 低温补偿
        if T < self.T_nominal:
            temp1 = (self.T_nominal - T)/(self.T_nominal - self.T_min)
            return self.f_base * (1 + (self.gemma * temp1))

        # 高温降频
        elif T >= self.T_nominal:
            temp1 = (T - self.T_nominal)/(self.T_max - self.T_nominal)
            return self.f_base * (1 - (self.alpha * temp1))
