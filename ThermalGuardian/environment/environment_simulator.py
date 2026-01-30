import time

from environment.set_frequency import setFrequency
from environment.temperature_simulator import temperature_simulator
from environment.temperature_frequency_mapper import temperature_frequency_mapper
from DataStruct.globalConfig import GlobalConfig
class environment_simulator:
    """
    环境模拟器

    属性:
        environment_type: 模拟的环境类型
        T_sim: 温度模拟器
        T_F_mapper: 温度频率映射器
    """

    def __init__(self, environment_type):
        self.environment_type = environment_type
        # 极寒启动
        if GlobalConfig.simulated_environment == 'cold_up':
            self.T_sim = temperature_simulator(T_env=95,T_0=-40,k=0.1,cycle_duration=60)
        # 室温启动
        elif GlobalConfig.simulated_environment == 'room_up':
            self.T_sim = temperature_simulator(T_env=95,T_0=40,k=0.1,cycle_duration=60)
        # 极寒冷却
        elif GlobalConfig.simulated_environment == 'cold_down':
            self.T_sim = temperature_simulator(T_env=-40,T_0=95,k=0.1,cycle_duration=60)
        # 室温冷却
        elif GlobalConfig.simulated_environment == 'room_down':
            self.T_sim = temperature_simulator(T_env=40,T_0=95,k=0.1,cycle_duration=60)
        # 温度环境映射器（只与芯片属性有关，与环境无关）
        self.T_F_mapper = temperature_frequency_mapper(f_base=645, T_nominal=40, T_min=-40, T_max=100, gemma=0.05, alpha=0.15)
    def simulate(self):
        while(True):
            # 每一秒定期修改芯片频率。
            current_temperature = self.T_sim.get_temperature()
            # print(f"当前温度: {current_temperature} ℃")
            current_goal_frequency = self.T_F_mapper.convert_temperature_to_frequency(current_temperature)
            # print(f"预期设置频率: {current_goal_frequency} MHz")
            setFrequency(int(current_goal_frequency))
            time.sleep(1)



