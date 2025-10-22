import numpy as np

class Instruments:
    @staticmethod
    def smooth_edge_weights(sim, threshold_weight):
        """
        对sim字典中的边权重进行平滑处理，确保所有权重在1 ± threshold_weight的范围内。
        使用权重均值作为1，离散值最大的权重设为1 ± threshold_weight的基准。

        参数：
        sim (dict): 一个包含边权重的字典，键为边对，值为权重。
        threshold_weight (float): 控制权重浮动范围的参数。

        返回：
        dict: 处理后的边权重字典。
        """
        weights = list(sim.values())
        print(np.isnan(weights))
        print(np.isinf(weights))
        mean_weight = np.mean(weights)
        max_deviation = max(weights, key=lambda x: abs(x - mean_weight))

        if max_deviation == mean_weight:
            # 如果所有权重都相同，不做处理
            return {edge: 1 for edge in sim}

        if max_deviation > mean_weight:
            max_val = 1 + threshold_weight
            min_val = 1 - threshold_weight * (mean_weight / max_deviation)
        else:
            max_val = 1 + threshold_weight * (max_deviation / mean_weight)
            min_val = 1 - threshold_weight

        smoothed_sim = {}
        for edge, weight in sim.items():
            if weight > mean_weight:
                new_weight = 1 + threshold_weight * ((weight - mean_weight) / (max_deviation - mean_weight))
            else:
                new_weight = 1 - threshold_weight * ((mean_weight - weight) / (mean_weight - max_deviation))
            smoothed_sim[edge] = new_weight

        return smoothed_sim