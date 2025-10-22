import numpy as np

class Tools:
    @staticmethod
    def smooth_edge_weights(sim, threshold_weight):
        weights = list(sim.values())
        print(np.isnan(weights))
        print(np.isinf(weights))
        mean_weight = np.mean(weights)
        max_deviation = max(weights, key=lambda x: abs(x - mean_weight))

        if max_deviation == mean_weight:
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

