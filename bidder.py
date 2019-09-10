from powerPlant import *
import numpy as np

def give_prices(prices, breaks, main_block, power_plant, is_working):
    loads = [power_plant.optimal_load(price) for price in prices]
    costs = [power_plant.cost(load) for load in loads]
    cost_block = [sum(costs[b[0]:b[1]]) for b in breaks]
    cost_block[main_block] += power_plant.startup_cost
    cost_block[0] -= power_plant.startup_cost if is_working and main_block == 1 else 0
    return [(b[0], b[1], (cost_block[i] / sum(loads[b[0]:b[1]]))) for i, b in enumerate(breaks)]

def evaluate_bid(bids, prices, power_plant, is_working):
    profits = [power_plant.profit(price) for price in prices]
    net_profit = 0
    workings = []
    for index, bid in enumerate(bids):
        if is_working:
            if np.array(prices[bid[0]:bid[1]]).mean() >= bid[2]:
                net_profit += sum(profits[bid[0]:bid[1]])
            else:
                is_working = False
        else:
            start = int(bid[0] if index == 0 else max(bids[index -  1][0] + power_plant.cooldown, bid[0]))
            if prices[start:bid[1]] and np.array(prices[start:bid[1]]).mean() >= bid[2]:
                net_profit += (sum(profits[int(start):bid[1]]) - power_plant.startup_cost)
                is_working = True
        workings.append(is_working)
    return net_profit, workings
