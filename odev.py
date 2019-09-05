import pandas as pd
import numpy as np
import locale
from datetime import datetime as dt
from datetime import timedelta as td
from collections import defaultdict
from workalendar.europe import Turkey
import csv
import sys
import intervaltree
import warnings

class PowerPlant():
    def __init__(self, data):
        self.name = data["NAME"][0]
        self.startup_cost = data["STARTUP_COST"][0]
        self.cooldown = data["COOLDOWN"][0]
        self.marginal_cost = data["EXTRA"][0]
        self.base_cost = data["BASE"][0]
        self.max_load = data["MAX_LOAD"][0]
        self.min_load = data["MIN_LOAD"][0]
    def cost(self, load):
        if not (self.min_load <= load <= self.max_load):
            sys.exit(f"{self.name} has a load of {load} which is out of bounds")
        return self.min_load * self.base_cost + (load - self.min_load) * self.marginal_cost
    def revenue(self, load, price):
        return load * price
    def optimal_load(self,price):
        load = self.max_load if price > self.marginal_cost else self.min_load
        return load
    def profit(self, price):
        load = self.optimal_load(price)
        return self.revenue(price, load) - self.cost(load)

def PowerPlant_factory(filename):
    data = pd.read_excel(filename, sheet_name = "Data").to_dict()
    return PowerPlant(data)

def get_prices(filename):
    locale.setlocale(locale.LC_NUMERIC,"tr")
    dolar_prices = defaultdict(lambda: [0] * 24)
    tl_prices = defaultdict(lambda: [0] * 24)
    with open(filename) as PTFdata:
        for row in csv.reader(PTFdata):
            date = dt.strptime(row[0], "%d/%m/%Y")
            hour = int(row[1][:2])
            dolar_price = locale.atof(row[3])
            tl_price = locale.atof(row[2])
            tl_prices[date][hour] = tl_price
            dolar_prices[date][hour] = dolar_price
    return dolar_prices, tl_prices

def get_bids(schedule):
    answer = []
    current_index = 0
    current = schedule[0]
    for index, value in enumerate(schedule):
        if value == current:
            pass
        else:
            answer.append((current_index, index, current))
            current_index = index
            current = value
    answer.append((current_index, len(schedule), current))
    return answer

def create_bid(working, bids):
    answer = []
    for index, bid in enumerate(bids):
        if working and index < len(bids) - 1 and bids[index + 1][2] == True:
            answer.append((bid[0], bid[1], -1))
        elif bid[2] == True:
            answer.append((bid[0], bid[1], 1))
        else:
            answer.append((bid[0], bid[1], 0))
    return answer

def get_bid_values(prices, schedule, power_plant):
    loads = [power_plant.optimal_load(price) for price in prices]
    costs = [power_plant.cost(load) for load in loads]
    ans = []
    for bid in schedule:
        ans.append(((sum(costs[bid[0]:bid[1]])) + (bid[2] * power_plant.startup_cost)) / sum(loads[bid[0]:bid[1]]))
    return ans

def optimal_schedule(profits, working, power_plant):
    answer = []
    i = 0
    while i < len(profits):
        # print(f"i: {i}")
        if working:
            if profits[i] > 0:
                answer.append(True)
                working = True
            else:
                potential = 0
                j = i
                while j < len(profits):
                    potential += profits[j]
                    # print(f"potential: {potential}")
                    if j - i < power_plant.cooldown:
                        if potential > 0:
                            answer += [True] * (j - i)
                            # working = True
                            i = j - 1
                            # print(f"K")
                            break
                        elif j == (len(profits) - 1):
                            answer += [False] * (j- i)
                            working = False
                            i = j -1
                            # print(f"J")
                            break
                    else:
                        # print(f"j: {j}")
                        # print(f"ii: {i}")
                        if potential < (-1 * power_plant.startup_cost):
                            answer += [False] * (j - i)
                            working = False
                            i = j - 1
                            # print(f"H")
                            break
                        if potential > 0:
                            answer += [True] * (j - i)
                            working = True
                            i = j - 1
                            # print(f"G")
                            break
                        if j == (len(profits) - 1) and potential <= 0: #TODO: it does not have to end at midnight
                            answer += [False] * (j - i)
                            working = False
                            # print(f"F")
                            i = j - 1
                    j += 1
        else:
            if profits[i] < 0:
                answer.append(False)
                working = False
                # print(f"D")
            else:
                potential_profit = 0
                j = i
                while j < len(profits):
                    potential_profit += profits[j]
                    if potential_profit > power_plant.startup_cost:
                        answer += [True] * (j - i)
                        working = True
                        i = j - 1
                        # print(f"C")
                        break
                    elif potential_profit < 0:
                        answer += [False] * (j - i)
                        working = False
                        i = j -1
                        # print(f"B")
                        break
                    if j == len(profits) - 1 and potential_profit <= power_plant.startup_cost:
                        answer += [False] * (j - i)
                        working = False
                        i = j - 1
                        # print(f"A")
                    j += 1
        i += 1
    return answer[0:24]

def sort_intervals(intervals):
    tree = intervaltree.IntervalTree()
    interval_counts = defaultdict(lambda: 0)
    for i in intervals:
        interval_counts[i] = interval_counts[i] + 1
    isss = [(key[0],key[1],value) for key,value in interval_counts.items()]
    isss.sort(key = lambda x: x[2], reversed = True)
    print(isss)
    for key, value in interval_counts.items():
            tree.add(intervaltree.Interval(key[0], key[1] + 1, value))
    answer = []
    for i in range(24):
        answer.append((sum(value[2] for value in tree.at(i))))
    return answer

def fun(prices, breaks, main_block, power_plant, is_working):
    loads = [power_plant.optimal_load(price) for price in prices]
    costs = [power_plant.cost(load) for load in loads]
    cost_block = [sum(costs[b[0]:b[1]]) for b in breaks]
    cost_block[main_block] += power_plant.startup_cost
    cost_block[0] -= power_plant.startup_cost if is_working and main_block == 1 else 0
    return [(b[0], b[1], (cost_block[i] / sum(loads[b[0]:b[1]]))) for i, b in enumerate(breaks)]

def evaluate_bid(bids, prices, power_plant, is_working):
    profits = [power_plant.profit(price) for price in prices]
    net_profit = 0
    for index, bid in enumerate(bids):
        if is_working:
            if np.array(prices[bid[0]:bid[1]]).mean() >= bid[2]:
                net_profit += sum(profits[bid[0]:bid[1]])
                is_working = True
            else:
                is_working = False
        else:
            start = bid[0] if index == 0 else max(bids[index -1][0] + power_plant.cooldown, bid[0])
            start = int(start)
            if len(prices[start:bid[1]]) > 0 and np.array(prices[start:bid[1]]).mean() >= bid[2]:
                net_profit += (sum(profits[start:bid[1]]) - power_plant.startup_cost)
                is_working = True
            else:
                is_working = False
    return net_profit

def drop_startups(max_profit, is_working, schedule):
    for x, y, state in schedule:
        if is_working and not state:
            is_working = False
        if state and not is_working:
            is_working = True
            max_profit -= BND2.startup_cost
    return max_profit

cal = Turkey()
BND2 = PowerPlant_factory("cem.xlsx")
dolar_prices, tl_prices = get_prices('PTF-01082009-01082019.csv')

intervals = []
monthly_bids = {}
c = []
max_profits = {}
a = defaultdict(lambda:0)
is_working = False
monthly_results = []
for mon in range(1,4):
    for day, prices in tl_prices.items():
        if cal.is_working_day(day) and day.month == mon and day.year == 2019:
            profits = [BND2.profit(price) for price in tl_prices[day]]
            schedule = get_bids(optimal_schedule(profits, is_working, BND2))
            # print(is_working)
            max_profit = sum([sum(profits[bid[0]:bid[1]]) for bid in schedule if bid[2]])
            max_profit = drop_startups(max_profit, is_working, schedule)
            max_profits[day] = max_profit
            b = []
            for i in range(3,22):
                for j in range(i+3, 24):
                    blocks = [sum(profits[0:i]), sum(profits[i:j]), sum(profits[j:24])]
                    main_block = blocks.index(max(blocks))
                    n = evaluate_bid(fun(tl_prices[day], [(0,i), (i,j), (j,24)], main_block, BND2, is_working), tl_prices[day], BND2, is_working)
                    a[(i,j)] += n
                    b.append((i, j, n))
    ac = [(key[0],key[1],value) for key,value in a.items()]
    ac.sort(key = lambda x:x[2])
    top_break_points = ac[-1]
    results = []
    for day, prices in tl_prices.items():
        if cal.is_working_day(day) and day.month == mon and day.year == 2019:
            profits = [BND2.profit(price) for price in tl_prices[day]]
            schedule = get_bids(optimal_schedule(profits, is_working, BND2))
            max_profit = sum([sum(profits[bid[0]:bid[1]]) for bid in schedule if bid[2]])
            max_profit = drop_startups(max_profit, is_working, schedule)
            max_profits[day] = max_profit
            n = (evaluate_bid(fun(tl_prices[day], [(0,top_break_points[0]), (top_break_points[0],top_break_points[1]), (top_break_points[1],24)], main_block, BND2, is_working), tl_prices[day], BND2, is_working))
            if max_profit == 0:
                results.append(1)
            else:
                results.append(n / max_profit)
    monthly_results.append((np.array(results).mean()))

print(np.array(monthly_results).mean())


# iss = [(i,j) for i,j,n in monthly_bids.values()]
# sort_intervals(iss)
# print(np.array(c).mean())



# is_working = False
# profits = [BND2.profit(price) for price in tl_prices[dt(2019,1,29)]]
# schedule = get_bids(optimal_schedule(profits, is_working, BND2))
# max_profit = sum([sum(profits[bid[0]:bid[1]]) for bid in schedule if bid[2]])
# max_profit = drop_startups(max_profit, is_working, schedule)
# actual_bid = fun(tl_prices[dt(2019,1,29)], [(0,16), (16,21), (21,24)], 1, BND2, is_working)
# n = evaluate_bid(actual_bid,  tl_prices[dt(2019,1,29)], BND2, is_working)

# print(profits)
# print(schedule)
# print(max_profit)
# print(actual_bid)
# print(n)






















        # intervals += [(bid[0], bid[1]) for bid in schedule if bid[2]]

# print(sort_intervals(intervals))
