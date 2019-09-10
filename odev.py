import math
import csv
import sys
import warnings
import locale
from collections import defaultdict
from datetime import datetime as dt
from datetime import timedelta as td
from itertools import groupby
from calendar import monthrange

import numpy as np
import cvxpy as cp
import intervaltree
from workalendar.europe import Turkey

from powerPlant import *

warnings.filterwarnings("ignore") #This is to suprress the calendar warning about Islamic Holidays

RUNS = 25

# This function used to get prices as exported by seffaflik
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

def get_predictions(filename):
    tl_predictions = defaultdict(lambda: [0] * 24)
    data = pd.read_excel(filename, sheet_name = "Data").to_numpy()
    for row in data:
        tl_predictions[row[0]][row[1]] = list(row[2:9])
    return tl_predictions


#MIP is setup in this function
def solver(profits, is_working, power_plant):
    hours = len(profits)
    startup_costs = [power_plant.startup_cost] * hours # This used as an array in case startup is variable instead of constant in future
    workings = cp.Variable(hours, boolean = True)
    startups = cp.Variable(hours, boolean = True)
    shutdowns = cp.Variable(hours, boolean = True)
    constraints = []
    for hour in range(hours):
        if hour == 0:
            constraints.append(workings[hour] - is_working == startups[hour] - shutdowns[hour]) #if we did not work before but work now we have done a startup if we don't work now but worked before then we had a shutdown
        else:
            constraints.append(workings[hour] - workings[hour - 1] == startups[hour] - shutdowns[hour])
        if hour >= power_plant.cooldown:
            constraints.append(cp.sum(shutdowns[(hour - power_plant.cooldown) + 1 : hour + 1]) <= (1 - workings[hour])) #can't work if we had a shutdown in the last COOLDOWN hours
    objective = cp.Maximize(cp.sum(profits @ workings) - cp.sum(startup_costs @ startups))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return workings.value.round().astype(bool) #Internally cvxpy keeps all values as floats so we round and cast into bool

def optimal_schedule(profits, is_working, power_plant):
    workings = solver(profits, is_working, power_plant)[0:24] #limit answer to 1 day
    schedule = []
    for value, group in groupby(range(len(workings)), key = lambda x: workings[x]):
        group = list(group)
        schedule.append((group[0], group[-1] + 1, value)) #The format for bids is [x, y) thus we need to add + 1 to second item.
    return schedule

def sort_intervals(intervals):
    tree = intervaltree.IntervalTree()
    interval_counts = defaultdict(lambda: 0)
    for i in intervals:
        interval_counts[i] = interval_counts[i] + 1
    for key, value in interval_counts.items():
            tree.add(intervaltree.Interval(key[0], key[1] + 1, value))
    answer = []
    for i in range(24):
        answer.append((sum(value[2] for value in tree.at(i))))
    return answer

def give_prices(prices, breaks, main_block, power_plant, is_working):
    loads = [power_plant.optimal_load(price) for price in prices]
    costs = [power_plant.cost(load) for load in loads]
    cost_block = [sum(costs[b[0]:b[1]]) for b in breaks]
    cost_block[main_block] += power_plant.startup_cost
    cost_block[0] -= power_plant.startup_cost if is_working and main_block == 1 else 0
    return [(b[0], b[1], (cost_block[i] / sum(loads[b[0]:b[1]]))) for i, b in enumerate(breaks)]


def calculate_variances(prices, date):
    workday_variance_lists = [[] for i in range(24)]
    holiday_variance_lists = [[] for i in range(24)]
    first_day_of_month = dt(date.year, date.month, 1)
    for day in (first_day_of_month + td(days = n) for n in range(monthrange(date.year, date.month)[1])):
        if day in prices:
            for hour in range(24):
                if cal.is_working_day(day):
                    workday_variance_lists[hour].append(prices[day][hour])
                else:
                    holiday_variance_lists[hour].append(prices[day][hour])
    workday_variances = [np.array(variances).var() for variances in workday_variance_lists]
    holiday_variances = [np.array(variances).var() for variances in holiday_variance_lists]
    return workday_variances, holiday_variances

def report(day, tl_predictions, workday_variances, holiday_variances, is_working, power_plant, alpha):
    print("Fiyat Tahmini Datasina Gore En iyi Calisma Sekilleri Uretiliyor")
    today = day
    tomorrow = today + td(days = 1)
    if today not in tl_predictions:
            print("Fiyat tahmin verisi yok")
            sys.exit()
    prices = tl_predictions[day]
    variances = workday_variances if cal.is_working_day(day) else holiday_variances
    if tomorrow not in tl_predictions:
        print("Yarinin fiyat tahmini bulunamadi. Santralin saat 24 te kapanmasi gerektigi varsayiliyor")
    else:
        prices += tl_predictions[tomorrow]
        variances += workday_variances if cal.is_working_day(day) else holiday_variances
    return generate_intervals(prices, variances, alpha)

def generate_intervals(prices, variances, alpha):
    intervals = []
    for level in range(7):
        print(f"PF + {(level - 3) * 500}")
        print(f"\tSchedule According to Actual Data")
        profits = [BND2.profit(hour[level]) for hour in prices]
        schedule = optimal_schedule(profits, is_working, BND2)
        print(f"\t {schedule}")
        for interval in schedule:
            intervals.append(interval)
        print(f"\tSchedule According to Data + Noise")
        for possiblity in range(RUNS):
            profits = []
            for i, hour in enumerate(tl_predictions[day]):
                noise = np.random.normal(loc = 0, scale = alpha * math.sqrt(variances[i]))
                profits.append(BND2.profit(hour[level] + noise))
            schedule = optimal_schedule(profits, is_working, BND2)
            print(f"\t {schedule}")
            for interval in schedule:
                intervals.append(interval)
    return intervals

def print_schedule(schedule):
    schedule_for_printing = [(interval[0], interval[1]) for interval in schedule]
    print("I think the optimum bid structure is the following:")
    print(schedule_for_printing)
    for index, item in enumerate(schedule):
        if item[1] - item[0] < 3:
            print(f"{schedule_for_printing[index]} is given as hourly")

def create_bid_schedule_for_3(max_index, min_index, is_working):
    schedule = ([(0, min_index, True), (min_index, max_index, False), (max_index, 24, True)] if is_working else
                [(0, max_index, False), (max_index, min_index, True), (min_index, 24, False)])
    print_schedule(schedule)
    return schedule


def create_bid_schedule_for_2(max_index, min_index, is_working):
    index = max_index if max_index - 1 != 0 else min_index
    schedule = [(0, index, True), (index, 24, False)] if min_index != 0 else [(0, index, False), (index, 24, True)]
    print_schedule(schedule)
    return schedule


def get_break_points(intervals):
    filtered_intervals = [interval for interval in intervals if interval[2]]
    interval_counts = (sort_intervals(filtered_intervals))
    interval_counts_diff = list(np.diff(interval_counts))
    max_index = interval_counts_diff.index(max(interval_counts_diff)) + 1
    min_index = interval_counts_diff.index(min(interval_counts_diff))
    return max_index, min_index

if __name__ == "__main__":
    cal = Turkey()
    BND2 = power_plant_factory("power_plant_data.xlsx")
    dolar_prices, tl_prices = get_prices('prices.csv')
    tl_predictions = get_predictions("KapasiteSensitivity.xlsx")
    day = dt.strptime(sys.argv[1], "%d/%m/%Y") if len(sys.argv) > 2 else dt.today().replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    is_working = sys.argv[2] if len(sys.argv) > 2 else 0
    workday_variances, holiday_variances = calculate_variances(tl_prices, day)
    alpha = 1

    intervals = report(day, tl_predictions, workday_variances, holiday_variances, is_working, BND2, alpha)
    max_index, min_index = get_break_points(intervals)
    #We can use a gradient search here to maybe decide what kind of noise produce most stable results
    while max_index - 1 == 0 and min_index == 0:
        alpha *= 1.5
        print(f"No break points with normal variance increased variance to {alpha} times its real value")
        intervals = report(day, tl_predictions, workday_variances, holiday_variances, is_working, BND2, alpha)
        max_index, min_index = get_break_points(intervals)

    if max_index - 1 == 0 and min_index == 0:
        create_bid_schedule_for_2(max_index, min_index, is_working)
    else:
        create_bid_schedule_for_3(max_index, min_index, is_working)
