import pandas as pd

class PowerPlant():
    def __init__(self, data):
        self.name = data["NAME"][0]
        self.startup_cost = data["STARTUP_COST"][0]
        self.cooldown = int(data["COOLDOWN"][0])
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

def power_plant_factory(filename):
    data = pd.read_excel(filename, sheet_name = "Data").to_dict()
    return PowerPlant(data)
