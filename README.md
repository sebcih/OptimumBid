# OptimumBid
OptimumBid is a bid optimization program that tries to offer bids in day-ahead market in Turkey  such that the profit
from a thermic power plant is maximized
# Setup
In order to use this software you must have python 3.7, numpy, cvxpy, intervaltree, workalendar  installed
# Usage
Use the generated data or supply your own data and run
```bash
$ cd <folder-with-odev.py>
$ python odev.py dd/mm/yyyy working_status
```
In the example dd/mm/yyyy is the date desired and working status is 1 if the power plant was working the day before else 0
