odev.py works as follows:
    For a given day it calculates the variances in the PTF of its month per hour, workdays and holidays are calculated individually
    Then for all 7 scenarios in price forecast data (PF - 1500 to PF + 1500):
        It then constructs an Integer Programming problem where the objective function is the maximize the total profit
        given it must stay down for 6 hours if it shuts down and startups have a cost defined by the power_plant

        It solves this using the that day and next days price forecast if it has it available else just uses that day

        It solves this RUN times more adding a random number from a normal distributuion centered at 0 with the calculated times. The results are printed

        Then the optimal schedules are tallied.

        The hour of maximum increase and decrease are returned and used to break the day into bid schedule

    In the previous version of this program, prices for that schedule were calculated using costs. However due to
    some changes in architecture, that functionality is currently broken. Most of the code that performed that funtionality
    can be seen in bidder.py; bidder.py is not called in the current build.
