# virus_sim

In the Endtimes (currently acknowledged as roughly February 2020 – ?) we need distractions.
Many dataviz people I follow (like [Harry Stevens](https://www.washingtonpost.com/graphics/2020/world/corona-simulator/) and [Kevin Simler](https://meltingasphalt.com/interactive/outbreak/) and [Grant Sanderson](https://twitter.com/3blue1brown/status/1241082514023649280))
have been distracting themselves by making pretty simulations of infection scenarios.  I had some questions that I wanted to answer so I
made my own Python implementation.

You can play around with the script, too.  Here are some examples:

```python
import virus_sim as vs
import matplotlib.pyplot as plt
import numpy as np

sim1 = vs.Epidemic(N = 1000,
                   inf_rad = 0.03,
                   pop_speed = 0.01,
                   inf_prob = 0.1,
                   recovery_rate = 14,
                   motion = "brownian")
```

Here, `sim1` has initialized an `Epidemic` simulation object.  The initialization parameters are:

- `N`: number of infectable bodies.  The bodies are initialized on a unit grid with periodic boundaries.
- `inf_rad`: "infection radius", how close do two bodies have to be to get infected
- `pop_speed`: How fast are your people?
- `inf_prob`: If two bodies are within `inf_rad`, how likely is it that they'll get the virus?
- `recovery_rate`: How many timesteps to recovery?
- `motion`: How do your people move?  The default, `brownian`, chooses a random direction
at every timestep, and moves that way.  The other option is `antisocial` which calculates a gradient on
the population density and chooses the direction which goes away from the most people.

Then, you can do
```python
history = sim1.time_evolve(200)
```
to get the numbers of who's infected, recovered, and healthy over time. `time_evolve` can also take the boolean parameter
`verbose`, which, when `True` will spit out the current stats to the screen.

Finally, there are two plotting functions, `Epidemic.plot_pop` which makes a scatter plot that looks like:
```python
sim1.plot_pop()
```
![popplot][pop_plot]


`Epidemic.plot_history` takes the output of the time evolution above and makes this plot:
```python
sim1.plot_history(history)
```
![histplot][hist_plot]


[pop_plot]: https://github.com/deichdeich/virus_sim/blob/master/plots/plot_pop.png
[hist_plot]: https://github.com/deichdeich/virus_sim/blob/master/plots/plot_history.png
