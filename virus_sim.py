import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


class Epidemic(object):
    def __init__(self,
                 N,  # population
                 inf_rad,  # infection radius
                 pop_speed,  # speed of population
                 inf_prob, # likelihood of infecting if inside infection radius
                 recovery_rate, # number of timesteps to recover
                 init_inf = 1, # number of initial infections
                 motion = "brownian" # how the dots move.  Options are "brownian" (random direction at every dt),
                                     #                                 "antisocial" (each dot avoids other dots)
                ):
        
        self.N = N
        self.inf_rad = inf_rad
        self.pop_speed = pop_speed
        self.inf_prob = inf_prob
        self.motion = motion
        self.recovery_rate = recovery_rate

        self.population = self.init_pop()
        self.init_test()


    def __str__(self):
        return(f"""Total population: {len(self.population)}\n
                  \tHealthy: {len(np.where(self.population[:, 2] == 0)[0])}\n
                  \tInfected: {len(np.where(self.population[:, 2] > 0)[0])}\n
                  \tRecovered: {len(np.where(self.population[:, 2] == -1)[0])}""")
    """
    initialize a population array.
    Shape: 3xN. Cols 0 and 1 are x,y coords. Value of col 2 is the number
    of days that the individual has been infected. Value is -1 if recovered.
    """
    def init_pop(self):
        pop = np.zeros((self.N, 3))
        pop[:-1,:2] = np.random.random((self.N-1, 2))
        pop[-1,:2] = np.array([0.5,0.5])
        return(pop)

    
    """
    Initialize infection.
    Randomly chooses a number of the population to be infected
    """
    def init_infection(self, n_inf):
        self.population[np.random.choice(range(self.N), n_inf), 2] = 1

    def init_test(self):
        self.population[-1, 2] = 1

    def populate_history(self, history, step):
        history[step, 0] = len(np.where(self.population[:, 2] == 0)[0])
        history[step, 1] = len(np.where(self.population[:, 2] > 0)[0])
        history[step, 2] = len(np.where(self.population[:, 2] == -1)[0])

    def move_pops(self, strategy):
        if strategy == "brownian":
            self.move_brownian()
        
        elif strategy == "antisocial":
            self.move_antisocial()
        
        # impose periodic boundaries on a unit square
        self.population[:, :2] %= 1


    def move_brownian(self):
        # get random directions to update the new position vector
        new_dirs = np.random.random((self.N, 2)) * self.pop_speed

        # np.random.random only does numbers on [0, 1), so we
        # need to randomize the directions further
        new_dirs[:, 0] *= np.sign(np.random.random(self.N) - 0.5)
        new_dirs[:, 1] *= np.sign(np.random.random(self.N) - 0.5)
        
        # update positions
        self.population[:, :2] += new_dirs



    """
    move the dots away from other dots
    prescription:
        1. get density map of dots with np.hist2d
        2. calculate gradient of that density map
        3. move in the opposite direction of the gradient
    complication:
        np.gradient doesn't do periodic boundary conditions
    solution:
        make a temporary density map which is 2 bigger in every dimension
        which copies the values from the corresponding row/column, and
        calculate the gradient on that
    """
    def move_antisocial(self, nbins = 40):
        healthy, infected, recovered = self.get_indices()
        density_map, xedges, yedges = np.histogram2d(self.population[infected][:, 0],
                                           self.population[infected][:, 1],
                                           bins = nbins)

        size_x, size_y = density_map.shape
        big_density_map = np.zeros((size_x + 2, size_y + 2))
        
        # fill interior with old density map
        big_density_map[1:-1, 1:-1] = density_map
        
        # wrap the edges around
        big_density_map[1:-1, 0] = density_map[:, -1]
        big_density_map[1:-1, -1] = density_map[:, 0]
        big_density_map[0, 1:-1] = density_map[-1, :]
        big_density_map[-1, 1:-1] = density_map[0, :]
        big_density_map[0, 0] = density_map[0, 0]
        big_density_map[0, 1] = density_map[0, 1]
        big_density_map[1, 0] = density_map[1, 0]
        big_density_map[1, 1] = density_map[1, 1]


        grad_y, grad_x = np.gradient(big_density_map)
        _, _, _, binnumbers = stats.binned_statistic_2d(self.population[:, 0],
                                                        self.population[:, 1],
                                                        self.population[:, 1],
                                                        expand_binnumbers = True,
                                                        bins=nbins - 1)
        binnumbers = binnumbers.T
        dx = -grad_x[(binnumbers[:,0], binnumbers[:,1])] * self.pop_speed
        dy = -grad_y[(binnumbers[:,0], binnumbers[:,1])] * self.pop_speed
        self.population[:, 0] += dx
        self.population[:, 1] += dy
    
    def get_indices(self):
        healthy_inds = np.where(self.population[:, 2] == 0)[0]
        infected_inds = np.where(self.population[:, 2] > 0)[0]
        recovered_inds = np.where(self.population[:, 2] == -1)[0]
        return(healthy_inds, infected_inds, recovered_inds)

    def get_distance(self, p1, p2):
        p1x, p1y = p1
        p2x, p2y = p2
        dist1 = np.sqrt((p1x - p2x)**2 + (p1y-p2y)**2)
        #dist2 = np.sqrt((p1x - p2x-2)**2 + (p1y-p2y-2)**2)%1
        return(dist1)

    def infect_or_recover(self):

        # first, figure out who has recovered
        self.population[np.where(self.population[:, 2] > self.recovery_rate), 2] = -1

        healthy_inds, infected_inds, recovered_inds = self.get_indices()
        
        # increment the infected day count
        self.population[infected_inds, 2] += 1

        # then, for only those who aren't infected, run an n-body thing to figure out
        # who's in the infection radius
        for ni in healthy_inds:
            ni_pop = self.population[ni]
            pos1 = (ni_pop[0], ni_pop[1])
            for i in infected_inds:
                i_pop = self.population[i]
                pos2 = (i_pop[0], i_pop[1])
                dist = self.get_distance(pos1, pos2)
                if dist < self.inf_rad and np.random.random() < self.inf_prob:
                    self.population[ni, 2] = 1

    def plot_history(self, history):
        plt.plot(history[:, 0]/self.N, c = 'b', linewidth = 2, label = 'healthy')
        plt.plot(history[:, 1]/self.N, c = 'r', linewidth = 2, label = 'infected')
        plt.plot(history[:, 2]/self.N, c = 'g', linewidth = 2, label = 'recovered')
        plt.legend()
        plt.xlabel("Days since first infection", size=15)
        plt.ylabel("Population percentage", size=15)
        plt.title(f"infection radius: {self.inf_rad},\npopulation speed: {self.pop_speed}, infection probability: {self.inf_prob},\nrecovery rate: {self.recovery_rate}, {self.motion} motion")

    def plot_pop(self, fname = False):
        plt.cla()
        healthy_inds, infected_inds, recovered_inds = self.get_indices()
        plt.scatter(self.population[healthy_inds][:,0], self.population[healthy_inds][:,1], c='b', label='healthy')
        plt.scatter(self.population[infected_inds][:,0], self.population[infected_inds][:,1], c='r', label='infected')
        plt.scatter(self.population[recovered_inds][:,0], self.population[recovered_inds][:,1], c='g', label='recovered')
        plt.legend(loc = 'upper right')
        if fname:
            plt.savefig(f'{fname}', dpi=300)

    """
    timestep the system
    """
    def time_evolve(self, Nsteps, verbose = False, make_movie = False):
        
        # totals of infected and recovered
        history = np.zeros((Nsteps, 3))

        for step in range(Nsteps):

            # record the status of everyone
            self.populate_history(history, step)

            # move people around
            self.move_pops(strategy = self.motion)

            # figure out who's recovered or infected
            self.infect_or_recover()
            
            if make_movie:
                self.plot_pop(fname = f"frame_{step}.png")


            if verbose:
                print(f"timestep {step}:")
                print(self)
            else:
                print(f"timestep {step} complete      \r", flush=True, end="")

        if make_movie:
            os.system(f'ffmpeg -framerate 10 -i frame_%d.png out.mp4')
        return(history)





if __name__ == "__main__":
    virus_sim = Epidemic(N = 100, inf_rad = 0.1, pop_speed = 0.05, inf_prob = 0.1, recovery_rate = 10)
    history = virus_sim.time_evolve(1000)
    print(virus_sim)
    virus_sim.plot_history(history)













