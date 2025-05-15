from sklearn.model_selection._split import _BaseKFold
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.stats import wasserstein_distance
from sklearn.model_selection import KFold
from tqdm import tqdm


class WDESKFold(_BaseKFold):
    def __init__(self, n_splits=10, shuffle=False, random_state=None, n_gen = 50, n_pop = 100):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.n_gen = n_gen
        self.n_pop = n_pop
        # Initializes equal split
        self.r = np.asarray([1 / self.n_splits] * self.n_splits)

    def split(self, dataset):
        self.dataset = dataset

        # Get dataaset information
        self.num_classes = self.dataset.num_classes
        self.num_samples = len(self.dataset)

        # Get mask information 
        self.pixel_counts = np.zeros([self.num_samples, self.num_classes])
        print("Reading dataset information for stratifier")
        for i in tqdm(range(len(self.dataset))):
            _, mask = self.dataset[i]
            self.pixel_counts[i,:] = [np.bincount(mask.flatten(), minlength=self.num_classes)[j] 
                                      for j in range(self.num_classes)]
            
        self.n_pixels_per_class =  self.pixel_counts.sum(axis = 0)
        self.desired_n_samples_in_fold = self.r * self.num_samples

        # Calculate optimal splits
        self.best_folds = self.optimize()
        self.best_folds = np.asarray(self.best_folds)

        # Yield
        for fold_number in range(self.n_splits):
            test_indices = np.where(self.best_folds == fold_number)[0]
            train_indices = np.where(self.best_folds != fold_number)[0]
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    def _fitness(self, individual):
        pdm = []
        for fold_number in range(self.n_splits):
            samples_in_fold = [i == fold_number for i in individual]
            pdm_f = []
            for class_number in range(self.num_classes):
                pdm_f_c = wasserstein_distance(u_values= self.pixel_counts[samples_in_fold, class_number], 
                                            v_values= self.pixel_counts[:, class_number])
                pdm_f.append(pdm_f_c)
            pdm.append(pdm_f)

        return (np.mean(np.array(pdm)), )
        
    def create_equal_distribution_individual(self, num_params, folds):
        # Calculate the base number of times each integer should appear
        base_count = num_params // (folds)
        remainder = num_params % (folds)

        # Create the list with the base count of each integer
        result = []
        for i in range(folds):
            result.extend([i] * base_count)

        # Distribute the remainder randomly
        for _ in range(remainder):
            result.append(random.randint(0, folds-1))

        # Shuffle the list to ensure randomness
        random.shuffle(result)
        return result
    
    def create_equal_distribution_individual_kfold(self, num_params, folds):
        # Create an array of indices
        indices = list(range(num_params))

        # Initialize KFold with the specified number of splits
        kf = KFold(n_splits=folds, shuffle=True)

        # Collect the indices for each split
        split_indices = [0] * num_params
        for i, (_, test_indices) in enumerate(kf.split(indices)):
            for index in test_indices:
                split_indices[index] = i

        return split_indices
    
    def float_to_int_array(self, float_array):
        # Calculate the total sum of the float array
        total_sum = sum(float_array)

        # Initialize the integer array with the floor values of the float array
        int_array = [int(num) for num in float_array]

        # Calculate the remaining sum that needs to be distributed
        remaining_sum = int(total_sum) - sum(int_array)

        # Distribute the remaining sum one by one to the integer array
        for i in range(remaining_sum):
            int_array[i % len(int_array)] += 1

        return int_array
    
    def correct_distribution(self, individual):
        desired_ed = self.float_to_int_array(self.desired_n_samples_in_fold)

        max_i = len(individual)
        iter = 0
        # Iterate until the list matches the target counts
        while iter < max_i:
            # Check if the list is already correct
            if all(abs(individual.count(i) - desired_ed[i]) == 0 for i in range(len(desired_ed))):
                break

            # Find the first number that appears more than its target count
            for i in range(len(desired_ed)):
                if individual.count(i) > desired_ed[i]:
                    excess_number = i
                    break

            # Find the first number that appears less than its target count
            for i in range(len(desired_ed)):
                if individual.count(i) < desired_ed[i]:
                    deficit_number = i
                    break

            # Replace the first occurrence of the excess number with the deficit number
            individual[individual.index(excess_number)] = deficit_number
            iter += 1

        assert all(abs(individual.count(i) - desired_ed[i]) == 0 for i in range(len(desired_ed)))
        return individual

    def uniform_mate_correct(self, ind1, ind2, indpb = 0.5):
        size = min(len(ind1), len(ind2))
        for i in range(size):
            if random.random() < indpb:
                ind1[i], ind2[i] = ind2[i], ind1[i]

        # Correct the offspring to ensure each integer appears exactly twice
        ind1 = self.correct_distribution(ind1)
        ind2 = self.correct_distribution(ind2)

        return ind1, ind2

    def optimize(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        toolbox.register("individual", tools.initIterate, creator.Individual,
                        lambda: self.create_equal_distribution_individual(self.num_samples, self.n_splits))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self._fitness)

        toolbox.register("mate", self.uniform_mate_correct, indpb=0.5)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=self.n_pop)
        hof = tools.HallOfFame(1)  # Store the best solution

        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)  # Assign fitness values

        # Evolutionary loop with fitness tracking
        print("Starting WDES")
        for _ in tqdm(range(1, self.n_gen + 1)):
            pop, _ = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=1, 
                                        halloffame=hof, verbose=False)
        
        return hof[0]  # Best individual found    

