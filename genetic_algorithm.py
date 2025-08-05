# genetic_algorithm.py

import random
from tqdm import tqdm  # A nice library for progress bars! Install with: pip install tqdm
import pickle  # For saving/loading the state
import os  # For checking file existence


class GeneticAlgorithm:
    """
    This class implements the core logic for the genetic algorithm.
    It is designed to be general and can be used for other optimization
    problems, not just neural architecture search.
    """

    def __init__(self, population_size, search_space, elitism_count, mutation_rate,
                 checkpoint_path='ga_checkpoint.pkl'):
        """
        Initializes the GA engine.

        Args:
            population_size (int): The number of individuals (chromosomes) in each generation.
            search_space (dict): The dictionary defining the search space (from Part 1).
            elitism_count (int): The number of best individuals to carry over to the next generation.
            mutation_rate (float): The probability (0.0 to 1.0) of a gene mutating.
            checkpoint_path (str): Path to save/load checkpoint file.
        """
        self.population_size = population_size
        self.search_space = search_space
        self.elitism_count = elitism_count
        self.mutation_rate = mutation_rate
        self.population = []  # This will hold our list of individuals
        self.checkpoint_path = checkpoint_path

    def _create_random_chromosome(self):
        """Creates a single random chromosome."""
        chromosome = {}
        for key, values in self.search_space.items():
            chromosome[key] = random.choice(values)
        return chromosome

    def _initialize_population(self):
        """Creates the initial population of random individuals."""
        for _ in range(self.population_size):
            chromosome = self._create_random_chromosome()
            # We initialize fitness to None. It will be calculated later.
            self.population.append({'chromosome': chromosome, 'fitness': None})
        print(f"Initialized population with {self.population_size} individuals.")

    def _save_checkpoint(self, generation_num):
        """Saves the current state to a file."""
        state = {
            'generation': generation_num,
            'population': self.population
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"--- Checkpoint saved for Generation {generation_num} ---")

    def _load_checkpoint(self):
        """Loads state from a checkpoint file if it exists."""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'rb') as f:
                state = pickle.load(f)
            print(f"--- Resuming from checkpoint at Generation {state['generation']} ---")
            return state['generation'], state['population']
        return 0, None  # Start from scratch if no checkpoint

    def _crossover(self, parent1_chromo, parent2_chromo):
        """
        Performs single-point crossover between two parent chromosomes.
        """
        child_chromo = {}
        genes = list(self.search_space.keys())
        crossover_point = random.randint(1, len(genes) - 1)

        for i, gene in enumerate(genes):
            if i < crossover_point:
                child_chromo[gene] = parent1_chromo[gene]
            else:
                child_chromo[gene] = parent2_chromo[gene]

        return child_chromo

    def _mutate(self, chromosome):
        """
        Mutates a chromosome by randomly changing some of its genes.
        """
        mutated_chromo = chromosome.copy()
        for gene in self.search_space.keys():
            if random.random() < self.mutation_rate:
                # Pick a new value that is different from the current one
                current_value = mutated_chromo[gene]
                possible_values = [v for v in self.search_space[gene] if v != current_value]
                if possible_values:  # Make sure there's another option to switch to
                    mutated_chromo[gene] = random.choice(possible_values)
        return mutated_chromo

    def _selection(self):
        """
        Selects a single individual from the population using tournament selection.
        """
        # We'll use a tournament size of 3, a common choice.
        tournament_size = 3
        # Pick 3 random individuals from the population
        competitors = random.sample(self.population, tournament_size)
        # Sort them by fitness (highest first) and return the winner
        winner = sorted(competitors, key=lambda x: x['fitness'], reverse=True)[0]
        return winner

    def run(self, num_generations, fitness_function):
        """
        The main loop that runs the entire evolutionary process.
        """
        start_gen, loaded_population = self._load_checkpoint()

        if loaded_population:
            self.population = loaded_population
        else:
            self._initialize_population()

        # Initialize variable to track the best individual found across all generations
        best_ever_individual = None
        # If loading from checkpoint, ensure best_ever_individual is set from the loaded population
        if loaded_population:
            # Sort the loaded population to find its best and set it as best_ever_individual
            # Ensure fitness values are not None (they should be, if loaded from a completed generation)
            evaluated_individuals = [ind for ind in self.population if ind['fitness'] is not None]
            if evaluated_individuals:
                best_loaded = sorted(evaluated_individuals, key=lambda x: x['fitness'], reverse=True)[0]
                best_ever_individual = best_loaded.copy()

        for gen in range(start_gen, num_generations):
            print(f"\n===== Generation {gen + 1}/{num_generations} =====")

            # 1. Fitness Evaluation: Calculate fitness for each individual
            # Only evaluate individuals whose fitness is still None (newly created or loaded without evaluation)
            print("Evaluating fitness of population...")
            for individual in tqdm(self.population):
                if individual['fitness'] is None:
                    individual['fitness'] = fitness_function(individual['chromosome'])

            # 2. Sort the population by fitness (best first)
            self.population.sort(key=lambda x: x['fitness'], reverse=True)

            # --- UPDATE: Compare the best of this generation with the best ever found ---
            current_best = self.population[0]
            if best_ever_individual is None or current_best['fitness'] > best_ever_individual['fitness']:
                # IMPORTANT: Make a copy to prevent future modifications to this individual's chromosome
                best_ever_individual = current_best.copy()
                print(f"*** New best overall fitness found: {best_ever_individual['fitness']:.4f} ***")

            best_fitness = self.population[0]['fitness']
            print(f"Best fitness in Generation {gen + 1}: {best_fitness:.4f}")
            print(f"Best chromosome: {self.population[0]['chromosome']}")

            # 3. Create the next generation
            next_generation = []

            # 3.1 Elitism: The best individuals automatically survive
            if self.elitism_count > 0:
                elites = self.population[:self.elitism_count]
                next_generation.extend(elites)
                print(f"Carried over {len(elites)} elite individuals.")

            # 3.2 Crossover and Mutation: Fill the rest of the population
            num_children = self.population_size - self.elitism_count
            if num_children > 0:
                print(f"Creating {num_children} new individuals through crossover and mutation...")

                for _ in range(num_children):
                    # Select two parents via tournament selection
                    parent1 = self._selection()
                    parent2 = self._selection()

                    # Create a child through crossover
                    child_chromosome = self._crossover(parent1['chromosome'], parent2['chromosome'])

                    # Mutate the child
                    mutated_child_chromosome = self._mutate(child_chromosome)

                    # Add the new child to the next generation
                    next_generation.append({'chromosome': mutated_child_chromosome, 'fitness': None})

            # Replace the old population with the new one
            self.population = next_generation

            # --- SAVE CHECKPOINT AT THE END OF EACH GENERATION ---
            self._save_checkpoint(generation_num=gen + 1)

        # After all generations, return the best individual found that was tracked
        print("\n===== Evolution Complete =====")
        return best_ever_individual