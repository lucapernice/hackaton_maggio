import json
import random
import math
import numpy as np
from utils.eval import Evaluator
import warnings
from tqdm import tqdm
import csv # Import the csv module

#read the json file
with open('tasks.json') as f:
    data = json.load(f)



def simulated_annealing(initial_temp=1.0, cooling_rate=0.95, n_iterations=100, output_csv='annealing_log.csv'):

    # Initialize the evaluator with the tasks
    current_solution = data
    evaluator = Evaluator(current_solution)
    current_score = evaluator.evaluate()
    best_solution = current_solution
    best_score = current_score
    temp = initial_temp

    # Open CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['iteration', 'temperature', 'current_score', 'new_score', 'accepted', 'best_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in tqdm(range(n_iterations), desc="Simulated Annealing"):
            new_solution = perturb_solution(current_solution)
            new_evaluator = Evaluator(new_solution)
            new_score = new_evaluator.evaluate()

            accept = False # Initialize accept flag
            if new_score > current_score:
                accept = True
            else:
                delta = new_score - current_score
                # Avoid division by zero or very small temp
                if temp > 1e-9:
                    probability = math.exp(delta / temp)
                else:
                    probability = 0.0 # Effectively reject worse solutions at very low temp
                if random.random() < probability:
                    accept = True

            if accept:
                current_solution = new_solution
                current_score = new_score

                if current_score > best_score:
                    best_solution = current_solution # Update best_solution only if accepted and better
                    best_score = current_score

            # Log data for the current iteration
            writer.writerow({
                'iteration': i + 1,
                'temperature': temp,
                'current_score': current_score, # Log the score *after* potential update
                'new_score': new_score,
                'accepted': accept,
                'best_score': best_score
            })

            temp *= cooling_rate


    return best_solution, best_score



def perturb_solution(tasks):
    # Ensure we are working with a deep copy if tasks contain nested structures like dicts
    new_tasks = [task.copy() for task in tasks]
    probabilities = []
    for task in new_tasks:
        probabilities.append(task['probability'])

    #perturb the probabilities (ensuring they sum to 1)
    perturbation = np.random.uniform(-0.1, 0.1, size=len(probabilities))
    new_probabilities = np.array(probabilities) + perturbation
    new_probabilities = np.clip(new_probabilities, 0, None) # Ensure non-negative probabilities
    # Handle the case where all probabilities become zero after clipping
    if new_probabilities.sum() == 0:
         # Re-initialize probabilities uniformly if sum is zero
         new_probabilities = np.ones(len(new_tasks)) / len(new_tasks)
    else:
        new_probabilities /= new_probabilities.sum() # Normalize to sum to 1

    for i, task in enumerate(new_tasks):
        task['probability'] = new_probabilities[i]

    return new_tasks


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # Pass the desired csv filename to the function
    best_solution, best_score = simulated_annealing(output_csv='simulation_log.csv')
    print("Best solution found:")
    # Consider printing only a summary or relevant parts if the solution is large
    print(best_solution)
    print("Best score:")
    print(best_score)
    print("Simulation log saved to simulation_log.csv")
    