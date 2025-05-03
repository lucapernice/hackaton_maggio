import numpy as np
import pybamm

class Evaluator:
    def __init__(self, tasks, num_iterations=1000):
        self.tasks = tasks
        self.num_iterations = num_iterations
        self.penalty = 2.0
        
    def genererate_sequence(self):
        sequence = []
        names = []
        for _ in range(self.num_iterations):
            task = np.random.choice(self.tasks, p=[task['probability'] for task in self.tasks])
            sequence.append(task)
            names.append(task['name'])

        counts = {}
        sum = 0
        for elem in set(names):

            c = names.count(elem)
            sum += c
            counts[elem]=c/1000        
        
        return sequence, counts
    
    
    def evaluate(self):
        """
        Evaluates the expected score by simulating multiple sequences.

        Runs `num_samples` simulations, each with a randomly generated sequence
        based on task probabilities. Calculates the score for each simulation
        and returns the average score as an estimate of the expected value.
        """
        num_samples = 10 # Simulate 10 samples
        scores = []

        for _ in range(num_samples):
            # 1. Generate a sequence and its task counts for this sample
            sequence, counts = self.genererate_sequence() # sequence length is num_iterations

            # 2. Set up simulation parameters based on the sequence
            t_end = self.num_iterations - 1 # Time runs from 0 to num_iterations - 1
            dt = 1
            # Time points for the simulation and interpolation: [0, 1, ..., num_iterations - 1]
            t_eval = np.arange(0, t_end + dt, dt) # length num_iterations
            # Current profile corresponding to each time point in t_eval
            current_profile = np.zeros_like(t_eval, dtype=float) # length num_iterations

            # Populate current profile based on the generated sequence
            for i, task in enumerate(sequence):
                # Ensure index i is within bounds of current_profile
                if i < len(current_profile):
                    current_profile[i] = round(task['power'], 2)
                else:
                    # This case should ideally not happen if len(sequence) == num_iterations
                    print(f"Warning: Sequence length ({len(sequence)}) mismatch with num_iterations ({self.num_iterations}) at index {i}")
                    break # Stop populating if sequence is longer than expected

            # 3. Run the PyBaMM simulation
            sample_score = -np.inf # Initialize score to indicate potential failure
            # Check if t_eval has enough points for interpolation (at least 2)
            if len(t_eval) < 2:
                 print(f"Warning: Not enough time points for interpolation ({len(t_eval)}). Skipping sample.")
                 # Keep sample_score as -np.inf
            else:
                try:
                    model = pybamm.lithium_ion.SPM()
                    param = model.default_parameter_values

                    # Create interpolant: current_profile[i] is the current at time t_eval[i]
                    # pybamm.t represents the simulation time variable
                    current_input = pybamm.Interpolant(t_eval, current_profile, pybamm.t)
                    param.update({"Current function [A]": current_input})

                    sim = pybamm.Simulation(model, parameter_values=param)

                    # Solve the simulation using the specified time points for output
                    solution = sim.solve(t_eval=t_eval)

                    # Get the final discharge capacity result
                    res = solution["Discharge capacity [A.h]"].data[-1]

                    # 4. Calculate the score for this sample
                    score_part = 1.0 # Use float for multiplicative accumulation
                    for task in self.tasks:
                        task_name = task['name']
                        # Get the proportion of this task in the sequence (0 if not present)
                        c = counts.get(task_name, 0.0)
                        # Multiply by the task's priority
                        score_part *= (c * float(task['priority'])) # Ensure priority is float

                    # Final score combines the task distribution score and the discharge penalty
                    sample_score = score_part * 100 - self.penalty * res

                except Exception as e:
                    # Catch potential errors during simulation or scoring
                    print(f"Error during simulation or scoring for sample: {e}")
                    # Keep sample_score as -np.inf to indicate failure for this sample

            scores.append(sample_score)

        # 5. Calculate the expected value (average score) from successful samples
        valid_scores = [s for s in scores if s != -np.inf]
        if not valid_scores: # Handle case where all samples failed
            print("Warning: All simulation samples failed.")
            return 0.0 # Or return -np.inf or raise an error, depending on desired behavior
        
        expected_score = np.mean(valid_scores)
        return expected_score
     