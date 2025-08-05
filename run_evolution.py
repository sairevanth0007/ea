# run_evolution_standalone.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import json
from ab.nn.util.Util import uuid4
import time

# --- Step 1: Import our custom modules ---
from AlexNet_evolvable import Net, SEARCH_SPACE, generate_model_code_string
from genetic_algorithm import GeneticAlgorithm

# --- Step 2: Define Experiment Parameters (Adjusted for ~1000 evaluations) ---
# GA Parameters
POPULATION_SIZE = 50      # Increased
NUM_GENERATIONS = 25      # Increased
MUTATION_RATE = 0.15
ELITISM_COUNT = 5         # Increased (10%)
CHECKPOINT_FILE = 'ga_evolution_checkpoint.pkl'

# Fitness Evaluation Parameters
BATCH_SIZE = 128
NUM_EPOCHS_PER_EVAL = 5   # Increased for better fitness estimate

# --- NEW: Paths and Setup ---
ARCHITECTURE_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'ga_architecture')
STATS_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'stats')
CHAMPION_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'champ_alexnet_ga.py')
os.makedirs(ARCHITECTURE_SAVE_DIR, exist_ok=True)
os.makedirs(STATS_SAVE_DIR, exist_ok=True)

# --- NEW: Global variables for tracking ---
seen_checksums = set()
architecture_counter = 0 # Reset counter for new run logic

# --- Step 3: The Main Execution Block ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load CIFAR-10 dataset ---
    print("Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_subset, val_subset = random_split(full_train_set, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    in_shape = (3, 32, 32)
    out_shape = (10,)
    print(f"Input shape: {in_shape}, Output shape: {out_shape}")

    # --- Step 4: Define the Enhanced Fitness Function ---
    def fitness_function(chromosome: dict) -> float:
        global architecture_counter # Access the global counter

        # --- 1. Generate Model Code String for Duplicate Check ---
        try:
            model_code_string = generate_model_code_string(chromosome, in_shape, out_shape)
        except Exception as e:
            print(f"  - Error generating code string for chromosome {chromosome}: {e}. Assigning fitness 0.")
            return 0.0

        # --- 2. Calculate Checksum ---
        model_checksum = uuid4(model_code_string)

        # --- 3. Check for Duplicate ---
        if model_checksum in seen_checksums:
             print(f"  - Duplicate architecture detected (checksum: {model_checksum}). Skipping evaluation and saving.")
             return 0.0

        # --- 4. Internal Training/Evaluation ---
        try:
            print(f"  - Evaluating unique architecture (checksum: {model_checksum[:8]}...)")
            start_time = time.time()

            model = Net(in_shape, out_shape, chromosome, device)
            model.train_setup(prm=chromosome)

            for epoch in range(NUM_EPOCHS_PER_EVAL):
                model.learn(train_loader)

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            duration_ns = int((time.time() - start_time) * 1_000_000_000)
            print(f"  - Chromosome evaluated. Fitness (Accuracy): {accuracy:.2f}%")

            # --- Determine the architecture number ---
            # Use the global counter for naming
            current_arch_number = architecture_counter

            # --- 5. Save Stats to JSON File (with corrected naming) ---
            stats_filename = f"img-classification_cifar-10_acc_alexnet_ga_{current_arch_number}.json"
            stats_filepath = os.path.join(STATS_SAVE_DIR, stats_filename)

            # Prepare stats dictionary (matching the requested format exactly)
            stats_data = {
                "accuracy": round(accuracy / 100.0, 4), # Convert % to fraction and round
                "batch": BATCH_SIZE, # Use the constant BATCH_SIZE
                "dropout": round(chromosome.get('dropout', 0.0), 4),
                "duration": duration_ns,
                "lr": round(chromosome.get('lr', 0.0), 4),
                "momentum": round(chromosome.get('momentum', 0.0), 4),
                "transform": "norm_256_flip", # Add default transform as requested
                "uid": model_checksum,
            }

            # Save stats JSON (wrapped in a list as requested)
            try:
                with open(stats_filepath, 'w') as f:
                    json.dump([stats_data], f, indent=4)
                print(f"  - Stats saved to JSON file: {stats_filepath}")
            except Exception as json_save_error:
                print(f"  - Error saving stats to JSON file {stats_filepath}: {json_save_error}")

            # --- 6. Save Unique Architecture Code to File (with original naming) ---
            arch_filename = f"alexnet_ga_{current_arch_number}.py"
            arch_filepath = os.path.join(ARCHITECTURE_SAVE_DIR, arch_filename)
            try:
                with open(arch_filepath, 'w') as f:
                    f.write(model_code_string)
                print(f"  - Unique architecture code saved to: {arch_filepath}")
                architecture_counter += 1 # Increment counter for next file
            except Exception as save_error:
                print(f"  - Error saving architecture file {arch_filepath}: {save_error}")

            # --- 7. Add Checksum to Seen Set ---
            seen_checksums.add(model_checksum)

            return accuracy

        except Exception as e:
            print(f"  - Error evaluating chromosome: {e}. Assigning fitness 0.")
            # Consider if failed evals should count as "seen" to prevent retries of bad configs
            # seen_checksums.add(model_checksum) # Optional
            # Increment counter even on failure to maintain sequential numbering?
            # architecture_counter += 1 # Optional: Depends on desired numbering behavior for failed evals
            return 0.0

    # --- Step 5: Initialize and Run the Genetic Algorithm ---
    print("\n--- Starting Genetic Algorithm ---")

    # Attempt to resume architecture counter from existing files if needed
    # This is a simple heuristic: find the highest number in existing alexnet_ga_X.py files
    try:
        existing_arch_files = [f for f in os.listdir(ARCHITECTURE_SAVE_DIR) if f.startswith("alexnet_ga_") and f.endswith(".py")]
        if existing_arch_files:
            numbers = []
            for f in existing_arch_files:
                parts = f.replace("alexnet_ga_", "").replace(".py", "").split('_') # Handle potential underscores
                for part in parts:
                    if part.isdigit():
                        numbers.append(int(part))
            if numbers:
                architecture_counter = max(numbers) + 1
                print(f"  - Resumed architecture counter from existing files: {architecture_counter}")
    except OSError as e:
        print(f"  - Could not scan {ARCHITECTURE_SAVE_DIR} to resume counter: {e}. Starting from 0.")

    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        search_space=SEARCH_SPACE,
        elitism_count=ELITISM_COUNT,
        mutation_rate=MUTATION_RATE,
        checkpoint_path=CHECKPOINT_FILE
    )

    best_individual = ga.run(
        num_generations=NUM_GENERATIONS,
        fitness_function=fitness_function
    )

    # --- Step 6: Display Final Results and Save Champion ---
    print("\n--- Evolution Finished! ---")
    if best_individual:
        print("Best performing network architecture found:")
        print(f"  - Fitness (Validation Accuracy): {best_individual['fitness']:.2f}%")
        print("  - Chromosome (Parameters):")
        for gene, value in best_individual['chromosome'].items():
            print(f"    - {gene}: {value}")

        # --- NEW: Save Champion Architecture ---
        try:
            champion_code_string = generate_model_code_string(best_individual['chromosome'], in_shape, out_shape)
            with open(CHAMPION_SAVE_PATH, 'w') as f:
                f.write(champion_code_string)
            print(f"\n--- Champion architecture saved to: {CHAMPION_SAVE_PATH} ---")
        except Exception as champ_error:
            print(f"\n--- Error saving champion architecture: {champ_error} ---")

    else:
        print("No successful individual found in any generation (all had errors).")

    print("\nTo fully train this best model, you would now create a new Net with this")
    print("chromosome and train it for many more epochs (e.g., 50-100).")
