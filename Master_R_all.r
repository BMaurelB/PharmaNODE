# --- Master R Script to Control the Entire Workflow ---

# --- Configuration ---
START_SEED <- 1
END_SEED <- 10 # Change this to your desired end seed
CONDA_ENV <- "NODE_test" # Define your conda environment name
setwd("~/Documents/Training_Sanofi/notebooks_hands_on/latent_ode")
# Define your script names
GENERATE_DATA_PY <- 'gen_tacro.py'
RUN_TACRO_R      <- 'all_run_tacro.R'
TRAIN_MODEL_PY   <- "run_models.py"
TEST_MODEL_PY    <- "test_model.py"
ANALYZE_PY       <- "analyse_std.py"
CONDA_PATH <- "/opt/miniconda3"

# --- Setup ---
ALL_TEST_RESULTS_FILE <- "test_gen_tacro_all.txt"
file.create(ALL_TEST_RESULTS_FILE, showWarnings = FALSE)

if (!dir.exists("logs")) {
  dir.create("logs")
}

cat("Starting multiple experiment runs...\n")


# --- Main Loop ---
for (seed in START_SEED:END_SEED) {
  
  experiment_id <- sample(10000:99999, 1)
  
  cat("----------------------------------------------------\n")
  cat(sprintf("Running experiment with Seed: %d, Experiment ID: %d\n", seed, experiment_id))
  
  # --- Step 1: Generate Data (Python) ---
  cat("Generating data...\n")
  system(sprintf("conda run -n %s python3 %s", CONDA_ENV, GENERATE_DATA_PY))
  
  # --- Step 2: Run the RStudio-dependent script ---
  cat("Running the R script inside RStudio...\n")
  source(RUN_TACRO_R)
  
  # --- Step 3: Train the model (Python) ---
  cat("Training model...\n")
  base_command_train <- sprintf(
    "conda run -n %s python3 %s --niters 5000 -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --noise-weight 0.01 --max-t 5. --seed %d --experiment %d",
    CONDA_ENV, TRAIN_MODEL_PY, seed, experiment_id
  )
  log_file_train <- sprintf("logs/train_run_models_%d.log", experiment_id)
  
  system(paste(base_command_train, ">", log_file_train, "2>&1"))
  cat(sprintf("Training complete. Check log: %s\n", log_file_train))
  
  # --- Step 4: Test the model (Python) ---
  cat("Testing model...\n")
  base_command_test <- sprintf(
    "conda run -n %s python3 %s -n 200 -s 40 -l 10 --dataset PK_Tacro --latent-ode --noise-weight 0.01 --max-t 5. --seed %d --load %d",
    CONDA_ENV, TEST_MODEL_PY, seed, experiment_id
  )
  
  test_output <- system(base_command_test, intern = TRUE)
  results_to_append <- grep("Error comparison:|RMSE comparison:|- error.*=|- error_be.*=|- rmse.*=|- rmse_be.*=", test_output, value = TRUE)
  
  write(sprintf("\n--- Results for Seed %d (Experiment ID: %d) ---", seed, experiment_id), file = ALL_TEST_RESULTS_FILE, append = TRUE)
  write(results_to_append, file = ALL_TEST_RESULTS_FILE, append = TRUE)
  
  cat("Testing complete.\n")
  
}

# --- Final Step: Analyze results (Python) ---
cat("----------------------------------------------------\n")
cat("Running analysis script...\n")
system(sprintf("conda run -n %s python3 %s %s", CONDA_ENV, ANALYZE_PY, ALL_TEST_RESULTS_FILE))

cat("----------------------------------------------------\n")
cat("Analysis complete.\n")