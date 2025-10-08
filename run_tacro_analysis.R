# Title: Tacrolimus Pharmacokinetic Analysis from Command Line
# Author: jbw (converted for command-line execution)
# Date: 2025-09-09

# -----------------------------------------------------------------------------
# 1. SETUP: Load Libraries & Parse Command-Line Arguments
# -----------------------------------------------------------------------------

# --- Load Libraries ---
# Ensure all required packages are installed first by running in your R console:
# install.packages(c("argparse", "tidyverse", "mrgsolve", "mapbayr", "MESS"))
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(mrgsolve))
suppressPackageStartupMessages(library(mapbayr))
suppressPackageStartupMessages(library(MESS))

# --- Define and Parse Command-Line Arguments ---
parser <- ArgumentParser(description = "Run mapbayr analysis for Tacrolimus PK data.")

parser$add_argument("--virtual_cohort", type = "character", required = TRUE,
                    help = "Path to the virtual cohort CSV file (e.g., virtual_cohort.csv)")

# parser$add_argument("--ids_to_keep", type = "character", required = TRUE,
#                     help = "Path to the CSV file containing IDs to include (e.g., ids_to_keep.csv)")

parser$add_argument("--output_dir", type = "character", default = ".",
                    help = "Directory to save the output PDF and CSV files [default: current directory]")

args <- parser$parse_args()


# -----------------------------------------------------------------------------
# 2. MODEL DEFINITION
# -----------------------------------------------------------------------------
message("Defining the tacrolimus pharmacokinetic model...")

code_tac <- "
[PROB]
# Population pharmacokinetics of tacrolimus
# Woillard de Winter BJCP 2011 

[PARAM] @annotated
TVCL : 21.2 : Typical value of clearance (L/h)
TVV1 : 486 : Typical apparent central volume of distribution (L)
TVQ : 79 : Typical intercomp clearance 1 (L/h)
TVV2 : 271 : Typical peripheral volume of distribution (L)
TVKTR : 3.34 : Typical transfer rate constant (1/h)
HTCL : -1.14 : Effect of hematocrit on clearance
CYPCL : 2.00 : Effect of CYP on clearance
STKTR : 1.53 : Effect of study on KTR
STV1 : 0.29 : Effect of study on V1


ETA1 : 0 : ETA on clearance
ETA2 : 0 : ETA on V1
ETA3 : 0 : ETA on Q
ETA4 : 0 : ETA on V2
ETA5 : 0 : ETA on KTR

$PARAM @annotated @covariates
HT : 35 : Hematocrit (percentage)
ST : 1 : Prograf (1) adv (0)
CYP : 0 : Expressor (1) non-expressor (0)


[CMT] @annotated
DEPOT : Dosing compartment (mg) [ADM]
TRANS1 : Transit compartment 1 (mg)
TRANS2 : Transit compartment 2 (mg)
TRANS3 : Transit compartment 3 (mg)
CENT : Central compartment (mg) [OBS]
PERI : Peripheral compartment (mg)

[OMEGA]
0.08
0.10
0.29
0.36
0.06

[MAIN]


// Apparent clearance (CL_app) and other PK parameters
double CL_app = TVCL * pow(HT / 35, HTCL) * pow(CYPCL, CYP) * exp(ETA1 + ETA(1));
double V1_app = TVV1  * pow(STV1, ST) * exp(ETA2 + ETA(2));

// Other parameters
double Q = TVQ * exp(ETA3 + ETA(3));
double V2 = TVV2 * exp(ETA4 + ETA(4));
double KTR = TVKTR * pow(STKTR, ST) * exp(ETA5 + ETA(5));

[SIGMA] @annotated
PROP : 0.012 : Proportional residual unexplained variability
ADD : 0.5 : Additive residual unexplained variability

[ODE]
dxdt_DEPOT = -KTR * DEPOT;
dxdt_TRANS1 = KTR * DEPOT - KTR * TRANS1;
dxdt_TRANS2 = KTR * TRANS1 - KTR * TRANS2;
dxdt_TRANS3 = KTR * TRANS2 - KTR * TRANS3;
dxdt_CENT = KTR * TRANS3 - (CL_app + Q) * CENT / V1_app + Q * PERI / V2;
dxdt_PERI = Q * CENT / V1_app - Q * PERI / V2;

[TABLE]
double CONC = CENT / (V1_app / 1000);
capture DV = CONC * (1 + PROP) + ADD;
$CAPTURE DV CL_app V1_app Q V2 KTR 
"
mod_tac <- mcode("tac_model", code_tac)

# -----------------------------------------------------------------------------
# 3. DATA PREPARATION
# -----------------------------------------------------------------------------
message(paste("Loading data from:", args$virtual_cohort))
# message(paste("Filtering IDs from:", args$ids_to_keep))

new_data <- read_csv(args$virtual_cohort, na = c("null", ".", "NA"), trim_ws = TRUE, show_col_types = FALSE) %>%
  mutate_if(is.character, factor) %>%
  filter(TIME >= 0) %>%
  mutate(
    ST = if_else(DRUG == "Advagraf", 0, 1),
    ii = if_else(DRUG == "Advagraf", 24, 12),
    ss = 1,
    addl = 4,
    evid = 1,
    cmt = 1,
  ) %>%
  select(ID, time = TIME, DV, amt = AMT, evid, ii, addl, ss, cmt, CYP, ST, AUC)

# ids_to_keep <- read_csv(args$ids_to_keep, show_col_types = FALSE)
# filtered_data <- new_data %>%
#   filter(ID %in% ids_to_keep$ID)

# --- Create the 3-point dataset for estimation ---
aadapt_3_points <- new_data %>%
  filter(
    abs(time - 0) <= 0.20 * 0 |
    abs(time - 1) <= 0.20 * 1 |
    abs(time - 3) <= 0.20 * 3
  )


# -----------------------------------------------------------------------------
# 4. ANALYSIS: MAPBAYESIAN ESTIMATION AND AUC CALCULATION
# -----------------------------------------------------------------------------
message("Starting analysis for each patient...")

# --- Define function to run analysis for one patient ID ---
run_one_id <- function(id, data) {
  df_id <- data %>%
    filter(ID == !!id) %>%
    arrange(time)

  if (nrow(df_id) == 0) return(NULL)

  # Extract consistent patient-specific values
  amt_val <- unique(na.omit(df_id$amt))[1]
  ii_val  <- unique(na.omit(df_id$ii))[1]
  auc_obs <- unique(na.omit(df_id$AUC))[1]
  st_val  <- unique(na.omit(df_id$ST))[1]
  cyp_val <- if ("CYP" %in% names(df_id)) unique(na.omit(df_id$CYP))[1] else NA_real_

  # Build the estimation dataset using mapbayr's pipeable functions
  ds <- mod_tac %>%
    adm_rows(time = 0, amt = amt_val, ss = 1, ii = ii_val, addl = 5) %>%
    add_covariates(CYP = cyp_val, ST = st_val)

  for (i in seq_len(nrow(df_id))) {
    ds <- ds %>% obs_rows(time = df_id$time[i], DV = df_id$DV[i])
  }
  # Perform estimation
  est <- ds %>% mapbayest(verbose = FALSE)

  # Plot individual fit (will be saved to PDF)
  plot(est, main = sprintf("ID %s — amt=%g mg — ii=%gh — ST=%s",
                           id, amt_val, ii_val, as.character(st_val)))
  
  # Define AUC window based on ST value
  auc_start <- 24
  auc_end <- if (st_val == 1) 36 else 48

  # Augment data to get a fine grid for IPRED curve
  aug <- mapbayr::augment(est, start = auc_start, end = auc_end, delta = 0.1)

  ipred_win <- aug$aug_tab %>%
    filter(type == "IPRED", dplyr::between(time, auc_start, auc_end)) %>%
    transmute(time, DV = value)

  # Calculate AUC using the trapezoidal rule from the MESS package
  auc_ipred <- MESS::auc(ipred_win$time, ipred_win$DV)
  ind_params <- get_param(est, .name = c("CL_app", "V1_app", "Q", "V2", "KTR"))
  
  print(ind_params)
  # Return a summary tibble
  tibble(
    ID = id,
    ST = st_val,
    amt = amt_val,
    ii  = ii_val,
    CYP = cyp_val,
    AUC_observed = auc_obs,
    auc_ipred = as.numeric(auc_ipred),
    auc_window_start = auc_start,
    auc_window_end = auc_end
  )
}

# --- Prepare output file paths ---
stamp    <- format(Sys.Date(), "%Y%m%d")
pdf_file <- file.path(args$output_dir, paste0("tacro_mapbayest_plots_", stamp, ".pdf"))
csv_file <- file.path(args$output_dir, paste0("tacro_mapbayest_auc_",   stamp, ".csv"))

# --- Execute the analysis loop over all unique IDs ---
id_list <- aadapt_3_points %>% distinct(ID) %>% pull(ID)

# Open the PDF device to save plots
pdf(pdf_file, width = 8, height = 6)
# Run the analysis for all IDs and combine results into a single data frame
results <- map_dfr(id_list, ~ run_one_id(.x, data = aadapt_3_points))
# Close the PDF device
dev.off()

# --- Export results to CSV ---
write_csv(results, csv_file)

message(sprintf("✓ Patient plots saved to: %s", pdf_file))
message(sprintf("✓ AUC results saved to: %s", csv_file))


# -----------------------------------------------------------------------------
# 5. RESULTS: CALCULATE AND DISPLAY BIAS AND RMSE
# -----------------------------------------------------------------------------
message("Calculating final bias and RMSE...")

summary_stats <- results %>%
  mutate(
    bias = (auc_ipred - AUC_observed) / AUC_observed,
    bias_sq = bias^2
  ) %>%
  summarise(
    relative_bias_percent = mean(bias, na.rm = TRUE) * 100,
    rmse_percent = sqrt(mean(bias_sq, na.rm = TRUE)) * 100
  )

# --- Print summary to the console ---
cat("\n--- Analysis Summary ---\n")
print(summary_stats)
cat("------------------------\n\n")