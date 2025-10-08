import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as pt

# Load the data
data = pd.read_csv('virtual_cohort.csv')
print("Data loaded successfully!")
print(f"Shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"Number of patients: {data['ID'].nunique()}")

# Filter out dosing events and keep only observations
obs_data = data[data['AMT'] == 0].copy()
dose_data = data[data['AMT'] > 0].copy()

print(f"Observation records: {len(obs_data)}")
print(f"Dosing records: {len(dose_data)}")

# Prepare data for modeling
patient_ids = obs_data['ID'].values
n_patients = len(obs_data['ID'].unique())

# Extract covariates
cyp_status = obs_data.groupby('ID')['CYP'].first().values
dosing_interval = obs_data.groupby('ID')['II'].first().values

# Standardize continuous covariates
hematocrit_mean = 37.5  # Based on your simulation range (30-45)
hematocrit_std = 4.5
# Note: You'll need to add hematocrit to your dataset if you want to use it

# Prepare time and concentration data
times = obs_data['TIME'].values
concentrations = obs_data['DV'].values

# Get unique patient IDs and create an index mapping
unique_ids = np.unique(patient_ids)
id_idx = np.searchsorted(unique_ids, patient_ids)

print("Data preparation complete!")

# Define the PK model function
def pk_model(t, dose, CL, Vc, Ka, F=1.0):
    """
    One-compartment model with first-order absorption
    """
    # Avoid division by zero and negative values
    CL = pt.maximum(CL, 0.01)
    Vc = pt.maximum(Vc, 0.01)
    Ka = pt.maximum(Ka, 0.01)
    
    # Calculate microconstants
    kel = CL / Vc
    
    # Model prediction
    concentration = (F * dose * Ka / (Vc * (Ka - kel))) * (pt.exp(-kel * t) - pt.exp(-Ka * t))
    return concentration

# Create the PyMC model
with pm.Model() as tacro_model:
    # Population priors (typical values)
    tv_CL = pm.LogNormal("tv_CL", mu=np.log(20), sigma=0.5)
    tv_Vc = pm.LogNormal("tv_Vc", mu=np.log(400), sigma=0.5)
    tv_Ka = pm.LogNormal("tv_Ka", mu=np.log(2), sigma=0.5)
    
    # Covariate effects
    # CYP3A5 expresser status effect on CL
    beta_CL_CYP = pm.Normal("beta_CL_CYP", mu=0, sigma=0.5)
    
    # Dosing interval effect on Ka (different for Prograf vs Advagraf)
    beta_Ka_II = pm.Normal("beta_Ka_II", mu=0, sigma=0.5)
    
    # Inter-individual variability (IIV)
    omega_CL = pm.HalfNormal("omega_CL", sigma=0.3)
    omega_Vc = pm.HalfNormal("omega_Vc", sigma=0.3)
    omega_Ka = pm.HalfNormal("omega_Ka", sigma=0.3)
    
    # Individual random effects
    eta_CL = pm.Normal("eta_CL", 0, 1, shape=n_patients)
    eta_Vc = pm.Normal("eta_Vc", 0, 1, shape=n_patients)
    eta_Ka = pm.Normal("eta_Ka", 0, 1, shape=n_patients)
    
    # Individual parameters
    # CYP effect: CL is multiplied by exp(beta_CL_CYP) for expressers
    indiv_CL = pm.Deterministic(
        "indiv_CL", 
        tv_CL * pt.exp(eta_CL * omega_CL) * pt.exp(beta_CL_CYP * cyp_status)
    )
    
    indiv_Vc = pm.Deterministic(
        "indiv_Vc", 
        tv_Vc * pt.exp(eta_Vc * omega_Vc)
    )
    
    # Dosing interval effect: Ka is adjusted based on formulation
    indiv_Ka = pm.Deterministic(
        "indiv_Ka", 
        tv_Ka * pt.exp(eta_Ka * omega_Ka) * pt.exp(beta_Ka_II * (dosing_interval[id_idx] - 18)/6)  # centered around 18h
    )
    
    # Get dose information for each observation
    # For simplicity, we'll use the last dose before each observation
    # This is a simplification - in practice, you'd need a more complex dosing history
    dose_amounts = np.zeros(len(obs_data))
    dose_times = np.zeros(len(obs_data))
    
    for i, (idx, row) in enumerate(obs_data.iterrows()):
        patient_id = row['ID']
        obs_time = row['TIME']
        
        # Find the most recent dose for this patient
        patient_doses = dose_data[(dose_data['ID'] == patient_id) & (dose_data['TIME'] <= obs_time)]
        if len(patient_doses) > 0:
            last_dose = patient_doses.iloc[-1]
            dose_amounts[i] = last_dose['AMT']
            dose_times[i] = obs_time - last_dose['TIME']  # time since last dose
        else:
            dose_amounts[i] = 0
            dose_times[i] = obs_time
    
    # Model prediction
    conc_pred = pk_model(
        dose_times,
        dose_amounts,
        indiv_CL[id_idx],
        indiv_Vc[id_idx],
        indiv_Ka[id_idx]
    )
    
    # Residual error model (proportional + additive)
    sigma_prop = pm.HalfNormal("sigma_prop", sigma=0.2)
    sigma_add = pm.HalfNormal("sigma_add", sigma=1.0)
    
    # Likelihood
    sigma_total = pt.sqrt((sigma_prop * conc_pred)**2 + sigma_add**2)
    obs = pm.Normal("obs", mu=conc_pred, sigma=sigma_total, observed=concentrations)
    
    # Sample from the posterior
    trace = pm.sample(
        1000, 
        tune=1000, 
        chains=4, 
        target_accept=0.9,
        return_inferencedata=True
    )

# Display results
print("Sampling complete! Here are the results:")
print(az.summary(trace, var_names=["tv_CL", "tv_Vc", "tv_Ka", "beta_CL_CYP", "beta_Ka_II", 
                                  "omega_CL", "omega_Vc", "omega_Ka", "sigma_prop", "sigma_add"]))

# Plot diagnostics
az.plot_trace(trace, var_names=["tv_CL", "tv_Vc", "tv_Ka", "beta_CL_CYP", "beta_Ka_II"])
plt.tight_layout()
plt.savefig('trace_plot.png')
plt.close()

# Plot posterior distributions
az.plot_posterior(trace, var_names=["tv_CL", "tv_Vc", "tv_Ka", "beta_CL_CYP", "beta_Ka_II"])
plt.tight_layout()
plt.savefig('posterior_plot.png')
plt.close()

# Check convergence
rhat = az.rhat(trace)
print("\nR-hat values (should be < 1.01):")
print(rhat)

# Posterior predictive check
with tacro_model:
    ppc = pm.sample_posterior_predictive(trace, extend_inferencedata=True)

# Plot posterior predictive check
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_ppc(ppc, num_pp_samples=100, ax=ax)
plt.savefig('ppc_plot.png')
plt.close()

print("Analysis complete! Check the generated plots for diagnostics.")