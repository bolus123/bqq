# Statistical Inference for BQQ Model - Version 4
#
# This version uses DATA-ADAPTIVE penalties for gamma and interquantile shrinkage,
# with cross-validation only for the non-crossing penalty (lambda_nc).
#
# Key changes from v3:
# - adaptive_gamma = TRUE (lambda^2 learned from data via Gamma prior)
# - IQ shrinkage via adaptive lambda_iq2 ~ Gamma(a, b) (or fixed lambda_iq2_fixed)
# - Cross-validation only tunes lambda_nc
#
# Key methodology (from BQQ_Methodology.md):
# - Posterior covariance: Sigma computed from MCMC/Laplace samples
# - Whitening: gamma_tilde^(s) = Sigma^{-1/2} gamma^(s)
# - Test statistic: T_tilde_j^(s) = sum_q (gamma_tilde_{qj}^(s))^2
# - Bayesian p-value: p_j = (1/S) sum_s 1(T_tilde_j^(s) <= epsilon)

# =============================================================================
# Setup
# =============================================================================

#source("bqq/R/cv_copss.R")
#source("bqq/R/getModel.R")
#source("bqq/R/getDesignMatrix.R")
#source("bqq/R/getInference.R")

#devtools::install_github("bolus123/bqq")

require(bqq)

set.seed(2024)

cat("================================================================\n")
cat("BQQ Statistical Inference - Data-Adaptive Penalties (v4)\n")
cat("================================================================\n\n")

# =============================================================================
# PART 1: Data Simulation
# =============================================================================

cat("=== PART 1: Data Simulation ===\n\n")

n <- 360
y <- rnorm(n, mean = 0, sd = 1)

# Add a sustained shift at 80% of the data
shift_start <- round(0.7 * n)
shift_magnitude <- 1
y[shift_start:n] <- y[shift_start:n] + shift_magnitude

cat("Data simulation:\n")
cat("  Total observations:", n, "\n")
cat("  Shift starts at observation:", shift_start, "\n")
cat("  Shift magnitude:", shift_magnitude, "\n\n")

# =============================================================================
# PART 2: Model Setup
# =============================================================================

cat("=== PART 2: Model Setup ===\n\n")

# Quintuple quantiles (for QSS statistics)
taus <- c(0.025, 0.25, 0.5, 0.75, 0.975)

# Block length and warm-up period
l <- 30   # block length
w <- 30   # warm-up period

# Sustained shift design matrix
H <- getSustainedShift(n = n, l = l, w = w)

cat("Model setup:\n")
cat("  Quantile levels:", paste(taus, collapse = ", "), "\n")
cat("  Block length (l):", l, "\n")
cat("  Warm-up period (w):", w, "\n")
cat("  H matrix dimensions:", nrow(H), "x", ncol(H), "\n")
cat("  Number of gamma coefficients per quantile:", ncol(H), "\n")
cat("  Total gamma parameters:", length(taus) * ncol(H), "\n\n")

# =============================================================================
# PART 3: Cross-Validation for lambda (with data-adaptive penalties)
# =============================================================================

cat("=== PART 3: Cross-Validation for lambda_nc, lambda_lasso2_b, lambda_iq2_b ===\n\n")

cat("Using data-adaptive penalties with grid search:\n")
cat("  adaptive_gamma = TRUE (lambda_lasso2 ~ Gamma(1, b), b searched via CV)\n")
cat("  adaptive_iq = TRUE (lambda_iq2 ~ Gamma(1, b), b searched via CV)\n")
cat("  Tuning lambda_nc, lambda_lasso2_b, lambda_iq2_b via cross-validation\n\n")

# 3D grid: lambda_nc x lambda_lasso2_b x lambda_iq2_b
cv_grid <- expand.grid(
  lambda_nc = c(100),
  lambda_lasso2_b = c(0.5),
  lambda_iq2_b = c(0.1)
)

cat("Grid dimensions:", nrow(cv_grid), "combinations\n")
cat("  lambda_nc:", paste(unique(cv_grid$lambda_nc), collapse = ", "), "\n")
cat("  lambda_lasso2_b:", paste(unique(cv_grid$lambda_lasso2_b), collapse = ", "), "\n")
cat("  lambda_iq2_b:", paste(unique(cv_grid$lambda_iq2_b), collapse = ", "), "\n\n")

cv_result <- cv_copss_grid(
  y = y, taus = taus, H = H, X = NULL, w = w,
  grid = cv_grid,
  base_args = list(
    adaptive_gamma = TRUE,
    lambda_lasso2_a = 1,
    adaptive_iq = TRUE,
    lambda_iq2_a = 1,
    T_rel = 0.1,
    prior_gamma = "group_lasso",
    map_iter = 3000
  ),
  seed = 2024,
  verbose = TRUE
)

cat("\nCross-validation results (top 10):\n")
print(head(cv_result, 10))

# Get best hyperparameters
best_lambda_nc <- cv_result$lambda_nc[1]
best_lambda_lasso2_b <- cv_result$lambda_lasso2_b[1]
best_lambda_iq2_b <- cv_result$lambda_iq2_b[1]
cat("\nBest lambda_nc:", best_lambda_nc, "\n")
cat("Best lambda_lasso2_b:", best_lambda_lasso2_b,
    "(Gamma(1,", best_lambda_lasso2_b, ") -> mean lambda_lasso2 =",
    round(1 / best_lambda_lasso2_b, 2), ")\n")
cat("Best lambda_iq2_b:", best_lambda_iq2_b,
    "(Gamma(1,", best_lambda_iq2_b, ") -> mean lambda_iq2 =",
    round(1 / best_lambda_iq2_b, 2), ")\n\n")

# =============================================================================
# PART 4: Final Model Fitting with Best lambda_nc
# =============================================================================

cat("=== PART 4: Model Fitting with Best CV Hyperparameters ===\n\n")

cat("Fitting model with (all from CV):\n")
cat("  lambda_nc =", best_lambda_nc, "\n")
cat("  lambda_lasso2_b =", best_lambda_lasso2_b,
    "(Gamma(1,", best_lambda_lasso2_b, "))\n")
cat("  lambda_iq2_b =", best_lambda_iq2_b,
    "(Gamma(1,", best_lambda_iq2_b, "))\n\n")

fit_result <- getModel(
  y = y, taus = taus, H = H, X = NULL, w = w,
  lambda_nc = best_lambda_nc,
  adaptive_gamma = TRUE,
  lambda_lasso2_a = 1, lambda_lasso2_b = best_lambda_lasso2_b,
  adaptive_iq = TRUE,
  lambda_iq2_a = 1, lambda_iq2_b = best_lambda_iq2_b,
  T_rel = 0.1,
  prior_gamma = "group_lasso",
  fit_method = "map",
  map_iter = 3000,
  map_hessian = TRUE,
  laplace_n_samples = 1000,
  laplace_noise_scale = 0.1,
  seed = 2024,
  verbose = FALSE
)

cat("Model fitting complete.\n\n")
cat("Estimation method:", fit_result$fit_method, "\n")
cat("Number of parameters:", length(fit_result$map$par), "\n")

if (!is.null(fit_result$laplace_samples)) {
  cat("Laplacian samples available:\n")
  cat("  mu samples:", dim(fit_result$laplace_samples$mu)[1], "draws\n")
  if (!is.null(fit_result$laplace_samples$gamma)) {
    cat("  gamma samples:", dim(fit_result$laplace_samples$gamma)[1], "draws\n")
    cat("  gamma dimensions: (samples x quantiles x H columns) =",
        paste(dim(fit_result$laplace_samples$gamma), collapse = " x "), "\n")
  }
}
cat("\n")

# Compute predictive quantiles (eta) and QSS statistics
cat("Computing predictive quantiles and QSS statistics...\n")
eta <- getEta(fit_result, H = H)
qss <- getQSS(eta, taus = taus)
cat("  eta dimensions:", paste(dim(eta), collapse = " x "), "\n")
cat("  qss dimensions:", paste(dim(qss), collapse = " x "), "\n\n")

# =============================================================================
# PART 5: Decorrelation-Based Change-Point Detection
# =============================================================================

cat("=== PART 5: Decorrelation-Based Change-Point Detection ===\n\n")

cat("Key steps in decorrelation:\n")
cat("  1. Extract gamma posterior samples: gamma^(s) for s = 1, ..., S\n")
cat("  2. Compute posterior covariance: Sigma\n")
cat("  3. Compute whitening matrix: Sigma^{-1/2}\n")
cat("  4. Apply whitening: gamma_tilde^(s) = Sigma^{-1/2} gamma^(s)\n")
cat("  5. Compute test statistic: T_tilde_j^(s) = sum_q (gamma_tilde_{qj}^(s))^2\n")
cat("  6. Compute Bayesian p-value: p_j = (1/S) sum_s 1(T_tilde_j^(s) <= epsilon)\n\n")

# Run gamma-based change-point detection with decorrelation
gamma_result <- detectChangepoints_gamma(
  fit_result, taus = taus, l = l, w = w,
  signal_position = "first",
  y = y, eta = NULL,
  alpha = 0.05
)

cat("Decorrelation diagnostics:\n")
cat("  Posterior covariance matrix dimension:", nrow(gamma_result$Sigma), "x", ncol(gamma_result$Sigma), "\n")
cat("  Whitening matrix dimension:", nrow(gamma_result$Sigma_inv_sqrt), "x", ncol(gamma_result$Sigma_inv_sqrt), "\n")
cat("  Decorrelated gamma samples dimension:", paste(dim(gamma_result$gamma_tilde), collapse = " x "), "\n\n")

# Check decorrelation effectiveness
cat("Correlation structure before/after decorrelation:\n")

# Get original gamma samples
gamma_samples <- fit_result$laplace_samples$gamma
n_iter <- dim(gamma_samples)[1]
m <- length(taus)
r <- ncol(H)

# Vectorize original gamma
gamma_vec_orig <- matrix(NA, n_iter, m * r)
for (s in 1:n_iter) {
  gamma_vec_orig[s, ] <- as.vector(gamma_samples[s, , ])
}

# Vectorize decorrelated gamma
gamma_vec_decor <- matrix(NA, n_iter, m * r)
for (s in 1:n_iter) {
  gamma_vec_decor[s, ] <- as.vector(gamma_result$gamma_tilde[s, , ])
}

# Compute correlations (first 6 parameters for display)
cat("  Original gamma correlation (first 6 params):\n")
cor_orig <- cor(gamma_vec_orig[, 1:6])
print(round(cor_orig, 3))
cat("\n")

cat("  Decorrelated gamma correlation (first 6 params):\n")
cor_decor <- cor(gamma_vec_decor[, 1:6])
print(round(cor_decor, 3))
cat("\n")

cat("  Max off-diagonal correlation (original):", round(max(abs(cor_orig[lower.tri(cor_orig)])), 4), "\n")
cat("  Max off-diagonal correlation (decorrelated):", round(max(abs(cor_decor[lower.tri(cor_decor)])), 4), "\n\n")

# =============================================================================
# PART 6: Test Statistics Results
# =============================================================================

cat("=== PART 6: Test Statistics Results ===\n\n")

cat("Significance level (alpha):", gamma_result$alpha, "\n")
cat("Alternative:", gamma_result$alternative, "\n\n")

cat("Detection results (using decorrelated test statistic):\n")
cat("  Significant blocks (BH/FDR):", gamma_result$n_significant_bh, "\n")
cat("  Significant blocks (Holm):", gamma_result$n_significant_holm, "\n")
cat("  Significant blocks (Bonferroni):", gamma_result$n_significant_bonf, "\n\n")

cat("First signal detection (BH/FDR):\n")
cat("  BH/FDR:",
    ifelse(is.na(gamma_result$first_signal_bh), "None", gamma_result$first_signal_bh), "\n\n")

# Detection delay
if (!is.na(gamma_result$first_signal_bh)) {
  cat("Detection delay (BH):", gamma_result$first_signal_bh - shift_start, "observations\n\n")
}

# --- Consecutive block filter (same as simulation study) ---
filter_consecutive <- function(indices, min_consecutive) {
  if (min_consecutive <= 1 || length(indices) < min_consecutive) {
    return(if (min_consecutive <= 1) indices else integer(0))
  }
  sorted_idx <- sort(indices)
  groups <- cumsum(c(1, diff(sorted_idx) != 1))
  keep <- integer(0)
  for (g in unique(groups)) {
    run <- sorted_idx[groups == g]
    if (length(run) >= min_consecutive) {
      keep <- c(keep, run)
    }
  }
  return(keep)
}

min_consecutive <- 1
df_gamma <- gamma_result$detected_blocks

sig_raw  <- which(df_gamma$significant_raw)
sig_bh   <- which(df_gamma$significant_bh)
sig_holm <- which(df_gamma$significant_holm)
sig_bonf <- which(df_gamma$significant_bonf)

sig_raw_filt  <- filter_consecutive(sig_raw,  min_consecutive)
sig_bh_filt   <- filter_consecutive(sig_bh,   min_consecutive)
sig_holm_filt <- filter_consecutive(sig_holm, min_consecutive)
sig_bonf_filt <- filter_consecutive(sig_bonf, min_consecutive)

cat("=== Consecutive block filter (min_consecutive =", min_consecutive, ") ===\n\n")
cat("  Before filter -> After filter:\n")
cat("  Raw:        ", length(sig_raw),  "blocks ->", length(sig_raw_filt),  "blocks\n")
cat("  BH:         ", length(sig_bh),   "blocks ->", length(sig_bh_filt),   "blocks\n")
cat("  Holm:       ", length(sig_holm), "blocks ->", length(sig_holm_filt), "blocks\n")
cat("  Bonferroni: ", length(sig_bonf), "blocks ->", length(sig_bonf_filt), "blocks\n")
cat("  Raw blocks before filter:", paste(sig_raw, collapse = ", "), "\n")
cat("  BH blocks before filter:", paste(sig_bh, collapse = ", "), "\n\n")

# =============================================================================
# PART 7: Comparison - Original vs Decorrelated
# =============================================================================

cat("=== PART 7: Comparison - Original vs Decorrelated Test Statistics ===\n\n")

df <- gamma_result$detected_blocks

cat("Block-by-block comparison:\n")
cat(sprintf("%-10s %12s %12s %15s %15s\n",
            "Block", "Obs range", "p-value", "Sig (BH)", "Sig (Holm)"))

for (i in 1:nrow(df)) {
  cat(sprintf("%-10d %5d-%-5d %12.4f %15s %15s\n",
              df$h_col[i],
              df$obs_start[i], df$obs_end[i],
              df$pvalue_posterior[i],
              ifelse(df$significant_bh[i], "Yes", "No"),
              ifelse(df$significant_bh[i], "Yes", "No")))
}
cat("\n")

cat("Blocks containing the true shift:\n")
shift_blocks <- which(df$obs_start <= shift_start & df$obs_end >= shift_start |
                      df$obs_start >= shift_start)
shift_blocks <- shift_blocks[1:min(3, length(shift_blocks))]  # First 3 blocks after shift

for (b in shift_blocks) {
  cat(sprintf("  Block %d (obs %d-%d): p-value=%.4f, significant=%s\n",
              df$h_col[b], df$obs_start[b], df$obs_end[b],
              df$pvalue_posterior[b],
              ifelse(df$significant_bh[b], "Yes", "No")))
}
cat("\n")

# =============================================================================
# PART 7B: Debiased Estimation - Two-Stage Testing (Mitra & Zhang 2016)
# =============================================================================

cat("=== PART 7B: Debiased Estimation - Two-Stage Testing ===\n\n")

cat("Procedure:\n")
cat("  Stage 1: Block-level chi-squared test (Mitra & Zhang 2016) with BY correction\n")
cat("  Stage 2: Within-block per-quantile z-tests for significant blocks\n")
cat("  All based on the debiased (desparsified) gamma estimator\n\n")

# --- Step 0: Extract MAP quantities ---
gamma_samples <- fit_result$laplace_samples$gamma  # (S x m x r)
m <- length(taus)
r <- ncol(H)
n_obs <- nrow(H)

# Gamma MAP: mean of Laplace samples (centered at MAP)
gamma_MAP <- apply(gamma_samples, c(2, 3), mean)  # (m x r)

# mu0 MAP: baseline quantile level (length m)
# Since mu[q,t] = mu0[q] for all t, extract from mu[q,1]
mu_samples <- fit_result$laplace_samples$mu  # (S x m x n)
mu0_MAP <- apply(mu_samples[, , 1, drop = FALSE], 2, mean)  # length m

# beta MAP: intercept + covariate coefficients (m x p)
beta_samples <- fit_result$laplace_samples$beta  # (S x m x p)
beta_MAP <- apply(beta_samples, c(2, 3), mean)  # (m x p)

# Reconstruct X design matrix (intercept column, matching getModel)
# When X = NULL, getModel sets X <- matrix(1, n, 1) (all-1s intercept)
X_design <- matrix(1, nrow = n_obs, ncol = 1)

cat("MAP estimates extracted:\n")
cat("  gamma_MAP: (", m, "x", r, ") - quantiles x blocks\n")
cat("  beta_MAP:  (", m, "x", ncol(beta_MAP), ") - quantiles x predictors\n")
cat("  mu0_MAP:   length", m, "- baseline quantile levels\n")
cat("  H:         (", n_obs, "x", r, ") - observations x blocks\n\n")

# --- Step 1: Construct Theta_hat = (H'H/n)^{-1} ---
# Since r << n, direct inversion is stable (no node-wise lasso needed).
# In high-dimensional settings (r comparable to n), use hdi::lasso.proj() instead.
Sigma_H <- crossprod(H) / n_obs  # (r x r) Gram matrix
Theta_hat <- solve(Sigma_H)       # (r x r)
diag_Theta <- diag(Theta_hat)     # diagonal entries for variance formula

cat("Gram matrix Sigma_H = H'H/n:\n")
cat("  Condition number:", round(kappa(Sigma_H), 1), "\n")
cat("  Direct inversion (r =", r, "<< n =", n_obs, ")\n\n")

# --- Step 2: Compute residuals and estimate sparsity function f_q(0) ---
#
# Residuals: y - eta_MAP where eta_MAP = mu0[q] + X*beta[q,] + H*gamma[q,]
# Uses the baseline quantile level mu0[q] (constant across time).
#
residuals_mat <- matrix(NA, m, n_obs)
for (q in 1:m) {
  xb_q <- as.numeric(X_design %*% beta_MAP[q, ])
  hg_q <- as.vector(H %*% gamma_MAP[q, ])
  residuals_mat[q, ] <- y - mu0_MAP[q] - xb_q - hg_q
}

f_hat <- numeric(m)
for (q in 1:m) {
  dens <- density(residuals_mat[q, ], n = 1024)
  f_hat[q] <- approx(dens$x, dens$y, xout = 0, rule = 2)$y
}
# Floor to prevent division by near-zero (NA or tiny values break debiasing)
f_hat <- pmax(f_hat, 1e-4, na.rm = TRUE)
f_hat[is.na(f_hat)] <- 1e-4

cat("Estimated sparsity function f_q(0):\n")
for (q in 1:m) cat(sprintf("  tau = %.3f: f_hat(0) = %.4f\n", taus[q], f_hat[q]))
cat("\n")

# --- Step 3: Debias gamma per quantile (QR score) ---
# Belloni, Chernozhukov & Kato (2019): use CHECK FUNCTION SCORE
#   psi_tau(e) = tau - I(e < 0)
#
# gamma_d[q,] = gamma_MAP[q,] + (1/f_hat[q]) * Theta_hat %*% (H' psi_q / n)
gamma_debiased <- matrix(NA, m, r)

for (q in 1:m) {
  e_q <- residuals_mat[q, ]
  psi_q <- taus[q] - as.numeric(e_q < 0)
  correction <- (1 / f_hat[q]) *
    as.vector(Theta_hat %*% (crossprod(H, psi_q) / n_obs))
  gamma_debiased[q, ] <- gamma_MAP[q, ] + correction
}

cat("Debiasing effect (MAP -> Debiased) for all blocks:\n")
cat(sprintf("  %-15s", "Block (obs)"))
for (q in 1:m) cat(sprintf("  tau=%-6.3f", taus[q]))
cat("\n")
for (j in 1:r) {
  obs_s <- w + (j - 1) * l + 1
  obs_e <- min(w + j * l, n)
  cat(sprintf("  B%-2d (%3d-%-3d)", j, obs_s, obs_e))
  for (q in 1:m) {
    cat(sprintf(" %5.3f>%-5.3f", gamma_MAP[q, j], gamma_debiased[q, j]))
  }
  cat("\n")
}
cat("\n")

# --- Step 4: Cross-quantile covariance matrix C ---
# Cov(psi_tau1(e), psi_tau2(e)) = min(tau1, tau2) - tau1*tau2
# Scaling by sparsity functions:
# C[q1, q2] = (min(tau_q1, tau_q2) - tau_q1*tau_q2) / (f_q1(0) * f_q2(0))
C_mat <- matrix(NA, m, m)
for (q1 in 1:m) {
  for (q2 in 1:m) {
    C_mat[q1, q2] <- (min(taus[q1], taus[q2]) - taus[q1] * taus[q2]) /
                       (f_hat[q1] * f_hat[q2])
  }
}
C_inv <- solve(C_mat)

cat("Cross-quantile covariance matrix C (m x m):\n")
print(round(C_mat, 4))
cat("\n")

# --- Step 5: Per-element z-statistics (for Stage 2) ---
# Var(gamma_d[q,j]) = C[q,q] * Theta_jj / n
#                    = tau_q*(1-tau_q) / (n * f_q(0)^2) * Theta_jj
z_mat <- matrix(NA, m, r)
pval_element <- matrix(NA, m, r)

for (q in 1:m) {
  for (j in 1:r) {
    var_qj <- C_mat[q, q] * diag_Theta[j] / n_obs
    z_mat[q, j] <- gamma_debiased[q, j] / sqrt(var_qj)
    pval_element[q, j] <- 2 * (1 - pnorm(abs(z_mat[q, j])))
  }
}

# =============================================
# Stage 1: Block-level chi-squared test
# =============================================
cat("=== Stage 1: Block-Level Chi-Squared Test (Mitra & Zhang 2016) ===\n\n")
cat("T_j^2 = (n / Theta_jj) * gamma_d_j' C^{-1} gamma_d_j ~ chi^2_m\n\n")

# Test all r blocks.
chisq_stat <- numeric(r)
for (j in 1:r) {
  gamma_j <- gamma_debiased[, j]  # (m x 1) vector across quantiles
  chisq_stat[j] <- (n_obs / diag_Theta[j]) *
    as.numeric(t(gamma_j) %*% C_inv %*% gamma_j)
}

block_pval <- 1 - pchisq(chisq_stat, df = m)

# BY correction across all r blocks (valid under arbitrary dependence)
block_adjp <- p.adjust(block_pval, method = "BY")

sig_blocks_debiased <- which(block_adjp < 0.05)

cat(sprintf("  %-15s %12s %12s %12s %8s\n",
            "Block (obs)", "Chi-sq(5)", "Raw p-val", "BY-adj p", "Sig"))
for (j in 1:r) {
  obs_s <- w + (j - 1) * l + 1
  obs_e <- min(w + j * l, n)
  cat(sprintf("  B%-2d (%3d-%-3d) %12.4f %12.6f %12.6f %8s\n",
              j, obs_s, obs_e, chisq_stat[j], block_pval[j], block_adjp[j],
              ifelse(j %in% sig_blocks_debiased, "***", "")))
}
cat("\n")
cat("  Significant blocks (BY, alpha=0.05):", length(sig_blocks_debiased), "\n")
if (length(sig_blocks_debiased) > 0) {
  cat("  Block indices:", paste(sig_blocks_debiased, collapse = ", "), "\n")
}
cat("\n")

# =============================================
# Stage 2: Within-block per-quantile z-tests
# =============================================
cat("=== Stage 2: Within-Block Per-Quantile Z-Tests ===\n\n")

debiased_results <- list()

if (length(sig_blocks_debiased) > 0) {
  cat("Stage 1 established shift existence (FDR controlled at 0.05 via BY).\n")
  cat("Stage 2 characterizes the shift type - no additional correction needed.\n\n")

  for (j in sig_blocks_debiased) {
    obs_s <- w + (j - 1) * l + 1
    obs_e <- min(w + j * l, n)

    # Per-quantile significance (alpha=0.05, no correction needed)
    sig_q <- which(pval_element[, j] < 0.05)
    shift_direction <- sign(gamma_debiased[, j])

    # Classify shift type from pattern of significant quantiles
    if (length(sig_q) == 0) {
      shift_type <- "Undetermined"
    } else {
      sig_dirs <- shift_direction[sig_q]
      sig_same_sign <- all(sig_dirs == sig_dirs[1])
      tail_sig <- (1 %in% sig_q) & (m %in% sig_q)
      median_not_sig <- !(ceiling(m / 2) %in% sig_q)

      if (sig_same_sign & length(sig_q) >= 3) {
        shift_type <- "Location"
      } else if (tail_sig & median_not_sig) {
        shift_type <- "Scale"
      } else {
        shift_type <- "Shape/Skewness"
      }
    }

    debiased_results[[as.character(j)]] <- list(
      block = j, obs_start = obs_s, obs_end = obs_e,
      chisq = chisq_stat[j], block_pval = block_pval[j], block_adjp = block_adjp[j],
      gamma_MAP = gamma_MAP[, j], gamma_debiased = gamma_debiased[, j],
      z_scores = z_mat[, j], pval = pval_element[, j],
      shift_direction = shift_direction, shift_type = shift_type
    )

    cat(sprintf("  Block %d (obs %d-%d): %s shift\n", j, obs_s, obs_e, shift_type))
    cat(sprintf("    Chi-sq(%d) = %.2f, BY-adj p = %.6f\n\n", m, chisq_stat[j], block_adjp[j]))
    cat(sprintf("    %-12s %12s %12s %12s %12s %8s\n",
                "Quantile", "gamma_MAP", "gamma_d", "Z-score", "p-value", "Sig"))
    for (q in 1:m) {
      cat(sprintf("    tau=%-7.3f %12.4f %12.4f %12.4f %12.6f %8s\n",
                  taus[q], gamma_MAP[q, j], gamma_debiased[q, j],
                  z_mat[q, j], pval_element[q, j],
                  ifelse(pval_element[q, j] < 0.05, "*", "")))
    }
    cat("\n")
  }
} else {
  cat("  No significant blocks detected at Stage 1.\n\n")
}

# --- Comparison with Bayesian decorrelation approach (PART 5) ---
cat("Comparison: Debiased Two-Stage vs Bayesian Decorrelation (BH)\n")
cat(sprintf("  %-40s %10s %10s\n", "Method", "# Blocks", "Blocks"))
cat(sprintf("  %-40s %10d %10s\n", "Bayesian decorrelation (BH)",
            gamma_result$n_significant_bh,
            paste(which(gamma_result$detected_blocks$significant_bh), collapse = ",")))
cat(sprintf("  %-40s %10d %10s\n", "Debiased chi-sq + BY (this section)",
            length(sig_blocks_debiased),
            paste(sig_blocks_debiased, collapse = ",")))
cat("\n")

# =============================================================================
# PART 8: Visualization
# =============================================================================

cat("=== PART 8: Generating Visualizations ===\n\n")

# =============================================================================
# Main Results PDF
# =============================================================================
#pdf("inference_iq_v4_results.pdf", width = 14, height = 10)

par(mfrow = c(2, 2), mar = c(4, 4, 2.5, 1))

# Plot 1: Simulated data with quantile estimates
# eta = mu0[q] + X*beta[q,] + H*gamma[q,] (full linear predictor)
eta_linear <- matrix(NA, m, n)
for (q in 1:m) {
  eta_linear[q, ] <- mu0_MAP[q] + as.numeric(X_design %*% beta_MAP[q, ]) +
    as.vector(H %*% gamma_MAP[q, ])
}

# Get the significant block range from gamma_result
df_gamma <- gamma_result$detected_blocks
sig_blocks <- which(df_gamma$significant_bh)

# Find the observation with max deviation from predictive median within the significant block
# Sign-aware: only consider positive residuals if gamma > 0, negative if gamma < 0
signal_obs <- NA
if (length(sig_blocks) > 0) {
  # Get the first significant block
  first_sig_block <- sig_blocks[1]
  block_start <- df_gamma$obs_start[first_sig_block]
  block_end <- df_gamma$obs_end[first_sig_block]
  block_obs <- block_start:block_end

  # Get the gamma for this block and determine its sign
  # gamma_samples is (n_iter x m x r), take mean across iterations and quantiles
  gamma_block <- gamma_samples[, , first_sig_block]  # n_iter x m
  gamma_mean <- mean(gamma_block)  # overall mean gamma for this block
  gamma_sign <- sign(gamma_mean)

  # Compute residuals: y - predictive_median (not absolute)
  # Median quantile is tau = 0.5 (index 3 for quintuple quantiles)
  median_idx <- which.min(abs(taus - 0.5))
  residuals <- y[block_obs] - eta_linear[median_idx, block_obs]

  # Filter residuals based on gamma sign
  if (gamma_sign > 0) {
    # Positive shift: only consider positive residuals
    valid_idx <- which(residuals > 0)
  } else if (gamma_sign < 0) {
    # Negative shift: only consider negative residuals
    valid_idx <- which(residuals < 0)
  } else {
    # If gamma is exactly 0, use all residuals (fallback)
    valid_idx <- seq_along(residuals)
  }

  if (length(valid_idx) > 0) {
    # Find the observation with maximum absolute deviation among valid ones
    deviations <- abs(residuals[valid_idx])
    max_dev_idx <- valid_idx[which.max(deviations)]
    signal_obs <- block_obs[max_dev_idx]
  }

  cat("Signal selection info:\n")
  cat("  Block", first_sig_block, "- gamma mean:", round(gamma_mean, 4),
      "sign:", gamma_sign, "\n")
  cat("  Valid residuals:", length(valid_idx), "out of", length(block_obs), "\n")
  if (!is.na(signal_obs)) {
    cat("  Selected signal obs:", signal_obs, "with residual:",
        round(y[signal_obs] - eta_linear[median_idx, signal_obs], 4), "\n\n")
  }
}

# Set up point colors - all gray except signal point
point_colors <- rep("gray50", n)
if (!is.na(signal_obs)) {
  point_colors[signal_obs] <- "darkgreen"
}

plot(1:n, y, type = "p", col = point_colors, pch = 16, cex = 0.5,
     xlab = "Observation", ylab = "y",
     main = "Simulated Data with Quantile Estimates",
     ylim = range(c(y, eta_linear)))

# Highlight the signal point with larger size
if (!is.na(signal_obs)) {
  points(signal_obs, y[signal_obs], pch = 16, col = "darkgreen", cex = 1.5)
}

# Plot predictive quantiles with different colors
quantile_colors <- c("lightblue", "steelblue", "darkblue", "steelblue", "lightblue")
for (q in 1:length(taus)) {
  lines(1:n, eta_linear[q, ], col = quantile_colors[q], lwd = 1.5)
}
# Add true shift line
abline(v = shift_start, col = "red", lwd = 2, lty = 2)

legend("topleft",
       legend = c("Data", paste0("Q", taus), "True shift", "Signal (max dev)"),
       col = c("gray50", quantile_colors, "red", "darkgreen"),
       pch = c(16, rep(NA, length(taus)), NA, 16),
       lty = c(NA, rep(1, length(taus)), 2, NA),
       lwd = c(NA, rep(1.5, length(taus)), 2, NA),
       bty = "n", cex = 0.7)

# Plot 2: Block-level p-values (barplot)
bar_colors <- ifelse(df$significant_bh, "red", "gray70")
neg_log_p <- -log10(pmax(df$pvalue_posterior, 1e-16))  # clamp to avoid -log10(0) = Inf
bp <- barplot(neg_log_p, names.arg = df$h_col,
              col = bar_colors,
              xlab = "H Column (Block)", ylab = "-log10(p-value)",
              main = "Block-Level P-Values (BH/FDR Method)")
abline(h = -log10(0.05), col = "blue", lty = 2, lwd = 2)
# Mark true shift block
shift_block_idx <- which(df$obs_start <= shift_start & df$obs_end >= shift_start)
if (length(shift_block_idx) > 0) {
  abline(v = bp[shift_block_idx[1]], col = "purple", lty = 3, lwd = 2)
}
legend("topright",
       legend = c("Significant (BH)", "Not significant", "alpha = 0.05", "True shift"),
       fill = c("red", "gray70", NA, NA),
       border = c("black", "black", NA, NA),
       lty = c(NA, NA, 2, 3), col = c(NA, NA, "blue", "purple"),
       bty = "n", cex = 0.7)

# Plot 3: Box plot of decorrelated gamma L2 norm by block
# Uses whitened gamma (gamma_tilde) to remove posterior correlations between blocks
shift_block <- which(df$obs_start <= shift_start & df$obs_end >= shift_start)
if (length(shift_block) == 0) shift_block <- which(df$obs_start >= shift_start)[1]

box_colors <- rep("lightblue", r)
box_colors[shift_block] <- "salmon"
if (any(df$significant_bh)) {
  box_colors[df$significant_bh] <- "red"
}

# Compute L2 norm from DECORRELATED gamma (gamma_tilde), not raw gamma
# gamma_tilde has shape (n_iter x m x r), same as gamma_samples
gamma_tilde <- gamma_result$gamma_tilde
gamma_by_block <- lapply(1:r, function(j) {
  apply(gamma_tilde[, , j], 1, function(x) sqrt(sum(x^2)))
})
names(gamma_by_block) <- paste0("B", 1:r)

boxplot(gamma_by_block, col = box_colors,
        xlab = "Block", ylab = "Decorrelated Gamma L2 Norm",
        main = "Box Plot: Decorrelated Gamma by Block",
        outline = FALSE)
legend("topleft", legend = c("Non-shift block", "Shift block (detected)", "True shift block"),
       fill = c("lightblue", "red", "salmon"), bty = "n", cex = 0.8)

# Plot 4: Posterior p-values
barplot(1 - df$pvalue_posterior, names.arg = df$h_col,
        col = ifelse(df$significant_bh, "red", "gray70"),
        xlab = "H Column (Block)", ylab = "1 - Bayesian P-value",
        main = "Bayesian P-values (1 - p)",
        ylim = c(0, 1))
abline(h = 0.95, col = "blue", lty = 2, lwd = 2)
legend("topleft", legend = c("Significant (BH adj.)", "Threshold (0.05)"),
       fill = c("red", NA), border = c("black", NA),
       lty = c(NA, 2), col = c(NA, "blue"), bty = "n", cex = 0.8)

#dev.off()
#cat("Main results saved to: inference_iq_v4_results.pdf\n")

# =============================================================================
# QSS Statistics PDF (separate)
# =============================================================================
#pdf("inference_iq_v4_qss.pdf", width = 14, height = 10)

par(mfrow = c(2, 2), mar = c(4, 4, 2.5, 1))

# Compute QSS from MAP estimates (eta_linear) and 95% credible intervals from posterior
# MAP-based QSS (4 x n): Location, Scale, Skewness, Kurtosis
qss_map <- matrix(NA, 4, n)
rownames(qss_map) <- c("Location", "Scale", "Skewness", "Kurtosis")
qss_map[1, ] <- eta_linear[3, ]                                      # Median (tau = 0.5)
qss_map[2, ] <- eta_linear[4, ] - eta_linear[2, ]                    # IQR (Q75 - Q25)
iqr_map <- qss_map[2, ]
qss_map[3, ] <- ifelse(abs(iqr_map) > 1e-10,
  ((eta_linear[4, ] - eta_linear[3, ]) - (eta_linear[3, ] - eta_linear[2, ])) / iqr_map, 0)
qss_map[4, ] <- ifelse(abs(iqr_map) > 1e-10,
  (eta_linear[5, ] - eta_linear[1, ]) / iqr_map, NA)

# 95% credible intervals from posterior QSS samples
qss_lo <- apply(qss, c(2, 3), quantile, 0.025)  # 2.5th percentile
qss_hi <- apply(qss, c(2, 3), quantile, 0.975)  # 97.5th percentile

obs_idx <- (w+1):n

# Helper: plot QSS with 95% credible band
plot_qss_ci <- function(k, ylab_text, main_text, line_col, band_col, add_hline = FALSE) {
  ylim_range <- range(c(qss_lo[k, obs_idx], qss_hi[k, obs_idx]), na.rm = TRUE)
  plot(obs_idx, qss_map[k, obs_idx], type = "n",
       xlab = "Observation", ylab = ylab_text, main = main_text,
       ylim = ylim_range)
  polygon(c(obs_idx, rev(obs_idx)),
          c(qss_lo[k, obs_idx], rev(qss_hi[k, obs_idx])),
          col = band_col, border = NA)
  lines(obs_idx, qss_map[k, obs_idx], col = line_col, lwd = 1.5)
  abline(v = shift_start, col = "red", lwd = 2, lty = 2)
  if (add_hline) abline(h = 0, col = "gray70", lty = 3)
  legend("topleft",
         legend = c("MAP estimate", "95% CI", "True shift"),
         col = c(line_col, NA, "red"),
         fill = c(NA, band_col, NA),
         border = c(NA, NA, NA),
         lty = c(1, NA, 2), lwd = c(1.5, NA, 2),
         bty = "n", cex = 0.7)
}

# Plot 1: QSS - Location (Median)
plot_qss_ci(1, "Location (Median)", "QSS: Location",
             "blue", adjustcolor("blue", alpha.f = 0.2), add_hline = TRUE)

# Plot 2: QSS - Scale (IQR)
plot_qss_ci(2, "Scale (IQR)", "QSS: Scale",
             "darkgreen", adjustcolor("darkgreen", alpha.f = 0.2))

# Plot 3: QSS - Skewness (Bowley)
plot_qss_ci(3, "Skewness (Bowley)", "QSS: Skewness",
             "purple", adjustcolor("purple", alpha.f = 0.2), add_hline = TRUE)

# Plot 4: QSS - Kurtosis (Tail Ratio)
plot_qss_ci(4, "Kurtosis (Tail Ratio)", "QSS: Kurtosis",
             "orange", adjustcolor("orange", alpha.f = 0.2))

#dev.off()
#cat("QSS results saved to: inference_iq_v4_qss.pdf\n\n")

# =============================================================================
# Comparison PDF: BQQ vs cpt.meanvar
# =============================================================================
cat("Generating comparison plot: BQQ vs cpt.meanvar...\n")

# Load changepoint package
library(changepoint)

# Run cpt.meanvar with Asymptotic penalty
cpt_result <- cpt.meanvar(y, penalty = "Asymptotic", pen.value = 0.05, method = "PELT")
cpt_locations <- cpts(cpt_result)

cat("  cpt.meanvar detected change points:",
    ifelse(length(cpt_locations) == 0, "None", paste(cpt_locations, collapse = ", ")), "\n")

#pdf("inference_iq_v4_comparison.pdf", width = 14, height = 6)

par(mfrow = c(1, 2), mar = c(4, 4, 2.5, 1))

# Panel 1: BQQ Method
# Recompute signal_obs for clarity
df_gamma <- gamma_result$detected_blocks
sig_blocks <- which(df_gamma$significant_bh)

signal_obs_bqq <- NA
if (length(sig_blocks) > 0) {
  first_sig_block <- sig_blocks[1]
  block_start <- df_gamma$obs_start[first_sig_block]
  block_end <- df_gamma$obs_end[first_sig_block]
  block_obs <- block_start:block_end

  # Get gamma for this block to determine sign
  gamma_block <- gamma_samples[, , first_sig_block]
  gamma_mean <- mean(gamma_block)
  gamma_sign <- sign(gamma_mean)

  # Compute residuals (not absolute)
  median_idx <- which.min(abs(taus - 0.5))
  residuals <- y[block_obs] - eta_linear[median_idx, block_obs]

  # Filter based on gamma sign
  if (gamma_sign > 0) {
    valid_idx <- which(residuals > 0)
  } else if (gamma_sign < 0) {
    valid_idx <- which(residuals < 0)
  } else {
    valid_idx <- seq_along(residuals)
  }

  if (length(valid_idx) > 0) {
    deviations <- abs(residuals[valid_idx])
    max_dev_idx <- valid_idx[which.max(deviations)]
    signal_obs_bqq <- block_obs[max_dev_idx]
  }
}

# Set up point colors for BQQ plot
point_colors_bqq <- rep("gray50", n)
if (!is.na(signal_obs_bqq)) {
  point_colors_bqq[signal_obs_bqq] <- "darkgreen"
}

plot(1:n, y, type = "p", col = point_colors_bqq, pch = 16, cex = 0.5,
     xlab = "Observation", ylab = "y",
     main = "BQQ Method (Data-Adaptive Penalties, BH/FDR alpha = 0.05)",
     ylim = range(c(y, eta_linear)))

# Highlight the signal point
if (!is.na(signal_obs_bqq)) {
  points(signal_obs_bqq, y[signal_obs_bqq], pch = 16, col = "darkgreen", cex = 1.5)
}

# Plot predictive quantiles
quantile_colors <- c("lightblue", "steelblue", "darkblue", "steelblue", "lightblue")
for (q in 1:length(taus)) {
  lines(1:n, eta_linear[q, ], col = quantile_colors[q], lwd = 1.5)
}

# Add true shift line
abline(v = shift_start, col = "red", lwd = 2, lty = 2)

legend("topleft",
       legend = c("Data", paste0("Q", taus), "True shift", "Signal (max dev)"),
       col = c("gray50", quantile_colors, "red", "darkgreen"),
       pch = c(16, rep(NA, length(taus)), NA, 16),
       lty = c(NA, rep(1, length(taus)), 2, NA),
       lwd = c(NA, rep(1.5, length(taus)), 2, NA),
       bty = "n", cex = 0.7)

# Panel 2: cpt.meanvar Method
plot(1:n, y, type = "p", col = "gray50", pch = 16, cex = 0.5,
     xlab = "Observation", ylab = "y",
     main = "cpt.meanvar (Asymptotic, pen.value = 0.05)",
     ylim = range(y))

# Add detected change points from cpt.meanvar
if (length(cpt_locations) > 0) {
  for (cp in cpt_locations) {
    abline(v = cp, col = "blue", lwd = 2, lty = 1)
  }
}

# Add true shift line
abline(v = shift_start, col = "red", lwd = 2, lty = 2)

# Add segment means
seg_starts <- c(1, cpt_locations + 1)
seg_ends <- c(cpt_locations, n)
for (i in seq_along(seg_starts)) {
  seg_mean <- mean(y[seg_starts[i]:seg_ends[i]])
  segments(seg_starts[i], seg_mean, seg_ends[i], seg_mean, col = "blue", lwd = 2)
}

legend("topleft",
       legend = c("Data", "True shift", "Detected CP", "Segment mean"),
       col = c("gray50", "red", "blue", "blue"),
       pch = c(16, NA, NA, NA),
       lty = c(NA, 2, 1, 1),
       lwd = c(NA, 2, 2, 2),
       bty = "n", cex = 0.8)

#dev.off()
#cat("Comparison plot saved to: inference_iq_v4_comparison.pdf\n\n")

# =============================================================================
# Debiased Two-Stage Testing PDF
# =============================================================================
cat("Generating debiased two-stage testing visualization...\n")

#pdf("inference_iq_v4_debiased.pdf", width = 14, height = 10)

par(mfrow = c(2, 2), mar = c(4, 4, 2.5, 1))

# Panel 1: Block-level chi-squared statistics (-log10 p-value scale)
shift_block_idx <- which(df$obs_start <= shift_start & df$obs_end >= shift_start)
neg_log_block_p <- -log10(pmax(block_adjp, 1e-16))
neg_log_block_p[!is.finite(neg_log_block_p)] <- 0
bar_colors_chi <- ifelse(!is.na(block_adjp) & block_adjp < 0.05, "red", "gray70")
bp_chi <- barplot(neg_log_block_p, names.arg = 1:r,
                  col = bar_colors_chi,
                  xlab = "Block", ylab = "-log10(BY-adj p-value)",
                  main = "Stage 1: Block-Level Chi-Squared Test (Mitra & Zhang 2016)")
abline(h = -log10(0.05), col = "blue", lty = 2, lwd = 2)
if (length(shift_block_idx) > 0) {
  abline(v = bp_chi[shift_block_idx[1]], col = "purple", lty = 3, lwd = 2)
}
legend("topleft",
       legend = c("Significant (BY)", "Not significant", "alpha = 0.05", "True shift block"),
       fill = c("red", "gray70", NA, NA),
       border = c("black", "black", NA, NA),
       lty = c(NA, NA, 2, 3), col = c(NA, NA, "blue", "purple"),
       bty = "n", cex = 0.7)

# Panel 2: Z-score heatmap across quantiles and blocks
z_plot <- z_mat
z_lim <- max(abs(z_mat), na.rm = TRUE)
n_colors <- 50
blue_white_red <- colorRampPalette(c("steelblue", "white", "tomato"))(n_colors)
image(1:r, 1:m, t(z_plot), col = blue_white_red, zlim = c(-z_lim, z_lim),
      xlab = "Block", ylab = "Quantile",
      main = "Z-Scores from Debiased Gamma (per quantile x block)",
      axes = FALSE)
axis(1, at = 1:r, labels = 1:r)
axis(2, at = 1:m, labels = paste0("tau=", taus), las = 1)
box()
# Mark significant cells (per-element, alpha=0.05)
for (j in 1:r) {
  for (q in 1:m) {
    if (pval_element[q, j] < 0.05) {
      points(j, q, pch = 8, cex = 1.0, col = "black")
    }
  }
}
# Mark Stage 1 significant blocks
if (length(sig_blocks_debiased) > 0) {
  for (jj in sig_blocks_debiased) {
    rect(jj - 0.5, 0.5, jj + 0.5, m + 0.5, border = "gold", lwd = 2)
  }
}
if (length(shift_block_idx) > 0) {
  abline(v = shift_block_idx[1], col = "purple", lty = 3, lwd = 2)
}
legend("topleft",
       legend = c("Sig. element (p<0.05)", "Sig. block (Stage 1)", "True shift block"),
       pch = c(8, NA, NA), lty = c(NA, 1, 3),
       col = c("black", "gold", "purple"), lwd = c(NA, 2, 2),
       bty = "n", cex = 0.65)

# Panel 3: MAP vs Debiased gamma for significant blocks (or last 3 blocks)
show_blocks <- if (length(sig_blocks_debiased) > 0) sig_blocks_debiased else max(1, r-2):r
n_show <- length(show_blocks)
# Grouped barplot: MAP and Debiased side by side per quantile
plot_data <- matrix(NA, nrow = 2 * m, ncol = n_show)
colnames(plot_data) <- paste0("B", show_blocks)
row_labels <- rep(paste0("tau=", taus), each = 2)
for (idx in seq_along(show_blocks)) {
  j <- show_blocks[idx]
  for (q in 1:m) {
    plot_data[2 * (q - 1) + 1, idx] <- gamma_MAP[q, j]
    plot_data[2 * (q - 1) + 2, idx] <- gamma_debiased[q, j]
  }
}
pair_colors <- rep(c("gray70", "tomato"), m)
ylim_gam <- range(c(plot_data[is.finite(plot_data)], 0)) * 1.2
if (diff(ylim_gam) < 1e-10) ylim_gam <- c(-0.5, 0.5)
bp_gam <- barplot(plot_data, beside = TRUE, col = pair_colors,
                  xlab = "Block", ylab = "Gamma coefficient",
                  main = "MAP vs Debiased Gamma (Significant Blocks)",
                  ylim = ylim_gam)
abline(h = 0, col = "black", lty = 1, lwd = 0.5)
legend("topleft",
       legend = c("MAP (shrunk)", "Debiased"),
       fill = c("gray70", "tomato"),
       bty = "n", cex = 0.7)

# Panel 4: Stage 2 - per-quantile z-scores for first significant block
if (length(sig_blocks_debiased) > 0) {
  j <- sig_blocks_debiased[1]
  res <- debiased_results[[as.character(j)]]
  z_vals <- res$z_scores
  z_cols <- ifelse(res$pval < 0.05 & z_vals > 0, "tomato",
                   ifelse(res$pval < 0.05 & z_vals < 0, "steelblue", "gray70"))
  bp_z <- barplot(z_vals, names.arg = paste0("tau=", taus),
                  col = z_cols,
                  xlab = "Quantile", ylab = "Z-score",
                  main = paste0("Stage 2: Block ", j, " (",
                                res$shift_type, " shift)"))
  abline(h = c(-1.96, 1.96), col = "blue", lty = 2, lwd = 1.5)
  abline(h = 0, col = "black", lty = 1, lwd = 0.5)
  legend("topleft",
         legend = c("Sig. positive", "Sig. negative", "Not sig.", "+/- 1.96"),
         fill = c("tomato", "steelblue", "gray70", NA),
         border = c("black", "black", "black", NA),
         lty = c(NA, NA, NA, 2), col = c(NA, NA, NA, "blue"),
         bty = "n", cex = 0.7)
} else {
  plot.new()
  text(0.5, 0.5, "No significant blocks detected\nat Stage 1",
       cex = 1.5, col = "gray50")
}

#dev.off()
#cat("Debiased testing results saved to: inference_iq_v4_debiased.pdf\n\n")

# =============================================================================
# PART 9: Summary
# =============================================================================

cat("=== PART 9: Summary ===\n\n")

cat("Data-Adaptive Penalties Summary:\n")
cat("--------------------------------\n")
cat("True shift point:", shift_start, "\n")
cat("Warm-up period:", w, "\n")
cat("Block length:", l, "\n")
cat("Number of H columns (gamma per quantile):", r, "\n")
cat("Number of quantiles:", m, "\n")
cat("Total gamma parameters:", m * r, "\n\n")

cat("Penalty Settings (all from CV):\n")
cat("  adaptive_gamma = TRUE, lambda_lasso2 ~ Gamma(1,", best_lambda_lasso2_b, ")\n")
cat("  adaptive_iq = TRUE, lambda_iq2 ~ Gamma(1,", best_lambda_iq2_b, ")\n")
cat("  lambda_nc =", best_lambda_nc, "\n\n")

cat("Cross-Validation Results:\n")
cat("  Grid: lambda_nc x lambda_lasso2_b x lambda_iq2_b =",
    nrow(cv_grid), "combinations\n")
cat("  Best lambda_nc:", best_lambda_nc, "\n")
cat("  Best lambda_lasso2_b:", best_lambda_lasso2_b, "\n")
cat("  Best lambda_iq2_b:", best_lambda_iq2_b, "\n")
cat("  Best validation loss:", round(cv_result$val_loss[1], 4), "\n\n")

cat("Detection Performance:\n")
cat("  [Bayesian decorrelation (BH)]:\n")
cat("    Significant blocks:", gamma_result$n_significant_bh, "\n")
cat("    Signal observation (max deviation):",
    ifelse(is.na(signal_obs_bqq), "None", signal_obs_bqq), "\n")
if (!is.na(signal_obs_bqq)) {
  cat("    Detection delay:", signal_obs_bqq - shift_start, "observations\n")
}
cat("  [Debiased chi-squared + BY (Mitra & Zhang 2016)]:\n")
cat("    Significant blocks:", length(sig_blocks_debiased), "\n")
if (length(sig_blocks_debiased) > 0) {
  cat("    Block indices:", paste(sig_blocks_debiased, collapse = ", "), "\n")
  for (j_str in names(debiased_results)) {
    res <- debiased_results[[j_str]]
    cat(sprintf("    Block %d: %s shift (Chi-sq=%.2f, BY-adj p=%.4f)\n",
                res$block, res$shift_type, res$chisq, res$block_adjp))
  }
}
cat("\n")

# =============================================================================
# Sensitivity and Specificity Calculation
# =============================================================================
# Using symmetric window for both BQQ and cpt.meanvar:
# - Tolerance = block length (l)
# - TP: detection within [shift_start - tolerance, shift_start + tolerance]
# - FP: detection outside this window

tolerance <- l  # use block length as tolerance

# For BQQ: use signal_obs_bqq as the detected change point
if (!is.na(signal_obs_bqq)) {
  # True Positive: signal observation is within tolerance of true shift
  TP_bqq <- abs(signal_obs_bqq - shift_start) <= tolerance
  # False Positive: signal observation is outside tolerance
  FP_bqq <- !TP_bqq

  sensitivity_bqq <- if (TP_bqq) 1.0 else 0.0
  specificity_bqq <- if (TP_bqq) 1.0 else 0.0  # No FP if detection is correct
} else {
  # No detection
  TP_bqq <- FALSE
  FP_bqq <- FALSE
  sensitivity_bqq <- 0.0
  specificity_bqq <- 1.0  # No false alarms if no detections
}

cat("Sensitivity and Specificity (BQQ Method):\n")
cat("  Signal observation:", ifelse(is.na(signal_obs_bqq), "None", signal_obs_bqq), "\n")
cat("  True shift:", shift_start, "\n")
cat("  Tolerance (block length):", tolerance, "\n")
cat("  Window: [", shift_start - tolerance, ",", shift_start + tolerance, "]\n")
cat("  True Positive (detection within tolerance):", TP_bqq, "\n")
cat("  False Positive (detection outside tolerance):", FP_bqq, "\n")
cat("  Sensitivity:", round(sensitivity_bqq, 4), "\n")
cat("  Specificity:", round(specificity_bqq, 4), "\n\n")

# For cpt.meanvar: consider a detection "correct" if within block_length of true shift
tolerance <- l  # use block length as tolerance
if (length(cpt_locations) > 0) {
  # True Positive: at least one detected CP is within tolerance of true shift
  TP_cpt <- any(abs(cpt_locations - shift_start) <= tolerance)
  # False Positives: detected CPs that are NOT within tolerance of true shift
  FP_cpt <- sum(abs(cpt_locations - shift_start) > tolerance)

  sensitivity_cpt <- if (TP_cpt) 1.0 else 0.0
  # For specificity, we use: 1 - (FP / total detections) as a proxy
  specificity_cpt <- if (length(cpt_locations) > 0) 1 - FP_cpt / length(cpt_locations) else 1.0
} else {
  sensitivity_cpt <- 0.0
  specificity_cpt <- 1.0  # No false alarms if no detections
  TP_cpt <- FALSE
  FP_cpt <- 0
}

cat("Sensitivity and Specificity (cpt.meanvar):\n")
cat("  Detected change points:", ifelse(length(cpt_locations) == 0, "None", paste(cpt_locations, collapse = ", ")), "\n")
cat("  True shift:", shift_start, "\n")
cat("  Tolerance (block length):", tolerance, "\n")
cat("  True Positive (detection within tolerance):", TP_cpt, "\n")
cat("  False Positives (detections outside tolerance):", FP_cpt, "\n")
cat("  Sensitivity:", round(sensitivity_cpt, 4), "\n")
cat("  Specificity:", round(specificity_cpt, 4), "\n\n")

cat("Decorrelation Effectiveness:\n")
cat("  Max correlation before whitening:", round(max(abs(cor_orig[lower.tri(cor_orig)])), 4), "\n")
cat("  Max correlation after whitening:", round(max(abs(cor_decor[lower.tri(cor_decor)])), 4), "\n")

cat("\n================================================================\n")
cat("Data-Adaptive Penalties Example Complete!\n")
cat("================================================================\n")
