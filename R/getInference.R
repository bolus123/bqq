# Statistical Inference Functions for Interquantile Shrinkage Model
#
# Implements the Bayesian Predictive Quantile-Based Charting approach
# as described in the BQQ methodology.
#
# Two types of inference:
# 1. Predictive quantile-based inference (chi-squared tests on quantile vectors)
# 2. Predictive distributional statistics-based inference (location, scale, skew, kurtosis)


# =============================================================================
# Laplacian Approximation for MAP Inference
# =============================================================================

#' Generate posterior samples using Laplacian approximation from MAP fit
#'
#' When only MAP estimation is available, this function generates approximate
#' posterior samples. Since Stan's Hessian is in unconstrained space with different
#' dimensions, we use a parametric bootstrap approach:
#' 1. Extract MAP estimates for mu, beta, gamma
#' 2. Add uncertainty based on estimated residual variance and parameter structure
#'
#' @param map_fit MAP optimization result from getModel (the $map element)
#' @param hessian Hessian matrix at MAP estimate (optional, currently not used)
#' @param n_samples Number of posterior samples to generate (default: 1000)
#' @param noise_scale Scale factor for parameter perturbation (default: 0.1)
#' @param seed Random seed for reproducibility
#' @return List with parameter samples in the same structure as rstan::extract()
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 100
#' y <- rnorm(n)
#' taus <- c(0.25, 0.5, 0.75)
#' H <- getIsolatedShift(n, l = 20, w = 20)
#' fit <- getModel(y, taus, H = H, w = 20, fit_method = "map",
#'                 map_hessian = FALSE, map_iter = 500)
#' samples <- getLaplaceSamples(fit$map, n_samples = 100)
#' }
#' @export
getLaplaceSamples <- function(map_fit, hessian = NULL, n_samples = 1000,
                              noise_scale = 0.1, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  # Extract MAP estimates
  par_map <- map_fit$par
  par_names <- names(par_map)
  n_par <- length(par_map)

  # Extract indices for unconstrained parameters
  mu_idx <- grep("^mu\\[", par_names)
  beta_idx <- grep("^beta\\[", par_names)
  gamma_idx <- grep("^gamma\\[", par_names)
  tau_rw_idx <- grep("^tau_rw\\[", par_names)
  unc_idx <- c(mu_idx, beta_idx, gamma_idx)

  # Helper: parse 2D indices from names like "gamma[1,2]"
  parse_2d <- function(idx, prefix) {
    dims_str <- gsub(paste0(prefix, "\\[|\\]"), "", par_names[idx])
    dims_split <- strsplit(dims_str, ",")
    list(row = as.integer(sapply(dims_split, `[`, 1)),
         col = as.integer(sapply(dims_split, `[`, 2)))
  }

  # Helper: scatter samples into 3D array
  scatter <- function(samp_mat, parsed, n_samples) {
    nr <- max(parsed$row); nc <- max(parsed$col)
    arr <- array(NA, dim = c(n_samples, nr, nc))
    for (i in seq_along(parsed$row)) arr[, parsed$row[i], parsed$col[i]] <- samp_mat[, i]
    arr
  }

  mu_parsed <- if (length(mu_idx) > 0) parse_2d(mu_idx, "mu") else NULL
  beta_parsed <- if (length(beta_idx) > 0) parse_2d(beta_idx, "beta") else NULL
  gamma_parsed <- if (length(gamma_idx) > 0) parse_2d(gamma_idx, "gamma") else NULL

  # Try proper Laplace approximation using Hessian
  samples_unc <- NULL
  if (!is.null(hessian) && nrow(hessian) == n_par && length(unc_idx) > 0) {
    samples_unc <- tryCatch({
      H_neg <- -(hessian + t(hessian)) / 2
      H_neg_reg <- H_neg + diag(1e-6, n_par)
      Sigma_full <- solve(H_neg_reg)
      Sigma_unc <- Sigma_full[unc_idx, unc_idx]
      eig <- eigen(Sigma_unc, symmetric = TRUE)
      eig$values <- pmax(eig$values, 1e-8)
      L <- t(eig$vectors %*% diag(sqrt(eig$values)))
      theta_map_unc <- par_map[unc_idx]
      z_mat <- matrix(rnorm(n_samples * length(unc_idx)), n_samples, length(unc_idx))
      sweep(z_mat %*% L, 2, theta_map_unc, "+")
    }, error = function(e) {
      warning("Hessian-based Laplace failed: ", conditionMessage(e),
              ". Falling back to heuristic noise.")
      NULL
    })
  }

  mu_unc_cols <- seq_along(mu_idx)
  beta_unc_cols <- length(mu_idx) + seq_along(beta_idx)
  gamma_unc_cols <- length(mu_idx) + length(beta_idx) + seq_along(gamma_idx)

  if (!is.null(samples_unc)) {
    mu_array <- if (length(mu_idx) > 0) scatter(samples_unc[, mu_unc_cols, drop = FALSE], mu_parsed, n_samples)
    beta_array <- if (length(beta_idx) > 0) scatter(samples_unc[, beta_unc_cols, drop = FALSE], beta_parsed, n_samples)
    gamma_array <- if (length(gamma_idx) > 0) scatter(samples_unc[, gamma_unc_cols, drop = FALSE], gamma_parsed, n_samples)
  } else {
    # Fallback: heuristic noise
    mu_array <- if (length(mu_idx) > 0) {
      m <- max(mu_parsed$row); n <- max(mu_parsed$col)
      mu_map <- matrix(NA, m, n)
      for (i in seq_along(mu_idx)) mu_map[mu_parsed$row[i], mu_parsed$col[i]] <- par_map[mu_idx[i]]
      mu_sd <- matrix(NA, m, n)
      for (q in 1:m) {
        d_sd <- sd(diff(mu_map[q, ]), na.rm = TRUE)
        if (is.na(d_sd) || d_sd < 1e-6) d_sd <- 0.1
        mu_sd[q, ] <- d_sd * noise_scale
      }
      arr <- array(NA, dim = c(n_samples, m, n))
      for (s in 1:n_samples) arr[s, , ] <- mu_map + matrix(rnorm(m * n, 0, mu_sd), m, n)
      arr
    }
    beta_array <- if (length(beta_idx) > 0) {
      mb <- max(beta_parsed$row); pb <- max(beta_parsed$col)
      beta_map <- matrix(NA, mb, pb)
      for (i in seq_along(beta_idx)) beta_map[beta_parsed$row[i], beta_parsed$col[i]] <- par_map[beta_idx[i]]
      beta_sd <- pmax(abs(beta_map) * noise_scale, 0.05)
      arr <- array(NA, dim = c(n_samples, mb, pb))
      for (s in 1:n_samples) arr[s, , ] <- beta_map + matrix(rnorm(mb * pb, 0, beta_sd), mb, pb)
      arr
    }
    gamma_array <- if (length(gamma_idx) > 0) {
      mg <- max(gamma_parsed$row); rg <- max(gamma_parsed$col)
      gamma_map <- matrix(NA, mg, rg)
      for (i in seq_along(gamma_idx)) gamma_map[gamma_parsed$row[i], gamma_parsed$col[i]] <- par_map[gamma_idx[i]]
      gamma_sd <- pmax(abs(gamma_map) * noise_scale, 0.02)
      arr <- array(NA, dim = c(n_samples, mg, rg))
      for (s in 1:n_samples) arr[s, , ] <- gamma_map + matrix(rnorm(mg * rg, 0, gamma_sd), mg, rg)
      arr
    }
  }

  # tau_rw (always heuristic — bounded parameter, Hessian is in log space)
  tau_rw_mat <- if (length(tau_rw_idx) > 0) {
    tau_rw_map <- par_map[tau_rw_idx]
    tau_rw_sd <- pmax(abs(tau_rw_map) * noise_scale, 0.01)
    mat <- matrix(NA, n_samples, length(tau_rw_idx))
    for (s in 1:n_samples) mat[s, ] <- pmax(tau_rw_map + rnorm(length(tau_rw_idx), 0, tau_rw_sd), 1e-6)
    mat
  }

  list(mu = mu_array, beta = beta_array, gamma = gamma_array, tau_rw = tau_rw_mat)
}


# =============================================================================
# Extract Predictive Quantiles
# =============================================================================

#' Extract eta (fitted quantiles) from posterior samples or Laplace approximation
#'
#' Automatically detects the fit_method used and extracts posterior samples accordingly:
#' - "mcmc" or "map_mcmc": Uses MCMC posterior draws from stanfit object
#' - "map": Uses pre-generated Laplacian approximation samples
#'
#' @param fit_result Full result from getModel()
#' @param H Design matrix for gamma coefficients
#' @param X Design matrix for beta coefficients (including intercept)
#' @param offset Optional numeric vector of length n added to the linear predictor.
#' @param n_samples Number of samples for Laplace approximation (only used if fit_result
#'   doesn't have laplace_samples and needs to generate them)
#' @param seed Random seed (only used if generating new Laplace samples)
#' @return 3D array [iterations, quantiles, time]
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 100
#' y <- rnorm(n)
#' taus <- c(0.25, 0.5, 0.75)
#' H <- getIsolatedShift(n, l = 20, w = 20)
#' fit <- getModel(y, taus, H = H, w = 20, fit_method = "map",
#'                 map_hessian = FALSE, map_iter = 500)
#' eta <- getEta(fit, H = H)
#' }
#' @export
getEta <- function(fit_result, H, X = NULL, offset = NULL, n_samples = 1000, seed = NULL) {

  n <- nrow(H)
  r <- ncol(H)

  # X should include intercept as first column
  if (is.null(X)) {
    X <- matrix(1, n, 1)
  }
  p <- ncol(X)

  # Offset defaults to zero
  if (is.null(offset)) {
    offset <- rep(0, n)
  }

  # Determine fit method (backward compatible)
  fit_method <- fit_result$fit_method
  if (is.null(fit_method)) {
    # Infer from available data for backward compatibility
    if (!is.null(fit_result$fit) && inherits(fit_result$fit, "stanfit")) {
      fit_method <- "mcmc"
    } else if (!is.null(fit_result$laplace_samples)) {
      fit_method <- "map"
    } else if (!is.null(fit_result$map)) {
      fit_method <- "map"
    } else {
      stop("Cannot determine fit_method from fit_result")
    }
  }

  # Extract posterior draws based on fit_method
  if (fit_method %in% c("mcmc", "map_mcmc")) {
    # Use MCMC samples
    if (is.null(fit_result$fit) || !inherits(fit_result$fit, "stanfit")) {
      stop("fit_method is '", fit_method, "' but no stanfit object found")
    }

    draws <- rstan::extract(fit_result$fit)
    n_iter <- dim(draws$mu)[1]
    m <- dim(draws$mu)[2]

    # Handle dimension collapse for beta
    beta_draws <- draws$beta
    if (length(dim(beta_draws)) == 2) {
      beta_draws <- array(beta_draws, dim = c(n_iter, m, 1))
    }

    # Handle dimension collapse for gamma
    gamma_draws <- NULL
    if (r > 0) {
      gamma_draws <- draws$gamma
      if (length(dim(gamma_draws)) == 2) {
        gamma_draws <- array(gamma_draws, dim = c(n_iter, m, 1))
      }
    }

    mu_draws <- draws$mu

  } else if (fit_method == "map") {
    # Use Laplacian approximation samples
    if (!is.null(fit_result$laplace_samples)) {
      # Use pre-generated samples from getModel
      laplace_samples <- fit_result$laplace_samples
    } else if (!is.null(fit_result$map)) {
      # Generate samples on the fly (backward compatibility)
      laplace_samples <- getLaplaceSamples(fit_result$map, fit_result$hessian,
                                           n_samples = n_samples, seed = seed)
    } else {
      stop("fit_method is 'map' but no MAP estimates or laplace_samples found")
    }

    n_iter <- dim(laplace_samples$mu)[1]
    m <- dim(laplace_samples$mu)[2]

    mu_draws <- laplace_samples$mu
    beta_draws <- laplace_samples$beta
    gamma_draws <- laplace_samples$gamma

  } else {
    stop("Unknown fit_method: ", fit_method)
  }

  # Initialize eta array
  eta <- array(NA, dim = c(n_iter, m, n))

  for (s in 1:n_iter) {
    mu_s <- mu_draws[s, , ]        # m x n

    # Handle beta
    if (length(dim(beta_draws)) == 3) {
      beta_s <- beta_draws[s, , , drop = FALSE]
      beta_s <- matrix(beta_s, nrow = m, ncol = p)
    } else {
      beta_s <- matrix(beta_draws[s, ], nrow = m, ncol = p)
    }

    for (q in 1:m) {
      xb <- as.numeric(X %*% beta_s[q, ])
      hg <- 0
      if (r > 0 && !is.null(gamma_draws)) {
        if (length(dim(gamma_draws)) == 3) {
          gamma_s <- gamma_draws[s, , , drop = FALSE]
          gamma_s <- matrix(gamma_s, nrow = m, ncol = r)
        } else {
          gamma_s <- matrix(gamma_draws[s, ], nrow = m, ncol = r)
        }
        hg <- as.numeric(H %*% gamma_s[q, ])
      }
      # mu_s[q, 1] = mu0[q] since mu[q,t] = mu0[q] for all t
      eta[s, q, ] <- mu_s[q, 1] + xb + hg + offset
    }
  }

  eta
}


# =============================================================================
# Chi-Squared Statistic (BQQ Approach)
# =============================================================================

#' Compute BQQ chi-squared statistic for quantile vectors
#'
#' Implements the chi-squared-inspired charting statistic from the BQQ methodology:
#' \eqn{W_t = Z_t' S^{-1} Z_t}
#' where Z_t = mean(Z_tr) across posterior draws r.
#'
#' The covariance S is estimated from the warm-up period to reflect the natural
#' variability under the null hypothesis (no change point). This captures both:
#' 1. Posterior uncertainty in parameter estimates
#' 2. Natural temporal variability from the random walk component
#'
#' @param eta 3D array [iterations, quantiles, time] from getEta()
#' @param w Warm-up period to exclude from testing and use for null calibration
#' @param use_differencing If TRUE, use first-order differencing (for sustained shifts)
#' @param df_method Method for degrees of freedom: "full" (m) or "reduced" (m-1)
#' @param p_method Method for p-values: "chisq", "bootstrap", or "empirical_null"
#' @param n_bootstrap Number of bootstrap replicates (if p_method = "bootstrap")
#' @param alpha Significance level for multiple testing adjustment
#' @param adjust_method P-value adjustment method (e.g., "holm", "BH", "bonferroni")
#' @param signal_position Method to determine signal position within a run of consecutive signals:
#'   - "first": First observation in the run (default)
#'   - "last": Last observation in the run
#'   - "middle": Middle observation in the run
#'   - "max_deviation": Observation with maximum deviation from the predictive median (fitted eta at tau = 0.5)
#' @param y Original data (required only for signal_position = "max_deviation")
#' @param taus Quantile levels (required only for signal_position = "max_deviation")
#' @return List with chi-squared statistics, p-values, and diagnostic info
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 100
#' y <- rnorm(n)
#' taus <- c(0.25, 0.5, 0.75)
#' H <- getIsolatedShift(n, l = 20, w = 20)
#' fit <- getModel(y, taus, H = H, w = 20, fit_method = "map",
#'                 map_hessian = FALSE, map_iter = 500)
#' eta <- getEta(fit, H = H)
#' result <- getChisq_BQQ(eta, w = 20)
#' }
#' @export
getChisq_BQQ <- function(eta, w = 0, use_differencing = FALSE,
                         df_method = "reduced", p_method = "empirical_null",
                         n_bootstrap = 1000, alpha = 0.05,
                         adjust_method = "holm",
                         signal_position = c("first", "last", "middle", "max_deviation"),
                         y = NULL, taus = NULL) {

  # Validate signal_position argument
  signal_position <- match.arg(signal_position)

  n_iter <- dim(eta)[1]
  m <- dim(eta)[2]      # number of quantiles
  n <- dim(eta)[3]      # number of time points

  # Apply first-order differencing if requested
  if (use_differencing) {
    eta_diff <- array(NA, dim = c(n_iter, m, n - 1))
    for (s in 1:n_iter) {
      for (q in 1:m) {
        eta_diff[s, q, ] <- diff(eta[s, q, ])
      }
    }
    eta <- eta_diff
    n <- n - 1
    # Adjust warm-up for differencing
    if (w > 0) w <- w - 1
  }

  # Degrees of freedom
  df <- if (df_method == "reduced") m - 1 else m

  # Compute Z_t (mean across posterior draws) for each time t
  Z_t <- matrix(NA, m, n)
  for (t in 1:n) {
    Z_t[, t] <- colMeans(eta[, , t])
  }

  # Compute in-control reference from warm-up period
  if (w > 0) {
    Z_ref <- rowMeans(Z_t[, 1:w, drop = FALSE])

    # Key improvement: Estimate the BETWEEN-TIME covariance from warm-up period
    # This captures the natural temporal variability, not just posterior uncertainty
    # S = Cov(Z_t) across time points in warm-up period
    Z_warmup <- Z_t[, 1:w, drop = FALSE]  # m x w matrix
    S_between <- cov(t(Z_warmup))  # Covariance across the w time points

    # Also compute pooled within-time covariance (posterior uncertainty)
    S_within <- matrix(0, m, m)
    for (t in 1:w) {
      S_within <- S_within + cov(eta[, , t])
    }
    S_within <- S_within / w

    # Total covariance: between-time variability + posterior uncertainty
    # Use between-time as primary (captures random walk), add small posterior uncertainty
    S_ref <- S_between + 0.1 * S_within

  } else {
    # Fallback: use overall statistics
    Z_ref <- rowMeans(Z_t)
    S_ref <- cov(t(Z_t))
  }

  # Add ridge for numerical stability
  S_ref_ridge <- S_ref + diag(1e-6, m)
  S_ref_inv <- tryCatch({
    solve(S_ref_ridge)
  }, error = function(e) {
    MASS::ginv(S_ref_ridge)
  })

  # Compute chi-squared statistic for each time point
  chisq <- numeric(n)
  for (t in 1:n) {
    d <- Z_t[, t] - Z_ref
    chisq[t] <- as.numeric(t(d) %*% S_ref_inv %*% d)
  }

  # P-values based on chosen method
  if (p_method == "chisq") {
    # Theoretical chi-squared distribution
    pvalue <- 1 - pchisq(chisq, df = df)

  } else if (p_method == "empirical_null") {
    # Use warm-up period to estimate the null distribution empirically
    if (w < 5) {
      warning("Empirical null requires warm-up >= 5, using chi-squared")
      pvalue <- 1 - pchisq(chisq, df = df)
    } else {
      # Compute chi-squared values for warm-up period (these are our null samples)
      chisq_null <- chisq[1:w]

      # Fit the null distribution (scale the chi-squared)
      # Under null, W ~ c * chi-squared(df) where c is a scaling factor
      # Estimate c from warm-up: E[W] = c * df, so c = mean(W) / df
      c_scale <- max(mean(chisq_null) / df, 0.1)

      # P-values using scaled chi-squared
      pvalue <- 1 - pchisq(chisq / c_scale, df = df)
    }

  } else if (p_method == "bootstrap") {
    # Bootstrap null distribution from warm-up
    if (w < 5) {
      warning("Bootstrap requires warm-up >= 5, using chi-squared")
      pvalue <- 1 - pchisq(chisq, df = df)
    } else {
      # Compute null chi-squared values from warm-up
      chisq_null <- chisq[1:w]

      # Empirical p-values
      pvalue <- numeric(n)
      for (t in 1:n) {
        pvalue[t] <- (sum(chisq_null >= chisq[t]) + 1) / (w + 1)
      }
    }
  }

  # Exclude warm-up from results
  test_idx <- if (w > 0) (w + 1):n else 1:n
  chisq_test <- chisq[test_idx]
  pvalue_test <- pvalue[test_idx]

  # Multiple testing adjustment
  adjpvalue <- p.adjust(pvalue_test, method = adjust_method)

  # Signals (raw indices within test period)
  signals_raw <- which(adjpvalue < alpha)

  # Identify runs of consecutive signals
  identify_runs <- function(sig_idx) {
    if (length(sig_idx) == 0) return(list())

    runs <- list()
    run_start <- sig_idx[1]
    run_end <- sig_idx[1]

    for (i in 2:length(sig_idx)) {
      if (sig_idx[i] == sig_idx[i-1] + 1) {
        # Consecutive, extend current run
        run_end <- sig_idx[i]
      } else {
        # Gap, save current run and start new one
        runs[[length(runs) + 1]] <- c(start = run_start, end = run_end)
        run_start <- sig_idx[i]
        run_end <- sig_idx[i]
      }
    }
    # Save final run
    runs[[length(runs) + 1]] <- c(start = run_start, end = run_end)
    runs
  }

  # Helper function to determine signal observation within a run
  get_run_signal_obs <- function(run_start, run_end, position_method,
                                 y_data = NULL, eta_data = NULL, tau_levels = NULL, w_offset = 0) {
    obs_start <- run_start + w_offset
    obs_end <- run_end + w_offset

    if (position_method == "first") {
      return(obs_start)

    } else if (position_method == "last") {
      return(obs_end)

    } else if (position_method == "middle") {
      return(floor((obs_start + obs_end) / 2))

    } else if (position_method == "max_deviation") {
      # Find observation with maximum deviation from the predictive median (fitted eta at tau = 0.5)
      if (is.null(y_data) || is.null(eta_data)) {
        warning("y and eta required for max_deviation; using 'first' instead")
        return(obs_start)
      }

      # Find the index of the median quantile (tau = 0.5)
      if (!is.null(tau_levels)) {
        median_idx <- which.min(abs(tau_levels - 0.5))
      } else {
        # Fallback to middle index if taus not provided
        median_idx <- ceiling(dim(eta_data)[2] / 2)
      }

      # Compute posterior mean of the predictive median (fitted eta at tau = 0.5)
      eta_median <- apply(eta_data[, median_idx, , drop = FALSE], 3, mean)

      # Calculate deviations for observations in this run
      run_obs <- obs_start:obs_end
      deviations <- abs(y_data[run_obs] - eta_median[run_obs])

      # Return observation with maximum deviation
      max_dev_idx <- which.max(deviations)
      return(run_obs[max_dev_idx])
    }
  }

  # Process runs
  signal_runs <- identify_runs(signals_raw)

  # Build signal runs data frame with signal_obs based on signal_position
  if (length(signal_runs) > 0) {
    signal_runs_df <- data.frame(
      run_id = seq_along(signal_runs),
      start_idx = sapply(signal_runs, `[`, "start"),
      end_idx = sapply(signal_runs, `[`, "end"),
      stringsAsFactors = FALSE
    )
    signal_runs_df$obs_start <- signal_runs_df$start_idx + w
    signal_runs_df$obs_end <- signal_runs_df$end_idx + w
    signal_runs_df$run_length <- signal_runs_df$end_idx - signal_runs_df$start_idx + 1

    # Compute signal_obs for each run based on signal_position
    signal_runs_df$signal_obs <- sapply(seq_len(nrow(signal_runs_df)), function(i) {
      get_run_signal_obs(
        signal_runs_df$start_idx[i],
        signal_runs_df$end_idx[i],
        signal_position,
        y, eta, taus, w
      )
    })

    # Representative signals (one per run, based on signal_position)
    signals_representative <- signal_runs_df$signal_obs
    first_signal <- min(signals_representative)
  } else {
    signal_runs_df <- data.frame(
      run_id = integer(0), start_idx = integer(0), end_idx = integer(0),
      obs_start = integer(0), obs_end = integer(0), run_length = integer(0),
      signal_obs = integer(0)
    )
    signals_representative <- integer(0)
    first_signal <- NA
  }

  list(
    chisq = chisq_test,
    pvalue = pvalue_test,
    adjpvalue = adjpvalue,
    signals = signals_raw,               # All significant observation indices (within test period)
    signals_obs = signals_raw + w,       # All significant observations (absolute)
    signal_runs = signal_runs_df,        # Data frame with run info
    signals_representative = signals_representative,  # One signal per run based on signal_position
    n_signals = length(signals_raw),
    n_runs = nrow(signal_runs_df),
    first_signal = first_signal,
    Z_ref = Z_ref,
    S_ref = S_ref,
    Z_t = Z_t,
    df = df,
    w = w,
    use_differencing = use_differencing,
    alpha = alpha,
    p_method = p_method,
    signal_position = signal_position
  )
}


# =============================================================================
# Gamma-Based Change-Point Detection
# =============================================================================

#' Detect change points using gamma coefficients
#'
#' Uses the H-matrix gamma coefficients directly for change-point detection.
#' This approach is more aligned with the model design where gamma explicitly
#' represents shift effects at each block.
#'
#' Automatically detects the fit_method used and extracts gamma samples accordingly:
#' - "mcmc" or "map_mcmc": Uses MCMC posterior draws
#' - "map": Uses pre-generated Laplacian approximation samples
#'
#' @param fit_result Full result from getModel()
#' @param taus Quantile levels
#' @param l Block length (for converting H-column to observation)
#' @param w Warm-up period
#' @param signal_position Method to determine signal position within a significant block:
#'   - "first": First observation in the block (default)
#'   - "last": Last observation in the block
#'   - "middle": Middle observation in the block
#'   - "max_deviation": Observation with maximum deviation from the predictive median (fitted eta at tau = 0.5)
#' @param y Original data (required only for signal_position = "max_deviation")
#' @param eta Predictive quantiles array (required only for signal_position = "max_deviation")
#' @param n_samples Number of samples for Laplace approximation (only used for backward
#'   compatibility when laplace_samples not available)
#' @param alpha Significance level for credible interval test
#' @param alternative Direction of the test: "two.sided" (default), "greater", or "less".
#'   Matches R's t.test convention. "two.sided" uses decorrelated L2-norm magnitude test.
#'   "greater" tests H1: gamma > 0 (positive shift). "less" tests H1: gamma < 0 (negative shift).
#' @param seed Random seed (only used when generating new Laplace samples)
#' @return List with change-point detection results
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 100
#' y <- rnorm(n)
#' taus <- c(0.25, 0.5, 0.75)
#' H <- getIsolatedShift(n, l = 20, w = 20)
#' fit <- getModel(y, taus, H = H, w = 20, fit_method = "map",
#'                 map_hessian = FALSE, map_iter = 500)
#' result <- detectChangepoints_gamma(fit, taus = taus, l = 20, w = 20)
#' }
#' @export
detectChangepoints_gamma <- function(fit_result, taus, l, w,
                                     signal_position = c("first", "last", "middle", "max_deviation"),
                                     y = NULL, eta = NULL,
                                     n_samples = 1000, alpha = 0.05,
                                     alternative = "two.sided",
                                     seed = NULL) {

  # Validate signal_position argument
  signal_position <- match.arg(signal_position)

  m <- length(taus)

  # Determine fit method (backward compatible)
  fit_method <- fit_result$fit_method
  if (is.null(fit_method)) {
    # Infer from available data
    if (!is.null(fit_result$fit) && inherits(fit_result$fit, "stanfit")) {
      fit_method <- "mcmc"
    } else if (!is.null(fit_result$laplace_samples)) {
      fit_method <- "map"
    } else if (!is.null(fit_result$map)) {
      fit_method <- "map"
    } else {
      stop("Cannot determine fit_method from fit_result")
    }
  }

  # Extract gamma samples based on fit_method
  if (fit_method %in% c("mcmc", "map_mcmc")) {
    if (is.null(fit_result$fit) || !inherits(fit_result$fit, "stanfit")) {
      stop("fit_method is '", fit_method, "' but no stanfit object found")
    }
    draws <- rstan::extract(fit_result$fit)
    gamma_samples <- draws$gamma
    if (length(dim(gamma_samples)) == 2) {
      gamma_samples <- array(gamma_samples, dim = c(dim(gamma_samples)[1], m, 1))
    }
  } else if (fit_method == "map") {
    if (!is.null(fit_result$laplace_samples)) {
      gamma_samples <- fit_result$laplace_samples$gamma
    } else if (!is.null(fit_result$map)) {
      # Backward compatibility: generate samples on the fly
      laplace <- getLaplaceSamples(fit_result$map, fit_result$hessian,
                                   n_samples = n_samples, seed = seed)
      gamma_samples <- laplace$gamma
    } else {
      stop("fit_method is 'map' but no laplace_samples or MAP estimates found")
    }
    if (is.null(gamma_samples)) {
      stop("No gamma coefficients found in laplace_samples")
    }
  } else {
    stop("Unknown fit_method: ", fit_method)
  }

  n_iter <- dim(gamma_samples)[1]
  r <- dim(gamma_samples)[3]  # number of H columns

  # ============================================================
  # Step 1: Decorrelation via Whitening Transformation
  # ============================================================
  # Vectorize gamma: reshape from (n_iter x m x r) to (n_iter x d) where d = m * r
  # gamma_vec[s, ] = (gamma[s,1,1], gamma[s,2,1], ..., gamma[s,m,1], gamma[s,1,2], ..., gamma[s,m,r])
  d <- m * r
  gamma_vec <- matrix(NA, n_iter, d)
  for (s in 1:n_iter) {
    gamma_vec[s, ] <- as.vector(gamma_samples[s, , ])  # columns stacked: (m x r) -> (d,)
  }

  # Compute posterior covariance matrix
  Sigma <- cov(gamma_vec)

  # Compute whitening transformation using Cholesky decomposition (faster than eigendecomposition)
  # If Sigma = L L', then L^{-1} gamma gives uncorrelated samples with unit variance
  # Add small ridge for numerical stability
  ridge <- 1e-8 * diag(d)
  Sigma_reg <- Sigma + ridge

  # Cholesky decomposition: Sigma_reg = L L'
  chol_result <- tryCatch({
    chol(Sigma_reg)  # Returns upper triangular R where Sigma = R'R
  }, error = function(e) {
    # Fallback to eigendecomposition if Cholesky fails
    warning("Cholesky decomposition failed, falling back to eigendecomposition")
    NULL
  })

  if (!is.null(chol_result)) {
    # chol() returns upper triangular R where Sigma = R'R
    # Whitening: solve(R', gamma') which decorrelates the samples
    # gamma_tilde = gamma %*% solve(R) equivalently
    R <- chol_result
    gamma_tilde_vec <- t(backsolve(R, t(gamma_vec), transpose = TRUE))
    Sigma_inv_sqrt <- backsolve(R, diag(d))  # R^{-1} for output (approximate Sigma^{-1/2})
  } else {
    # Fallback: eigendecomposition (more stable for ill-conditioned matrices)
    eig <- eigen(Sigma_reg, symmetric = TRUE)
    eigenvalues <- pmax(eig$values, 1e-8)
    eigenvectors <- eig$vectors
    Sigma_inv_sqrt <- eigenvectors %*% diag(1 / sqrt(eigenvalues)) %*% t(eigenvectors)
    gamma_tilde_vec <- t(Sigma_inv_sqrt %*% t(gamma_vec))
  }

  # Reshape back to (n_iter x m x r)
  gamma_tilde <- array(NA, dim = c(n_iter, m, r))
  for (s in 1:n_iter) {
    gamma_tilde[s, , ] <- matrix(gamma_tilde_vec[s, ], nrow = m, ncol = r)
  }

  # ============================================================
  # Step 2: Per-Quantile, Per-Block Test Statistic
  # ============================================================
  # T_{q,j}^{(s)} = gamma_tilde[s, q, j]  (the decorrelated gamma itself)
  # gamma_tilde already has shape (n_iter x m x r), so no new computation needed.

  # Null threshold: Under H0 (no shift), gamma = 0
  # We test against 0, not a data-driven threshold
  epsilon <- rep(0, m)  # Null hypothesis: gamma = 0

  # ============================================================
  # Step 3: Per-Quantile, Per-Block Bayesian Posterior P-Values
  # ============================================================
  # Compute posterior p-value: P(gamma_tilde <= 0 | data) for each (q, j)
  # This tests whether the posterior mass is above or below 0
  pvalue_star <- matrix(NA, m, r)
  for (q in 1:m) {
    for (j in 1:r) {
      pvalue_star[q, j] <- mean(gamma_tilde[, q, j] <= 0)
    }
  }

  # Directional p-values (respecting the 'alternative' parameter)
  # Two-sided: reject if posterior is concentrated far from 0 (either side)
  # Greater: reject if posterior is concentrated above 0 (small pvalue_star)
  # Less: reject if posterior is concentrated below 0 (large pvalue_star)
  if (alternative == "two.sided") {
    pvalue_qj <- 2 * pmin(pvalue_star, 1 - pvalue_star)
  } else if (alternative == "greater") {
    # H1: gamma > 0, so p-value = P(gamma <= 0 | data) = pvalue_star
    pvalue_qj <- pvalue_star
  } else if (alternative == "less") {
    # H1: gamma < 0, so p-value = P(gamma >= 0 | data) = 1 - pvalue_star
    pvalue_qj <- 1 - pvalue_star
  } else {
    stop("Unknown alternative: ", alternative, ". Must be 'two.sided', 'less', or 'greater'.")
  }

  # Apply multiple testing corrections across ALL m*r p-values jointly
  pvalue_vec <- as.vector(pvalue_qj)  # length m*r (column-major: q varies fastest)
  adjp_holm_vec <- p.adjust(pvalue_vec, method = "holm")
  adjp_bonf_vec <- p.adjust(pvalue_vec, method = "bonferroni")
  adjp_bh_vec <- p.adjust(pvalue_vec, method = "BH")

  # Reshape back to m x r matrices
  adjp_holm <- matrix(adjp_holm_vec, m, r)
  adjp_bonf <- matrix(adjp_bonf_vec, m, r)
  adjp_bh <- matrix(adjp_bh_vec, m, r)

  # Block-level decision: block j significant if ANY quantile significant after correction
  sig_blocks_raw <- which(apply(pvalue_qj < alpha, 2, any))
  significant_holm <- which(apply(adjp_holm < alpha, 2, any))
  significant_bonf <- which(apply(adjp_bonf < alpha, 2, any))
  significant_bh <- which(apply(adjp_bh < alpha, 2, any))

  # Per-block summary p-value: minimum across quantiles (for backward compat)
  pvalue_posterior <- apply(pvalue_qj, 2, min)

  # Convert H column to observation number
  h_to_obs <- function(h_col) {
    obs_start <- w + (h_col - 1) * l + 1
    obs_end <- min(w + h_col * l, w + r * l)
    c(obs_start, obs_end)
  }

  # Helper function to determine signal observation within a block
  get_signal_obs <- function(h_col, position_method, y_data = NULL, eta_data = NULL, tau_levels = NULL) {
    obs_range <- h_to_obs(h_col)
    obs_start <- obs_range[1]
    obs_end <- obs_range[2]

    if (position_method == "first") {
      return(obs_start)

    } else if (position_method == "last") {
      return(obs_end)

    } else if (position_method == "middle") {
      return(floor((obs_start + obs_end) / 2))

    } else if (position_method == "max_deviation") {
      # Find observation with maximum deviation from the predictive median (fitted eta at tau = 0.5)
      if (is.null(y_data) || is.null(eta_data)) {
        warning("y and eta required for max_deviation; using 'first' instead")
        return(obs_start)
      }

      # Find the index of the median quantile (tau = 0.5)
      if (!is.null(tau_levels)) {
        median_idx <- which.min(abs(tau_levels - 0.5))
      } else {
        # Fallback to middle index if taus not provided
        median_idx <- ceiling(dim(eta_data)[2] / 2)
      }

      # Compute posterior mean of the predictive median (fitted eta at tau = 0.5)
      eta_median <- apply(eta_data[, median_idx, , drop = FALSE], 3, mean)

      # Calculate deviations for observations in this block
      block_obs <- obs_start:obs_end
      deviations <- abs(y_data[block_obs] - eta_median[block_obs])

      # Return observation with maximum deviation
      max_dev_idx <- which.max(deviations)
      return(block_obs[max_dev_idx])
    }
  }

  # Compile results
  detected_blocks <- data.frame(
    h_col = 1:r,
    # Block-level summary p-value (min across quantiles)
    pvalue_posterior = pvalue_posterior,
    # Block-level significance flags (after joint m*r correction)
    significant_raw = 1:r %in% sig_blocks_raw,
    significant_holm = 1:r %in% significant_holm,
    significant_bonf = 1:r %in% significant_bonf,
    significant_bh = 1:r %in% significant_bh
  )

  # Add observation ranges
  detected_blocks$obs_start <- sapply(detected_blocks$h_col, function(j) h_to_obs(j)[1])
  detected_blocks$obs_end <- sapply(detected_blocks$h_col, function(j) h_to_obs(j)[2])

  # Add signal observation based on signal_position method
  detected_blocks$signal_obs <- sapply(detected_blocks$h_col, function(j) {
    get_signal_obs(j, signal_position, y, eta, taus)
  })

  # First detection (using signal_obs rather than obs_start)
  first_signal_holm <- if (length(significant_holm) > 0) {
    first_sig_block <- min(significant_holm)
    detected_blocks$signal_obs[first_sig_block]
  } else NA

  first_signal_bonf <- if (length(significant_bonf) > 0) {
    first_sig_block <- min(significant_bonf)
    detected_blocks$signal_obs[first_sig_block]
  } else NA

  first_signal_bh <- if (length(significant_bh) > 0) {
    first_sig_block <- min(significant_bh)
    detected_blocks$signal_obs[first_sig_block]
  } else NA

  list(
    detected_blocks = detected_blocks,
    # Decorrelated gamma samples (n_iter x m x r)
    gamma_tilde = gamma_tilde,
    # Per-quantile, per-block p-value matrices (m x r)
    pvalue_star = pvalue_star,
    pvalue_qj = pvalue_qj,
    adjp_holm = adjp_holm,
    adjp_bonf = adjp_bonf,
    adjp_bh = adjp_bh,
    # Per-quantile null threshold
    epsilon = epsilon,
    # Whitening matrix used for decorrelation
    Sigma_inv_sqrt = Sigma_inv_sqrt,
    # Posterior covariance matrix
    Sigma = Sigma,
    # Block-level summary
    pvalue_posterior = pvalue_posterior,
    n_significant_holm = length(significant_holm),
    n_significant_bonf = length(significant_bonf),
    n_significant_bh = length(significant_bh),
    first_signal_holm = first_signal_holm,
    first_signal_bonf = first_signal_bonf,
    first_signal_bh = first_signal_bh,
    signal_position = signal_position,
    alternative = alternative,
    alpha = alpha
  )
}


#' Plot gamma-based change-point detection
#'
#' Plots block-level -log10(p-values) with significance coloring based on the
#' selected multiple testing correction method.
#'
#' @param result Output from detectChangepoints_gamma()
#' @param true_shift True shift observation (optional)
#' @param main Plot title
#' @param method Multiple testing correction to highlight: "holm" (default), "bonf", or "bh"
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 100
#' y <- rnorm(n)
#' taus <- c(0.25, 0.5, 0.75)
#' H <- getIsolatedShift(n, l = 20, w = 20)
#' fit <- getModel(y, taus, H = H, w = 20, fit_method = "map",
#'                 map_hessian = FALSE, map_iter = 500)
#' result <- detectChangepoints_gamma(fit, taus = taus, l = 20, w = 20)
#' plot_gamma_detection(result)
#' }
#' @export
plot_gamma_detection <- function(result, true_shift = NULL,
                                 main = "Gamma-Based Change-Point Detection",
                                 method = c("holm", "bonf", "bh")) {

  method <- match.arg(method)
  df <- result$detected_blocks

  # Color bars based on selected correction method
  if (method == "holm") {
    bar_colors <- ifelse(df$significant_holm, "red", "gray70")
    legend_items <- c("Significant (Holm)", "Not significant")
    legend_fills <- c("red", "gray70")
  } else if (method == "bonf") {
    bar_colors <- ifelse(df$significant_bonf, "red", "gray70")
    legend_items <- c("Significant (Bonferroni)", "Not significant")
    legend_fills <- c("red", "gray70")
  } else {
    bar_colors <- ifelse(df$significant_bh, "red", "gray70")
    legend_items <- c("Significant (BH)", "Not significant")
    legend_fills <- c("red", "gray70")
  }

  # Plot -log10(p-value) for each block
  neg_log_p <- -log10(pmax(df$pvalue_posterior, 1e-16))
  bp <- barplot(neg_log_p, names.arg = df$h_col,
                col = bar_colors,
                xlab = "H Column (Block)", ylab = "-log10(p-value)",
                main = main, ylim = c(0, max(neg_log_p) * 1.2))

  # Significance threshold line
  abline(h = -log10(result$alpha), col = "blue", lty = 2, lwd = 2)

  # True shift (convert to H column)
  if (!is.null(true_shift)) {
    shift_block <- which(df$obs_start <= true_shift & df$obs_end >= true_shift)
    if (length(shift_block) > 0) {
      abline(v = bp[shift_block[1]], col = "purple", lty = 3, lwd = 2)
    }
  }

  # Build legend
  n_fills <- length(legend_fills)
  legend_lty <- rep(NA, n_fills)
  legend_col <- rep(NA, n_fills)

  legend_items <- c(legend_items, paste0("alpha = ", result$alpha))
  legend_fills <- c(legend_fills, NA)
  legend_lty <- c(legend_lty, 2)
  legend_col <- c(legend_col, "blue")

  if (!is.null(true_shift)) {
    legend_items <- c(legend_items, "True shift")
    legend_fills <- c(legend_fills, NA)
    legend_lty <- c(legend_lty, 3)
    legend_col <- c(legend_col, "purple")
  }

  legend("topright",
         legend = legend_items,
         fill = legend_fills,
         border = c(rep("black", n_fills), rep(NA, length(legend_items) - n_fills)),
         lty = legend_lty,
         col = legend_col,
         bty = "n", cex = 0.7)
}


# =============================================================================
# QSS: Quantile Shape Statistics (Location, Scale, Skewness, Kurtosis)
# =============================================================================

#' Compute Quantile Shape Statistics (QSS) from predictive quantiles
#'
#' Using quintuple quantiles (tau = 0.05, 0.25, 0.5, 0.75, 0.95), computes:
#' - Location: L_t = Q_t(0.5) (median)
#' - Scale: S_t = Q_t(0.75) - Q_t(0.25) (IQR)
#' - Skewness: Sk_t = [(Q_t(0.75) - Q_t(0.5)) - (Q_t(0.5) - Q_t(0.25))] / IQR (Bowley)
#' - Kurtosis: K_t = [Q_t(0.95) - Q_t(0.05)] / IQR
#'
#' @param eta 3D array [iterations, quantiles, time] from getEta()
#' @param taus Vector of quantile levels (must include 0.05, 0.25, 0.5, 0.75, 0.95 or similar)
#' @return 3D array [iterations, 4, time] containing QSS statistics
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 100
#' y <- rnorm(n)
#' taus <- c(0.05, 0.25, 0.5, 0.75, 0.95)
#' H <- getIsolatedShift(n, l = 20, w = 20)
#' fit <- getModel(y, taus, H = H, w = 20, fit_method = "map",
#'                 map_hessian = FALSE, map_iter = 500)
#' eta <- getEta(fit, H = H)
#' qss <- getQSS(eta, taus = taus)
#' }
#' @export
getQSS <- function(eta, taus = c(0.05, 0.25, 0.5, 0.75, 0.95)) {

  n_iter <- dim(eta)[1]
  m <- dim(eta)[2]
  n <- dim(eta)[3]

  if (m != 5) {
    warning("QSS expects 5 quantile levels. Attempting to use available quantiles.")
  }

  # Find indices for key quantiles
  # Assuming order: tau1 < tau2 < tau3 < tau4 < tau5
  idx_lo <- 1      # lowest (e.g., 0.05)
  idx_q1 <- 2      # first quartile (e.g., 0.25)
  idx_med <- 3     # median (e.g., 0.5)
  idx_q3 <- 4      # third quartile (e.g., 0.75)
  idx_hi <- 5      # highest (e.g., 0.95)

  # Initialize QSS array: [iterations, 4 stats, time]
  qss <- array(NA, dim = c(n_iter, 4, n))
  dimnames(qss) <- list(NULL, c("Location", "Scale", "Skewness", "Kurtosis"), NULL)

  for (s in 1:n_iter) {
    for (t in 1:n) {
      q_lo <- eta[s, idx_lo, t]
      q1 <- eta[s, idx_q1, t]
      med <- eta[s, idx_med, t]
      q3 <- eta[s, idx_q3, t]
      q_hi <- eta[s, idx_hi, t]

      iqr <- q3 - q1

      # Location (median)
      qss[s, 1, t] <- med

      # Scale (IQR)
      qss[s, 2, t] <- iqr

      # Skewness (Bowley coefficient)
      if (abs(iqr) > 1e-10) {
        qss[s, 3, t] <- ((q3 - med) - (med - q1)) / iqr
      } else {
        qss[s, 3, t] <- 0
      }

      # Kurtosis (tail weight ratio)
      if (abs(iqr) > 1e-10) {
        qss[s, 4, t] <- (q_hi - q_lo) / iqr
      } else {
        qss[s, 4, t] <- NA
      }
    }
  }

  qss
}


#' Chi-squared test on QSS statistics
#'
#' @param qss 3D array [iterations, 4, time] from getQSS()
#' @param w Warm-up period
#' @param use_differencing If TRUE, use first-order differencing
#' @param df_method Method for degrees of freedom
#' @param p_method Method for p-values
#' @param n_bootstrap Bootstrap replicates
#' @param alpha Significance level
#' @param adjust_method P-value adjustment method
#' @param signal_position Method to determine signal position within a run of consecutive signals
#' @param y Original data (required only for signal_position = "max_deviation")
#' @param taus Quantile levels (required only for signal_position = "max_deviation")
#' @return List with chi-squared statistics and diagnostic info
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 100
#' y <- rnorm(n)
#' taus <- c(0.05, 0.25, 0.5, 0.75, 0.95)
#' H <- getIsolatedShift(n, l = 20, w = 20)
#' fit <- getModel(y, taus, H = H, w = 20, fit_method = "map",
#'                 map_hessian = FALSE, map_iter = 500)
#' eta <- getEta(fit, H = H)
#' qss <- getQSS(eta, taus = taus)
#' result <- getChisq_QSS(qss, w = 20)
#' }
#' @export
getChisq_QSS <- function(qss, w = 0, use_differencing = FALSE,
                          df_method = "reduced", p_method = "chisq",
                          n_bootstrap = 1000, alpha = 0.05,
                          adjust_method = "holm",
                          signal_position = c("first", "last", "middle", "max_deviation"),
                          y = NULL, taus = NULL) {

  # QSS has dimension [iterations, 4, time]
  # Convert to same format as eta: [iterations, stats, time]
  getChisq_BQQ(qss, w = w, use_differencing = use_differencing,
               df_method = df_method, p_method = p_method,
               n_bootstrap = n_bootstrap, alpha = alpha,
               adjust_method = adjust_method,
               signal_position = signal_position,
               y = y, taus = taus)
}


# =============================================================================
# Visualization Functions
# =============================================================================

#' Plot fitted quantile curves with signals
#'
#' @param y Original data
#' @param eta 3D array [iterations, quantiles, time]
#' @param taus Quantile levels
#' @param w Warm-up period
#' @param chisq_result Result from getChisq_BQQ() (optional)
#' @param true_shift True shift point (optional, for simulation)
#' @param alpha Significance level for highlighting signals
#' @param main Plot title
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 100
#' y <- rnorm(n)
#' taus <- c(0.25, 0.5, 0.75)
#' H <- getIsolatedShift(n, l = 20, w = 20)
#' fit <- getModel(y, taus, H = H, w = 20, fit_method = "map",
#'                 map_hessian = FALSE, map_iter = 500)
#' eta <- getEta(fit, H = H)
#' plot_quantile_chart(y, eta, taus, w = 20)
#' }
#' @export
plot_quantile_chart <- function(y, eta, taus, w = 0, chisq_result = NULL,
                                true_shift = NULL, alpha = 0.05,
                                main = "Fitted Quantile Curves") {
  n <- length(y)
  m <- length(taus)

  # Posterior mean of quantiles
  eta_mean <- apply(eta, c(2, 3), mean)

  # Plot data
  plot(1:n, y, type = "p", pch = 16, cex = 0.5, col = "gray60",
       xlab = "Time", ylab = "Value", main = main)

  # True shift line
  if (!is.null(true_shift)) {
    abline(v = true_shift - 0.5, col = "red", lwd = 2, lty = 2)
  }

  # Warm-up boundary
  if (w > 0) {
    abline(v = w + 0.5, col = "blue", lwd = 1, lty = 3)
  }

  # Quantile curves
  colors <- rainbow(m, alpha = 0.8)
  for (j in 1:m) {
    lines(1:n, eta_mean[j, ], col = colors[j], lwd = 1.5)
  }

  # Signal markers
  if (!is.null(chisq_result) && length(chisq_result$signals) > 0) {
    sig_times <- chisq_result$signals + w
    abline(v = sig_times, col = "orange", lty = 2, lwd = 0.5)
  }

  legend("topleft", legend = paste0("tau=", taus),
         col = colors, lty = 1, lwd = 1.5, bty = "n", cex = 0.7)
}


#' Plot chi-squared control chart
#'
#' @param chisq_result Result from getChisq_BQQ()
#' @param w Warm-up period
#' @param true_shift True shift point (optional)
#' @param main Plot title
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 100
#' y <- rnorm(n)
#' taus <- c(0.25, 0.5, 0.75)
#' H <- getIsolatedShift(n, l = 20, w = 20)
#' fit <- getModel(y, taus, H = H, w = 20, fit_method = "map",
#'                 map_hessian = FALSE, map_iter = 500)
#' eta <- getEta(fit, H = H)
#' chisq_result <- getChisq_BQQ(eta, w = 20)
#' plot_chisq_chart(chisq_result, w = 20)
#' }
#' @export
plot_chisq_chart <- function(chisq_result, w = 0, true_shift = NULL,
                             main = "Chi-Squared Control Chart") {

  n_test <- length(chisq_result$chisq)
  time_idx <- (w + 1):(w + n_test)

  # UCL from chi-squared distribution
  ucl <- qchisq(1 - chisq_result$alpha, df = chisq_result$df)

  # Plot
  plot(time_idx, chisq_result$chisq, type = "l", lwd = 1.5,
       xlab = "Time", ylab = "Chi-Squared Statistic",
       main = main, col = "darkblue")

  # UCL
  abline(h = ucl, col = "red", lwd = 2, lty = 2)

  # True shift
  if (!is.null(true_shift)) {
    abline(v = true_shift - 0.5, col = "green", lwd = 2, lty = 3)
  }

  # Signal points
  if (length(chisq_result$signals) > 0) {
    sig_times <- chisq_result$signals + w
    sig_vals <- chisq_result$chisq[chisq_result$signals]
    points(sig_times, sig_vals, pch = 19, col = "red", cex = 1.2)
  }

  legend("topright",
         legend = c(paste0("UCL (alpha=", chisq_result$alpha, ")"),
                    if (!is.null(true_shift)) "True shift" else NULL,
                    if (length(chisq_result$signals) > 0) "Signals" else NULL),
         col = c("red", if (!is.null(true_shift)) "green" else NULL,
                 if (length(chisq_result$signals) > 0) "red" else NULL),
         lty = c(2, if (!is.null(true_shift)) 3 else NULL, NA),
         pch = c(NA, NA, if (length(chisq_result$signals) > 0) 19 else NULL),
         bty = "n", cex = 0.8)
}


#' Plot QSS time series with posterior uncertainty
#'
#' @param qss 3D array [iterations, 4, time] from getQSS()
#' @param w Warm-up period
#' @param true_shift True shift point (optional)
#' @param main_prefix Prefix for subplot titles
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 100
#' y <- rnorm(n)
#' taus <- c(0.05, 0.25, 0.5, 0.75, 0.95)
#' H <- getIsolatedShift(n, l = 20, w = 20)
#' fit <- getModel(y, taus, H = H, w = 20, fit_method = "map",
#'                 map_hessian = FALSE, map_iter = 500)
#' eta <- getEta(fit, H = H)
#' qss <- getQSS(eta, taus = taus)
#' plot_qss_series(qss, w = 20)
#' }
#' @export
plot_qss_series <- function(qss, w = 0, true_shift = NULL,
                             main_prefix = "") {

  n_iter <- dim(qss)[1]
  n <- dim(qss)[3]

  stat_names <- c("Location", "Scale", "Skewness", "Kurtosis")

  par(mfrow = c(2, 2))

  for (k in 1:4) {
    # Posterior mean and quantiles
    stat_mean <- apply(qss[, k, , drop = FALSE], 3, mean)
    stat_lo <- apply(qss[, k, , drop = FALSE], 3, quantile, probs = 0.025)
    stat_hi <- apply(qss[, k, , drop = FALSE], 3, quantile, probs = 0.975)

    # Plot
    ylim <- range(c(stat_lo, stat_hi), na.rm = TRUE)
    plot(1:n, stat_mean, type = "l", lwd = 2, col = "darkblue",
         xlab = "Time", ylab = stat_names[k],
         main = paste0(main_prefix, stat_names[k]),
         ylim = ylim)

    # Credible band
    polygon(c(1:n, n:1), c(stat_lo, rev(stat_hi)),
            col = rgb(0, 0, 1, 0.2), border = NA)

    # Warm-up boundary
    if (w > 0) {
      abline(v = w + 0.5, col = "blue", lty = 3)
    }

    # True shift
    if (!is.null(true_shift)) {
      abline(v = true_shift - 0.5, col = "red", lwd = 2, lty = 2)
    }
  }

  par(mfrow = c(1, 1))
}
