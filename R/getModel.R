#' Smoothed Quantile Regression with Group-Lasso Shrinkage (Stan)
#'
#' Fits a multi-quantile (\eqn{m}) regression model where the conditional
#' quantile function is modeled as a latent random walk in time (or index)
#' with optional fixed effects \eqn{X} and structured effects \eqn{H}.
#' The \eqn{H}-coefficients are shrunk via a **grouped Bayesian LASSO**
#' (column-wise sharing across quantiles), and adjacent quantiles are softly
#' penalized to discourage crossings. The likelihood is optimized using a
#' **smoothed score** (logit-smoothed indicator) with temperature
#' \code{T_rel * base_scale}. The model is implemented in Stan and estimated
#' via MCMC using \strong{rstan}.
#'
#' @section Model (high level):
#' \describe{
#'   \item{Data & design}{
#'     \itemize{
#'       \item \eqn{y_i} is optionally jittered (\code{u ~ Beta(1,1)}) and/or log-transformed.
#'       \item Combined linear predictor \eqn{\eta_{qi} = \mu_{q,i} + x_i^\top \beta_q + h_i^\top \gamma_q + \mathrm{offset}_i}.
#'       \item \eqn{\mu_{q,\cdot}} is a random walk per quantile: \eqn{\mu_{q,t} = \mu_{q,t-1} + \tau^{(rw)}_q z_{q,t-1}}.
#'     }
#'   }
#'   \item{Score objective}{
#'     Uses the logit-smoothed check-function derivative
#'     \eqn{\psi_{qi} = \tau_q - \mathrm{logit}^{-1}(-r_{qi}/(T_{\text{rel}} \cdot \text{base\_scale}))}
#'     with a quadratic form based on the quantile kernel \eqn{Q} and the Gram of \eqn{[X|H]}.
#'   }
#'   \item{Shrinkage on \eqn{\gamma}}{
#'     Column-wise grouped Bayesian LASSO:
#'     \eqn{\gamma_{qj} \sim \mathcal{N}(0, \sqrt{\tau_{\gamma,j}})},\quad
#'     \eqn{\tau_{\gamma,j} \sim \mathrm{Gamma}((m+1)/2,\ 0.5\,\lambda)},\quad
#'     \eqn{\lambda \sim \mathrm{Gamma}(\texttt{lambda\_lasso2\_a},\ \texttt{lambda\_lasso2\_b})}.
#'   }
#'   \item{Non-crossing penalty}{
#'     Adds an L1 hinge on the finite-difference derivative in \eqn{\tau},
#'     scaled by \code{penalty_c}.
#'   }
#'   \item{Optional transforms}{
#'     \code{jittering = 1} adds \eqn{u \sim \mathrm{Beta}(1,1)} to \eqn{y};
#'     \code{log\_flag = 1} fits the model on \eqn{\log(y + u)} and includes the Jacobian.
#'   }
#' }
#'
#' @param y Numeric vector of responses of length \eqn{n}.
#' @param taus Numeric vector of target quantile levels in \eqn{(0,1)}, length \eqn{m}.
#' @param H Numeric matrix \eqn{n \times r} of structured predictors for group-lasso
#'   coefficients \eqn{\gamma}. If \eqn{r = 0}, pass a zero-column matrix.
#' @param w Integer \eqn{\ge 1}. Creates an intercept-like column \code{X0} with the
#'   first \code{w} entries set to 1 and the rest 0 (e.g., a pre-period indicator),
#'   which is then included in \code{X}.
#' @param X Optional numeric matrix \eqn{n \times p_x} of additional predictors.
#'   If supplied, \code{X0} (defined by \code{w}) is prepended as the first column.
#'   If \code{NULL} (default), \code{X} consists only of \code{X0}.
#' @param offset Optional numeric vector of length \eqn{n} added to the linear predictor.
#'   Defaults to 0.
#' @param alpha,eps_w Reserved for future adaptive LASSO weights; currently unused
#'   (weight construction code is commented out).
#' @param c_sigma Nonnegative scalar controlling the (currently disabled) per-quantile
#'   scale prior; kept for compatibility.
#' @param beta_sd Positive scalar prior std dev for \code{beta} coefficients (default 1.0).
#' @param penalty_c Positive scalar weight for the non-crossing penalty (larger is stricter).
#' @param penalty_curv_c Positive scalar weight for the curvature of the non-crossing penalty (larger is stricter).
#' @param curvature_p Exponent for the curvature penalty (default 2 for L2; set >1 for other norms). With m quantiles,
#'   let dtL = tau_q - tau_{q-1}, dtR = tau_{q+1} - tau_q, d1 = (eta_q - eta_{q-1})/dtL, d2 = (eta_{q+1} - eta_q)/dtR,
#'   curv = (d2 - d1)/((dtL+dtR)/2). The penalty accumulates |curv|^p * ((dtL+dtR)/2) over q=2..m-1, averaged over n(m-2).
#' @param T_rel Positive scalar “smoothing temperature” (dimensionless). The actual
#'   smoothing scale is \code{base_scale * T_rel}, where \code{base_scale = sd(y)}.
#' @param lambda_lasso2_a,lambda_lasso2_b Positive shape/rate hyperparameters for the
#'   global LASSO rate \eqn{\lambda}.
#' @param log_flag Integer \code{0/1}. If 1, fit on \code{log(y)} (with optional jitter)
#'   and include Jacobian.
#' @param jittering Integer \code{0/1}. If 1, add \eqn{u \sim \mathrm{Beta}(1,1)} to \eqn{y}
#'   (or inside the log if \code{log_flag = 1}) to mitigate ties / boundary issues.
#' @param chains Number of MCMC chains (passed to \code{rstan::sampling()}).
#' @param iter Total iterations per chain.
#' @param warmup Warmup iterations per chain.
#' @param control Optional list passed to \code{rstan::sampling()} (e.g., \code{adapt\_delta}).
#' @param seed RNG seed.
#' @param verbose show the log.
#'
#' @details
#' Internally, \code{H} is standardized (column-wise) and, if \code{p > 0}, first orthogonalized
#' against the columns of \code{X} to reduce competition. The Gram matrix of
#' \code{Z = [X | H]} is computed once for scoring, with a tiny ridge for numerical stability.
#'
#' The function constructs a \code{stan\_model} from the embedded \code{stan\_code} string and
#' then runs \code{rstan::sampling()}. By default, only the group-LASSO hierarchy is active; the
#' element-wise adaptive-LASSO section is present but commented out in the Stan program.
#'
#' @return An object of class \code{rstan::stanfit} containing posterior draws for
#'   \eqn{\mu}, \eqn{\beta}, \eqn{\gamma}, \eqn{\tau^{(rw)}}, \eqn{\lambda}, group variances,
#'   and other latent quantities used in the smoothed score.
#'
#' @section Convergence & diagnostics:
#' Inspect R-hat, effective sample sizes, and divergences (increase \code{adapt\_delta}
#' and/or \code{iter} as needed). Because the likelihood is based on a smoothed score,
#' \code{T_rel} can materially affect mixing and sharpness of the posterior.
#'
#' @examples
#' \dontrun{
#'   set.seed(1)
#'   n   <- 100
#'   m   <- 3
#'   tau <- c(0.25, 0.5, 0.75)
#'
#'   # Design
#'   w <- 20
#'   X <- cbind(rnorm(n))          # one extra covariate (X0 is added automatically)
#'   H <- cbind(rnorm(n), rnorm(n))# two structured columns for group-lasso
#'   off <- rep(0, n)
#'
#'   # Data (toy)
#'   f  <- 0.3 * (1:n)/n + 0.5 * X[,1] - 0.7 * H[,1] + 0.2 * H[,2]
#'   y  <- f + rnorm(n, 0, 0.3)
#'
#'   fit <- getModel(
#'     y = y, taus = tau, H = H, w = w, X = X, offset = off,
#'     chains = 2, iter = 1000, warmup = 500, seed = 123
#'   )
#'
#'   print(fit, pars = c("lambda_lasso2", "tau_gamma_group"))
#'   # posterior summaries for gamma (H-coefficients) by quantile:
#'   # rstan::summary(fit, pars = "gamma")$summary
#' }
#'
#' @seealso \code{\link[rstan]{sampling}}, \code{\link[rstan]{stan_model}}
#'
#' @importFrom rstan stan_model sampling stan
#' @importFrom stats lm resid sd
#' @export
getModel <- function(y, taus, H = NULL, X = NULL, offset = NULL, w = 0,
                     alpha = 0.75, eps_w = 1e-3, c_sigma = 1.0,
                     beta_sd = 1.0,
                     penalty_c = 10, penalty_curv_c = 5, curvature_p = 1, T_rel = 0.1,
                     lambda_lasso2_a = 1, lambda_lasso2_b = 1,
                     log_flag = 0, jittering = 0,
                     chains = 1, iter = 1500, warmup = 500,
                     control = NULL,
                     seed = 123, verbose = FALSE,
                     fit_method = c("map_then_sampling", "sampling", "map"),
                     prior_gamma = c("group_lasso", "lasso", "spike_slab",
                                     "het_group_lasso", "adaptive_lasso"),
                     # spike-and-slab hyperparameters
                     spike_sd = 0.05, slab_sd = 2.0,
                     slab_pi_a = 1, slab_pi_b = 1) {

  prior_gamma <- match.arg(prior_gamma)
  fit_method  <- match.arg(fit_method)
  prior_code <- switch(
    prior_gamma,
    group_lasso        = 1L,
    lasso              = 2L,
    spike_slab         = 3L,
    het_group_lasso = 4L,
    adaptive_lasso     = 5L
  )

  safe_gamma_weights <- function(y, H, tau, alpha, eps_w, lambda_lasso = NULL) {
    r <- ncol(H)
    if (r == 0) return(numeric(0))

    # First try a more stable solver and silence non-unique warnings
    fit_q <- try(
      suppressWarnings(quantreg::rq(y ~ H - 1, tau = tau, method = "fn")),
      silent = TRUE
    )

    if (inherits(fit_q, "try-error")) {
      # If 'fn' fails, fall back to lasso
      n <- length(y)

      if (is.null(lambda_lasso)) {
        # simple heuristic for lambda (you can replace with your favorite)
        lambda_lasso <- sqrt(log(r + 1L) / n)
      }

      fit_q <- try(
        suppressWarnings(
          quantreg::rq(y ~ H - 1, tau = tau,
                       method = "lasso", lambda = lambda_lasso)
        ),
        silent = TRUE
      )

      if (inherits(fit_q, "try-error")) {
        warning("Both rq(method = 'fn') and rq(method = 'lasso') failed; using w = 1 for this tau.")
        return(rep(1, r))
      }
    }

    gamma_hat <- as.numeric(stats::coef(fit_q))
    if (length(gamma_hat) != r) {
      warning("Pilot quantile fit returned a length mismatch; using w = 1 for this tau.")
      return(rep(1, r))
    }

    # Adaptive weight: (|hat| + eps_w)^(-alpha)
    (abs(gamma_hat) + eps_w)^(-alpha)
  }

  n <- length(y)

  stan_code <- "
  data {
      int<lower=1> n;                  // observations
      int<lower=0> p;                  // predictors in eta (X)
      int<lower=2> m;                  // quantiles
      int<lower=0> r;                  // predictors in eta (H)

      matrix[n, p] X;                  // n x p
      matrix[n, r] H;                  // n x r
      vector[n] y;
      vector[n] offset;
      vector[m] tau_q;
      vector[m] mu0_init;

      real<lower=1e-12> base_scale;
      real<lower=0>      c_sigma;
      real<lower=0>      beta_sd;

      real<lower=0> penalty_c;
      real<lower=0> penalty_curv_c;
      real<lower=1> curvature_p;

      real T_rel;                      // smoothing temperature (dimensionless)

      real<lower=0> lambda_lasso2_a;
      real<lower=0> lambda_lasso2_b;

      real<lower=0, upper = 1> jittering;
      real<lower=0, upper = 1> log_flag;

      // prior selector:
      // 1 = group lasso
      // 2 = lasso (w_gamma all 1)
      // 3 = spike-slab
      // 4 = heterogeneous group lasso with Lévy mixing
      // 5 = adaptive lasso (same as 2, but w_gamma from data)
      int<lower=1, upper=5> prior_code;

      // weights for lasso / adaptive lasso / hetero group lasso
      matrix[m, r] w_gamma;

      // spike-and-slab hyperparameters
      real<lower=0> spike_sd;
      real<lower=0> slab_sd;
      real<lower=0> slab_pi_a;
      real<lower=0> slab_pi_b;
  }

  transformed data {
      // Quantile kernel Q[a,b] = min(tau_a, tau_b) - tau_a * tau_b
      matrix[m, m] Q;
      for (a in 1:m)
        for (b in 1:m)
          Q[a, b] = fmin(tau_q[a], tau_q[b]) - tau_q[a] * tau_q[b];

      // Combined design Z = [X | H] (n x pr)
      int pr = p + r;
      matrix[n, pr] Z;
      {
        for (j in 1:p)
          for (i in 1:n)
            Z[i, j] = X[i, j];

        for (j in 1:r)
          for (i in 1:n)
            Z[i, p + j] = H[i, j];
      }

      // Gram for score: Gs = Z'Z / n and its Cholesky
      matrix[pr, pr] Gs;
      matrix[pr, pr] L_Gs;
      if (pr > 0) {
        matrix[pr, n] Zt = Z';
        Gs = (Zt * Z) / n;
        for (k in 1:pr) Gs[k, k] = Gs[k, k] + 1e-8;  // tiny ridge
        L_Gs = cholesky_decompose(Gs);
      } else {
        Gs   = rep_matrix(0, 0, 0);
        L_Gs = rep_matrix(0, 0, 0);
      }
  }

  parameters {
      // Random-walk increments (non-centered)
      matrix[m, n-1] z_incr;
      vector<lower=0>[m] tau_rw;
      vector[m]       mu0;

      // X-coefficients
      matrix[m, p] beta;

      // H-coefficients
      matrix[m, r] gamma;

      // Group-level scale for group lasso (one per H column)
      vector<lower=0>[r] tau_gamma_group;

      // Element-wise local scales for lasso/adaptive lasso
      matrix<lower=0>[m, r] tau_gamma;

      // Global LASSO rate
      real<lower=0> lambda_lasso2;

      // Spike-and-slab mixing weight
      real<lower=0, upper=1> pi_slab;

      // Group-level mixer for hetero group lasso (Lévy)
      vector<lower=0>[m] omega_group;

      // jitter variable
      vector<lower=1e-12, upper = 1>[n] u;
  }

  transformed parameters {
      // RW paths
      matrix[m, n] mu;
      for (q in 1:m) {
        mu[q,1] = mu0[q];
        for (t in 2:n)
          mu[q,t] = mu[q,t-1] + tau_rw[q] * z_incr[q,t-1];
      }

      // Smoothing temperature on data scale
      real<lower=1e-12> smooth_T = base_scale * T_rel;

      vector[n] y_eff;
      y_eff = y;

      if (jittering == 1) {
        y_eff = y_eff + u;
      }
      if (log_flag == 1) {
        y_eff = log(y_eff);
      }

      vector[m-1] dtau;
      for (q in 1:(m-1)) dtau[q] = tau_q[q+1] - tau_q[q];
  }

  model {

      // jitter prior
      u ~ beta(1, 1);

      // Random walk priors
      to_vector(z_incr) ~ normal(0, 1);
      tau_rw ~ student_t(3, 0, base_scale / 3);
      mu0 ~ normal(mu0_init, 2 * base_scale);

      // beta prior
      if (p > 0) to_vector(beta) ~ normal(0, beta_sd);

      // ----- Score-based likelihood using Z = [X | H] with logit smoothing -----
      {
        if ((p + r) > 0) {
          //int pr = p + r;
          matrix[pr, m] S;

          for (q in 1:m) {
            vector[pr] s_q = rep_vector(0.0, pr);
            for (i in 1:n) {
              real xb = (p > 0) ? dot_product(to_vector(row(X, i)), to_vector(beta[q])) : 0;
              real hb = (r > 0) ? dot_product(to_vector(row(H, i)), to_vector(gamma[q])) : 0;

              real eta = mu[q, i] + xb + hb + offset[i];
              real r_i = y_eff[i] - eta;

              real z  = fmin(20, fmax(-20, -r_i / smooth_T));
              real Ilt = inv_logit(z);
              real psi = tau_q[q] - Ilt;

              if (p > 0) s_q[1:p]      += to_vector(row(X, i)) * psi;
              if (r > 0) s_q[(p+1):pr] += to_vector(row(H, i)) * psi;
            }
            S[, q] = s_q;
          }

          matrix[m, m] L_Q = cholesky_decompose(Q);
          // A = L_Gs^{-1} S   (pr x m)
          // B = L_Q^{-1} A'   (m x pr)
          matrix[pr, m] A = mdivide_left_tri_low(L_Gs, S);
          matrix[m, pr] B = mdivide_left_tri_low(L_Q, A');

          target += -0.5 * dot_self(to_vector(B)) / n;
        }
      }

      // ----- Priors on gamma (H-coefficients) -----
      if (r > 0) {

        if (prior_code != 3)  {
          lambda_lasso2 ~ gamma(lambda_lasso2_a, lambda_lasso2_b);
        }

        // 1 = group lasso (original grouped Bayesian lasso)
        if (prior_code == 1) {

          for (i in 1:r) {
            tau_gamma_group[i] ~ gamma( (m + 1) / 2, 0.5 * lambda_lasso2 );
            for (j in 1:m) {
              gamma[j, i] ~ normal(0, sqrt(tau_gamma_group[i]));
            }
          }

        // 2 = lasso (all w_gamma = 1) or
        // 5 = adaptive lasso (w_gamma from data)
        } else if (prior_code == 2 || prior_code == 5) {

          for (j in 1:m) {
            for (i in 1:r) {
              // tau_gamma[q,j] ~ Exp( (lambda * w_gamma[q,j])^2 / 2 )
              tau_gamma[j, i] ~ exponential(0.5 * lambda_lasso2 * square(w_gamma[j, i]));
              gamma[j, i] ~ normal(0, sqrt(tau_gamma[j, i]));
            }
          }

        // 3 = spike-and-slab
        } else if (prior_code == 3) {
          pi_slab ~ beta(slab_pi_a, slab_pi_b);

          for (j in 1:m) {
            for (i in 1:r) {
              target += log_mix(
                pi_slab,
                normal_lpdf(gamma[j, i] | 0, slab_sd),
                normal_lpdf(gamma[j, i] | 0, spike_sd)
              );
            }
          }

        // 4 = heterogeneous group lasso with Lévy(0, c_levy) mixing
        } else if (prior_code == 4) {

          real c_levy = lambda_lasso2 / 2;


          for (j in 1:m) {
            // Lévy(0, c_levy) ⇔ InvGamma(1/2, c_levy/2)
            omega_group[j] ~ inv_gamma(0.5, 0.5 * c_levy);

            for (i in 1:r) {

              tau_gamma[j, i] ~ exponential(0.5 * square(omega_group[j] * w_gamma[j, i]));
              // gamma_{qj} | omega_j ~ N(0, 2 * omega_j / w_gamma[q,j])
              gamma[j, i] ~ normal(0, sqrt(tau_gamma[j, i]));
            }
          }
        }
      }

      // ---- Non-crossing penalty and curvature penalty ----
      {
        real pen = 0;
        real pen_curv = 0;

        for (i in 1:n) {
          vector[m] eta_row;
          for (q in 1:m) {
            real xb = (p > 0) ? dot_product(to_vector(row(X, i)), to_vector(beta[q])) : 0;
            real hb = (r > 0) ? dot_product(to_vector(row(H, i)), to_vector(gamma[q])) : 0;
            eta_row[q] = mu[q, i] + xb + hb + offset[i];
          }

          // ---- Non-crossing penalty
          for (q in 1:(m-1)) {
            real dfdtau = (eta_row[q + 1] - eta_row[q]) / dtau[q];
            pen += fmax(0, -dfdtau);
          }

          // ----- Curvature penalty (accumulate curvature across time)
          if (m > 2) {
            for (q in 2:(m-1)) {
              real dtL = tau_q[q]   - tau_q[q-1];
              real dtR = tau_q[q+1] - tau_q[q];

              real d1 = (eta_row[q]   - eta_row[q-1]) / dtL;
              real d2 = (eta_row[q+1] - eta_row[q])   / dtR;
              real curv = (d2 - d1) / ((dtL + dtR) / 2);

              // accumulate |curv|^p per (i,q)
              pen_curv += pow(fabs(curv), curvature_p);
            }
          }
        }

        // ---- Non-crossing penalty
        pen /= (n * (m - 1));
        target += - penalty_c * pen;

        // ----- Curvature penalty: mean |curv|^p over i,q, Lp norm
        if (m > 2) {
          pen_curv /= (n * (m - 2));
          pen_curv = pow(pen_curv, 1.0 / curvature_p);
          target += - penalty_curv_c * pen_curv;
        }
      }

      // Jacobian adjustment for log transform
      if (log_flag == 1) {
        if (jittering == 1) {
          target += -sum(log(y + u));
        } else {
          target += -sum(log(y));
        }
      }
  }
  "

  # ---------------- R-side pre-processing ----------------

  n <- length(y)
  m <- length(taus)

  if (is.null(offset)) {
    offset <- rep(0, n)
  }

  # X0: pre-period indicator
  #X0 <- matrix(0, nrow = n, ncol = 1)
  #if (w > 0) {
  #  X0[1:w, ] <- 1
  #}
  X0 <- matrix(1, nrow = n, ncol = 1)

  if (is.null(X)) {
    X <- X0
  } else {
    X <- cbind(X0, X)
  }
  p <- ncol(X)

  # H processing: standardize and orthogonalize against X (if any)
  if (is.null(H)) {
    r <- 0
    H <- matrix(0, n, 0)
  } else {
    r <- ncol(H)
    if (r == 0) {
      H <- matrix(0, n, 0)
    } else {
      if (p > 0) {
        H_ortho <- apply(H, 2, function(col) stats::resid(stats::lm(col ~ X)))
        Hs <- scale(H_ortho, center = TRUE, scale = TRUE)
      } else {
        Hs <- scale(H, center = TRUE, scale = TRUE)
      }
      H <- Hs
    }
  }

  # Base scale for smoothing (robust)
  base_scale <- max(1e-8, 1.4826 * stats::mad(y))

  # Initial mu0 by quantiles, possibly only in pre-period
  if (w > 0) {
    mu0_init <- stats::quantile(y[1:w], probs = taus)
  } else {
    mu0_init <- stats::quantile(y, probs = taus)
  }

  # ---- w_gamma construction ----
  # default: all ones
  w_gamma <- matrix(1, nrow = m, ncol = r)

  if (r > 0 && prior_gamma %in% c("het_group_lasso", "adaptive_lasso")) {
    if (!requireNamespace("quantreg", quietly = TRUE)) {
      stop("Package 'quantreg' is required for het_group_lasso / adaptive_lasso weights.")
    }

    for (q in seq_len(m)) {
      w_gamma[q, ] <- safe_gamma_weights(
        y   = y,
        H   = H,
        tau = taus[q],
        alpha  = alpha,
        eps_w  = eps_w
        # you can optionally pass lambda_lasso = ... here
      )
    }

    # Normalize so median weight is 1 (helps interpret the penalty level)
    med_w <- stats::median(w_gamma)
    if (is.finite(med_w) && med_w > 0) {
      w_gamma <- w_gamma / med_w
    } else {
      warning("Median of w_gamma is non-finite or non-positive; skipping normalization.")
    }
  }


  stan_data <- list(
    n = n, p = p, m = m, r = r,
    X = X, H = H,
    y = y, offset = offset, tau_q = taus,
    mu0_init = as.vector(mu0_init),
    base_scale = base_scale, c_sigma = c_sigma, beta_sd = beta_sd,
    penalty_c = penalty_c, penalty_curv_c = penalty_curv_c, curvature_p = curvature_p, T_rel = T_rel,
    lambda_lasso2_a = lambda_lasso2_a, lambda_lasso2_b = lambda_lasso2_b,
    log_flag = log_flag, jittering = jittering,
    prior_code = prior_code,
    w_gamma = if (r > 0) w_gamma else matrix(0, m, 0),
    spike_sd = spike_sd,
    slab_sd  = slab_sd,
    slab_pi_a = slab_pi_a,
    slab_pi_b = slab_pi_b
  )

  # Compile Stan model once
  sm <- rstan::stan_model(model_code = stan_code)

  # Always obtain a MAP estimate first
  if ((fit_method == "map_then_sampling") | (fit_method == "map")) {
    map_fit <- {
      opt_args <- list(
        object = sm,
        data = stan_data,
        hessian = TRUE,
        as_vector = FALSE,
        seed = seed,
        verbose = verbose
      )
      # rstan::optimizing does not accept 'control' in some versions; only add if non-null and recognized
      # old: control = control
      suppressWarnings(do.call(rstan::optimizing, opt_args))
    }
    hessian <- map_fit$hessian
  } else {
    map_fit <- NULL
    hessian <- NULL
  }
  

  if (fit_method == "map_then_sampling") {
    init_theta <- map_fit$par  # constrained params
    fit <- suppressWarnings(
      rstan::sampling(
        sm, data = stan_data,
        chains = chains, iter = iter, warmup = warmup,
        init = function() init_theta, init_r = 0.01,
        control = control, seed = seed, verbose = verbose
      )
    )
  } else if (fit_method == "sampling") {
    fit <- suppressWarnings(
      rstan::sampling(
        sm, data = stan_data,
        chains = chains, iter = iter, warmup = warmup,
        control = control, seed = seed, verbose = verbose
      )
    )
  } else { # map only
    fit <- NULL
  }

  list(
    fit = fit,
    map = map_fit,
    y = y, H = H, X = X,
    hessian = hessian
  )
}
