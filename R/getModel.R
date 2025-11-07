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
#' @param penalty_c Positive scalar weight for the non-crossing penalty (larger is stricter).
#' @param penalty_curv_c Positive scalar weight for the curvature of the non-crossing penalty (larger is stricter).
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
                     penalty_c = 10, penalty_curv_c = 10, T_rel = 0.1,
                     lambda_lasso2_a = 1, lambda_lasso2_b = 1,
                     log_flag = 0, jittering = 0,
                     chains = 1, iter = 1500, warmup = 500,
                     control = NULL,
                     seed = 123, verbose = FALSE) {



  stan_code <- "data {
      int<lower=1> n;                  // observations
      int<lower=0> p;                  // predictors in eta (X)
      int<lower=2> m;                  // quantiles
      int<lower=0> r;                  // predictors in eta (H) — can be 0

      matrix[n, p] X;                  // n x p
      matrix[n, r] H;                  // n x r  (new)
      vector[n] y;
      vector[n] offset;
      vector[m] tau_q;
      vector[m] mu0_init;

      real<lower=1e-12> base_scale;
      real<lower=0>      c_sigma;

      real<lower=0> penalty_c;
      real<lower=0> penalty_curv_c;      // weight of curvature penalty

      real T_rel;                  // smoothing temperature (dimensionless, learned upstream)


      // Adaptive LASSO for gamma
      //matrix[m, r] w_gamma;           // weights >= 0 (element-wise, or make it vector[r] if shared)
      //real<lower=0> lambda_lasso2;     // global L1 penalty level

      real<lower=0> lambda_lasso2_a;
      real<lower=0> lambda_lasso2_b;

      real<lower=0, upper = 1> jittering;
      real<lower=0, upper = 1> log_flag;

    }

    transformed data {
      // Quantile kernel Q[a,b] = min(tau_a, tau_b) - tau_a * tau_b
      matrix[m, m] Q;
      for (a in 1:m)
        for (b in 1:m)
          Q[a, b] = fmin(tau_q[a], tau_q[b]) - tau_q[a] * tau_q[b];

      // ----- Build combined design Z = [X | H] (n x pr) -----
      int pr = p + r;
      matrix[n, pr] Z;
      {
        // Fill X columns (if any)
        for (j in 1:p)
          for (i in 1:n)
            Z[i, j] = X[i, j];

        // Fill H columns (if any)
        for (j in 1:r)
          for (i in 1:n)
            Z[i, p + j] = H[i, j];
      }

      // ----- Gram for score: Gs = Z'Z / n and its Cholesky -----
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

      // X-coefficients (simple Normal prior, as in your code)
      matrix[m, p] beta;

      // KEEP gamma as the actual coefficients:
      //matrix[m, r] gamma;

      // Bayesian LASSO local scales (variance parameters)
      //matrix<lower=0>[m, r] tau_gamma;          // element-wise local variances

      matrix[m, r] gamma;
      vector<lower=0>[r] tau_gamma_group;   // one per H column

      // Global LASSO rate (λ > 0); you can put prior on λ or on log λ
      real<lower=0> lambda_lasso2;


      // per-quantile scale carried in score
      //real<lower=1e-12> sigma_q;

      vector<lower=1e-12, upper = 1>[n] u;
    }

    transformed parameters {
      // RW paths
      matrix[m, n] mu;
      for (q in 1:m) {
        mu[q,1] = mu0[q];
        for (t in 2:n)
          mu[q,t] = mu[q,t-1] + tau_rw[q] * z_incr[q,t-1];
          //mu[q,t] = mu[q,t-1] + z_incr[q,t-1];
      }

      // Smoothing temperature on data scale
      real<lower=1e-12> smooth_T = base_scale * T_rel;

      vector[n] y_eff;
      y_eff = y;

      {

        if (jittering == 1) {
          y_eff = y_eff + u;
        }

        if (log_flag == 1) {
          y_eff = log(y_eff);
        }
      }

      vector[m-1] dtau;
      for (q in 1:(m-1)) dtau[q] = tau_q[q+1] - tau_q[q];


    }

    model {

      u ~ beta(1, 1);
      // --- Priors ---
      to_vector(z_incr) ~ normal(0, 1);
      tau_rw ~ student_t(3, 0, base_scale/10);
      mu0 ~ normal(mu0_init, 2 * base_scale);

      if (p > 0) to_vector(beta) ~ normal(0, 1);

      //sigma_q ~ student_t(3, 0, c_sigma * base_scale);

      // ----- σ-aware score using Z = [X | H] with logit smoothing -----
      {
        if (pr > 0) {
          matrix[pr, m] S;
          for (q in 1:m) {
            vector[pr] s_q = rep_vector(0.0, pr);
            for (i in 1:n) {
              // Linear predictor uses separate beta (for X) and gamma (for H)
              real xb = (p > 0) ? dot_product(to_vector(row(X, i)), to_vector(beta[q])) : 0;
              real hb = (r > 0) ? dot_product(to_vector(row(H, i)), to_vector(gamma[q])) : 0;

              real eta = mu[q, i] + xb + hb + offset[i];
              real r_i = y_eff[i] - eta;

              // smoothed indicator: I(r<0) ≈ inv_logit( -r / smooth_T )
              real z  = fmin(20, fmax(-20, -r_i / smooth_T));
              real Ilt = inv_logit(z);
              real psi = tau_q[q] - Ilt;

              if (p > 0) s_q[1:p]      += to_vector(row(X, i)) * (psi);
              if (r > 0) s_q[(p+1):pr] += to_vector(row(H, i)) * (psi);

            }
            S[, q] = s_q;
          }

          // Q_sigma = D^{-1} Q D^{-1}, D = diag(sigma_q)
          //matrix[m, m] Q_sigma;
          //{
          //  vector[m] invsig = 1;
          //  matrix[m, m] Dinv = diag_matrix(invsig);
          //  Q_sigma = Dinv * Q * Dinv;
          //  for (k in 1:m) Q_sigma[k,k] = Q_sigma[k,k] + 1e-10;
          //}
          //matrix[m, m] L_Q = cholesky_decompose(Q_sigma);
          matrix[m, m] L_Q = cholesky_decompose(Q);

          // A = L_Gs^{-1} S   (pr x m)
          // B = L_Q^{-1} A'   (m x pr)
          matrix[pr, m] A = mdivide_left_tri_low(L_Gs, S);
          matrix[m, pr] B = mdivide_left_tri_low(L_Q, A');

          target += -0.5 * dot_self(to_vector(B)) / n;
        }
      }

      // ----- Bayesian adaptive LASSO for gamma (Park & Casella, 2008) -----
      // gamma_{qj} | tau_{qj} ~ Normal(0, sqrt(tau_{qj}))
      // tau_{qj} ~ Exponential( (lambda_lasso2 * w_gamma[q,j])^2 / 2 )
      {
        //// 1) local variances
        //for (q in 1:m)
        //  for (j in 1:r)
        //    tau_gamma[q, j] ~ exponential(0.5 * square(lambda_lasso2 * w_gamma[q, j]));
      //
        //// 2) conditional normals
        //// Stan's normal() takes SD; use sqrt(tau) as the SD
        //for (q in 1:m)
        //  for (j in 1:r)
        //    gamma[q, j] ~ normal(0, sqrt(tau_gamma[q, j]));
      //
        //// 3) weakly informative prior on lambda_lasso2
        //// (you can tune this—smaller mean => stronger shrinkage => more sensitivity)
        //lambda_lasso2 ~ lognormal(log(1.0), 0.7);

        // global shrinkage (you already had this)
        lambda_lasso2 ~ gamma(lambda_lasso2_a, lambda_lasso2_b);

        // group scales per H column (j = 1..r)
        for (j in 1:r) {
          tau_gamma_group[j] ~ gamma( (m + 1) / 2, 0.5 * lambda_lasso2 );
        }

        // coefficients, sharing the column-wise scale
        for (j in 1:r) {
          for (q in 1:m) {
            gamma[q, j] ~ normal(0, sqrt(tau_gamma_group[j]));
          }
        }

      }




      // ---- Non-crossing penalty ----
      {
        real pen = 0;
        for (i in 1:n) {
          for (q in 1:(m-1)) {
            real eta_q =
                mu[q, i]
              + ((p>0)? dot_product(to_vector(row(X,i)), to_vector(beta[q])) : 0)
              + ((r>0)? dot_product(to_vector(row(H,i)), to_vector(gamma[q])) : 0)
              + offset[i];
            real eta_q1 =
                mu[q+1, i]
              + ((p>0)? dot_product(to_vector(row(X,i)), to_vector(beta[q+1])) : 0)
              + ((r>0)? dot_product(to_vector(row(H,i)), to_vector(gamma[q+1])) : 0)
              + offset[i];

            real dfdtau = (eta_q1 - eta_q) / dtau[q];   // finite-diff ∂τ f
            pen += fmax(0, -dfdtau);                    // L1 hinge on negative derivative
          }
        }
        pen /= (n * (m - 1));       // empirical average over i and adjacent τ-intervals
        target += - penalty_c * pen; // Stan maximizes ⇒ subtract the positive penalty
      }

      // ----- Curvature (second-difference) penalty across τ -----
      {
        real pen_curv = 0;

        for (i in 1:n) {
          // you can precompute eta[q] into a small local vector to avoid recomputing dot products
          vector[m] eta_row;
          for (q in 1:m) {
            real xb = (p > 0) ? dot_product(to_vector(row(X, i)), to_vector(beta[q])) : 0;
            real hb = (r > 0) ? dot_product(to_vector(row(H, i)), to_vector(gamma[q])) : 0;
            eta_row[q] = mu[q, i] + xb + hb + offset[i];
          }

          for (q in 2:(m-1)) {
            real dtL = tau_q[q]   - tau_q[q-1];         // Δτ_{q-1}
            real dtR = tau_q[q+1] - tau_q[q];           // Δτ_q

            // non-uniform second derivative approximation:
            real d1 = (eta_row[q]   - eta_row[q-1]) / dtL;
            real d2 = (eta_row[q+1] - eta_row[q])   / dtR;
            real curv = 2.0 * (d2 - d1) / (dtL + dtR);  // ≈ f''(τ_q)

            // scale and (optional) tail emphasis weight
            pen_curv += square(curv);   // quadratic; swap for Huber if preferred
          }
        }

        pen_curv /= (n * (m - 2));  // normalize
        target += - penalty_curv_c * pen_curv;
      }


      // Jacobian adjustment

      if (log_flag == 1) {
        if (jittering == 1) {
          target += -sum(log(y + u));
        } else {
          target += -sum(log(y));
        }


      }

    }

  "



  n <- length(y)
  m <- length(taus)

  if (is.null(offset)) {
    offset <- rep(0, n)
  }

  X0 <- matrix(0, nrow = n, ncol = 1);
  X0[1:w, ] <- 1

  if (is.null(X)) {
    X <- X0
  } else {
    X <- cbind(X0, X)
  }

  p <- ncol(X)


  # -- H handling (must set both r and H) ---------------------------------------
  if (is.null(H)) {
    r <- 0
    H <- matrix(0, n, 0)        # n x 0 matrix for Stan
  } else {
    r <- ncol(H)
    if (r == 0) {
      H <- matrix(0, n, 0)      # just in case a 0-col matrix was passed
    } else {
      # Standardize H; if X present, orthogonalize H against X first
      if (p > 0) {
        H_ortho <- apply(H, 2, function(col) resid(stats::lm(col ~ X)))
        Hs <- scale(H_ortho, center = TRUE, scale = TRUE)
      } else {
        Hs <- scale(H, center = TRUE, scale = TRUE)
      }
      H <- Hs                    # <--- IMPORTANT: use the processed H
    }
  }
  # ------------------------------------------------------------------------------


  # Pilot estimate for gamma (per-quantile or pooled). Easiest: ridge on H (per quantile):
  # alpha <- 0.75               # adaptive exponent (0.5..1)
  # eps_w <- 1e-3               # stabilizer
  #w_gamma <- matrix(1, m, r)  # default weights
  #
  #for (q in 1:m) {
  #  # Build pseudo-target: residual after removing current mu and X (if desired),
  #  # but as a first pass you can just regress y on Hs for weights:
  #  fit_ridge <- glmnet::glmnet(Hs, y, alpha = 0, lambda = NULL, standardize = FALSE)
  #  # choose lambda by CV or take minimal lambda:
  #  lam_star <- tail(fit_ridge$lambda, 1)
  #  gamma_hat <- as.numeric(coef(fit_ridge, s = lam_star))[-1]  # drop intercept
  #  w_gamma[q, ] <- (abs(gamma_hat) + eps_w)^(-alpha)
  #}
  #
  ## Scale weights so median is 1 (helps interpret lambda_lasso2):
  #w_gamma <- w_gamma / median(w_gamma)

  base_scale <- max(1e-8, sd(y))

  stan_data <- list(
    n = n, p = p, m = m, r = r,
    X = X, H = H,
    y = y, offset = offset, tau_q = taus,
    mu0_init = quantile(y, probs = taus),
    base_scale = base_scale, c_sigma = c_sigma,
    penalty_c = penalty_c, penalty_curv_c = penalty_curv_c, T_rel = T_rel,
    lambda_lasso2_a = lambda_lasso2_a, lambda_lasso2_b = lambda_lasso2_b,
    log_flag = log_flag, jittering = jittering
  )

  #sm <- stan_model(model_code = stan_code)


  fit <- suppressWarnings(stan(
    model_code = stan_code, data = stan_data,
    chains = chains, iter = iter, warmup = warmup,
    control = control,
    seed = seed,
    verbose = verbose
  ))

}
