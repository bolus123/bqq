#' Posterior z-scores and normal-approx p-values for gamma
#'
#' Computes elementwise posterior \emph{z}-scores and corresponding
#' normal-approximation p-values for the grouped-\eqn{\gamma} coefficients
#' extracted from a \code{rstan::stanfit} object. The \emph{z}-score for each
#' \eqn{(q, j)} is defined as \eqn{\bar{\gamma}_{qj} / \mathrm{sd}(\gamma_{qj})}
#' over posterior draws.
#'
#' @param fit A fitted \code{rstan::stanfit} from \code{\link{getModel}} (the
#'   Stan program must include a parameter \code{gamma} with draws shaped
#'   \code{[iterations, m, r]}).
#' @param side Character, one of \code{"two.sided"}, \code{"greater"},
#'   or \code{"less"}; controls the tail for the p-value calculation.
#'
#' @details
#' This function uses \code{rstan::extract()} to get the posterior array
#' \code{gamma}. For each quantile index \eqn{q = 1,\dots,m} and column
#' \eqn{j = 1,\dots,r}, it computes
#' \deqn{z_{qj} = \frac{\mathrm{mean}(\gamma_{qj}^{(s)})}{\mathrm{sd}(\gamma_{qj}^{(s)})}}
#' and converts \eqn{z_{qj}} to a p-value using the standard normal CDF:
#' \itemize{
#'   \item \code{"two.sided"}: \eqn{2 \min\{\Phi(z), 1-\Phi(z)\}}
#'   \item \code{"greater"}: \eqn{1 - \Phi(z)}
#'   \item \code{"less"}: \eqn{\Phi(z)}
#' }
#' No multiple-testing adjustment is applied.
#'
#' @return A numeric matrix of p-values with dimension \code{m x r}
#'   (quantiles by \code{H}-columns), matching the second and third
#'   dimensions of \code{gamma}.
#'
#' @examples
#' \dontrun{
#'   # After fitting with getModel(...)
#'   pmat <- extractGamma(fit, side = "two.sided")
#'   image(t(pmat[nrow(pmat):1, ]))  # quick heatmap
#'   which(pmat < 0.05, arr.ind = TRUE)
#' }
#'
#' @seealso \code{\link{getModel}}, \code{\link{getIsolatedShift}}, \code{\link{getSustainedShift}}
#' @importFrom rstan extract
#' @importFrom stats pnorm
#' @export
extractGamma <- function(fit, side = "two.sided") {

  # extract posterior draws
  post <- rstan::extract(fit)

  post_gamma <- post$gamma
  q <- dim(post$gamma)[2]
  r <- dim(post$gamma)[3]

  post_gamma_stat <- apply(post_gamma, c(2, 3), mean) / apply(post_gamma, c(2, 3), sd)

  post_gamma_pval <- post_gamma_stat
  for (i in 1:r) {
    for (j in 1:q) {
      if (side == "two.sided") {
        post_gamma_pval[j, i] <- 2 * min(pnorm(post_gamma_stat[j, i]), 1 - pnorm(post_gamma_stat[j, i]))
      } else if (side == "greater") {
        post_gamma_pval[j, i] <- 1 - pnorm(post_gamma_stat[j, i])
      } else if (side == "less") {
        post_gamma_pval[j, i] <- pnorm(post_gamma_stat[j, i])
      }
    }
  }

  post_gamma_pval

}


#' Reconstruct posterior linear predictors \eqn{\eta} from a stanfit
#'
#' Rebuilds the per-quantile, per-time linear predictors \eqn{\eta_{q,i}}
#' from posterior draws of \eqn{\mu}, \eqn{\beta}, and \eqn{\gamma}, using the
#' same design construction as in \code{\link{getModel}} (adds the \code{X0}
#' step column defined by \code{w} and prepends it to any supplied \code{X}).
#'
#' @param fit A fitted \code{rstan::stanfit} from \code{\link{getModel}}.
#' @param H Numeric matrix \eqn{n \times r} used in the fit (same ordering as during fitting).
#' @param w Integer \(\ge 1\). Length of the initial step column \code{X0} (first \code{w} rows are 1).
#' @param X Optional numeric matrix of additional predictors (\eqn{n \times p_x}).
#'   If supplied, \code{X0} is prepended; if \code{NULL}, \code{X} is just \code{X0}.
#' @param log_flag Integer \code{0/1}. If 1, applies \code{exp()} to reconstructed
#'   \eqn{\eta} to invert the log link (to align with \code{log\_flag} used in fitting).
#' @param jittering Integer \code{0/1}. If 1, applies \code{floor()} to \eqn{\eta}
#'   to mimic any integer-valued response post-processing used during fitting.
#' @param offset Optional numeric vector of length \eqn{n} added to \eqn{\eta}.
#'   Defaults to zeros.
#'
#' @details
#' The function pulls posterior arrays via \code{rstan::extract()}:
#' \itemize{
#'   \item \code{mu}: \code{[iterations, m, n]}
#'   \item \code{beta}: \code{[iterations, m, p]} (present only if \eqn{p > 0})
#'   \item \code{gamma}: \code{[iterations, m, r]} (present if \eqn{r > 0})
#' }
#' Two reconstructions are returned:
#' \describe{
#'   \item{\code{eta_draws0}}{\eqn{\mu + H\gamma + \mathrm{offset}} (ignoring \eqn{X\beta}).}
#'   \item{\code{eta_draws1}}{\eqn{\mu + X\beta + H\gamma + \mathrm{offset}} (full predictor).}
#' }
#' If \code{log_flag = 1}, both arrays are exponentiated. If \code{jittering = 1},
#' both arrays are \code{floor()}ed.
#'
#' @return A list with two 3D arrays, each of dimension
#'   \code{[iterations, m, n]}:
#' \describe{
#'   \item{\code{eta_draws0}}{Without \eqn{X\beta}.}
#'   \item{\code{eta_draws1}}{With \eqn{X\beta}.}
#' }
#'
#' @examples
#' \dontrun{
#'   et <- getEta(fit, H = H, w = 20, X = X, log_flag = 0, jittering = 0)
#'   dim(et$eta_draws1)   # iterations x m x n
#'   # Posterior median trajectory for the median quantile:
#'   med <- apply(et$eta_draws1[, 2, ], 2, median)
#'   plot(med, type = "l")
#' }
#'
#' @seealso \code{\link{getModel}}, \code{\link{extractGamma}}
#' @importFrom rstan extract
#' @export
getEta <- function(fit, H, w, X = NULL, log_flag = 0, jittering = 0, offset = NULL) {

  post <- rstan::extract(fit)


  # mu: [iterations, m, n]
  mu_draws <- post$mu
  # beta: [iterations, m, p] (only if p > 0)
  if ("beta" %in% names(post)) {
    beta_draws <- post$beta
  } else {
    beta_draws <- NULL
  }

  if ("gamma" %in% names(post)) {
    gamma_draws <- post$gamma
  } else {
    gamma_draws <- NULL
  }

  # dimensions
  n_iter <- dim(mu_draws)[1]
  m <- dim(mu_draws)[2]
  n <- dim(mu_draws)[3]

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

  # rebuild eta_hat on the R side
  eta_draws <- array(NA, dim = c(n_iter, m, n))
  eta_draws1 <- array(NA, dim = c(n_iter, m, n))
  for (s in 1:n_iter) {
    for (q in 1:m) {
      # base: latent random walk + offset
      eta_draws[s, q, ] <- mu_draws[s, q, ] + offset
      eta_draws1[s, q, ] <- mu_draws[s, q, ] + offset
      # add predictors if present
      if (!is.null(beta_draws)) {
        xb <- if (!is.null(beta_draws)) as.vector(X %*% beta_draws[s, q, ]) else 0
        hg <- if (!is.null(gamma_draws)) as.vector(H %*% gamma_draws[s, q, ]) else 0
        eta_draws[s, q, ] <- eta_draws[s, q, ] + hg
        eta_draws1[s, q, ] <- eta_draws1[s, q, ] + xb + hg
      }
    }
  }

  if (log_flag == 1) {
    eta_draws = exp(eta_draws)
    eta_draws1 = exp(eta_draws1)
  }

  if (jittering == 1) {
    eta_draws = floor(eta_draws)
    eta_draws1 = floor(eta_draws1)
  }


  out <- list(
    eta_draws0 = eta_draws,
    eta_draws1 = eta_draws1
  )

  out

}
