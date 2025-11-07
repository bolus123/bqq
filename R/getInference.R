#' Global chi-square tests across quantiles over time
#'
#' Computes timewise (or difference-wise) \eqn{\chi^2_m} statistics that test
#' whether the vector of quantile-specific linear predictors \eqn{\eta_{\cdot,i}}
#' deviates from its overall mean (or from zero after differencing), using the
#' posterior mean vector and a pooled posterior covariance across simulations.
#'
#' @param eta 3D numeric array \code{[iterations, m, t]} of posterior draws for
#'   linear predictors (e.g., from \code{\link{getEta}}).
#' @param differences Integer \(\ge 0\). If \code{0}, test levels of
#'   \eqn{\eta_{\cdot,i}}; if \(\ge 1\), test the \code{differences}-order
#'   time differences of each quantile series.
#' @param method P-value adjustment method passed to \code{stats::p.adjust()}
#'   (e.g., \code{"holm"}, \code{"bonferroni"}, \code{"BH"}).
#' @param alternative Character, one of \code{"two.sided"}, \code{"less"},
#'   \code{"greater"}. For one-sided options, entries in the opposing direction
#'   are truncated at 0 before forming the quadratic form.
#'
#' @details
#' When \code{differences = 0}, the pooled covariance \eqn{\Sigma} of the
#' \eqn{m}-vectors \eqn{\eta_{\cdot,i}} is computed by averaging sample covariances
#' across time \eqn{i = 1,\dots,t}. The statistic for time \eqn{i} is
#' \deqn{Q_i = (\bar{\eta}_{\cdot,i} - \bar{\eta}_{\cdot,\cdot})^\top
#'   \Sigma^{-1} (\bar{\eta}_{\cdot,i} - \bar{\eta}_{\cdot,\cdot}),}
#' which is compared to \eqn{\chi^2_m}. For \code{differences > 0}, the same
#' is applied to the \code{differences}-order differences of \eqn{\eta}.
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{chisq}}{Numeric vector of length \code{t} (or \code{t - differences})
#'     with \eqn{\chi^2} statistics.}
#'   \item{\code{pvalue}}{Unadjusted p-values (vector).}
#'   \item{\code{adjpvalue}}{Adjusted p-values via \code{p.adjust(method)}.}
#'   \item{\code{differences}}{The input \code{differences}.}
#' }
#'
#' @examples
#' \dontrun{
#'   # eta: draws x quantiles x time
#'   inf <- getChisq(eta, differences = 0, method = "holm")
#'   which(inf$adjpvalue < 0.05)
#' }
#'
#' @importFrom stats pchisq p.adjust
#' @export
getChisq <- function(eta, differences = 0, w = 0, method = "holm", alternative = "two.sided") {

  nsim <- dim(eta)[1]
  m <- dim(eta)[2]
  t <- dim(eta)[3]

  if (differences == 0) {

    gm <- apply(eta, 2, mean)

    samp_cov <- matrix(0, nrow = m, ncol = m)

    for (i in 1:t) {
      tmp_samp_cov <- ((t(eta[, , i]) - gm)) %*% t(t(eta[, , i]) - gm) / (nsim - 1)
      samp_cov <- samp_cov + tmp_samp_cov

    }
    samp_cov <- samp_cov  / t

    {
      invsamp_cov <- solve(samp_cov)

      dm <- apply(eta, c(2, 3), mean)
      dm <- dm - gm

      out <- rep(NA, t)

      for (i in 1:t) {
        tmp <- dm[, i]
        if (alternative == "less") {
          tmp[tmp > 0] <- 0
        } else if (alternative == "greater") {
          tmp[tmp < 0] <- 0
        }
        out[i] <- as.numeric(tmp) %*% invsamp_cov %*% as.numeric(tmp)

      }

      out_pval <- 1 - pchisq(out, m)
      adjpvalue <- p.adjust(out_pval, method)

    }


  } else if (differences > 0) {

    tmp_eta <- array(NA, c(nsim, m, t - differences))

    for (j in 1:m) {
      tmp_eta[, j, ] <- t(diff(t(eta[, j, ]), differences = differences))
    }

    samp_cov <- matrix(0, nrow = m, ncol = m)

    for (i in 1:(t - differences)) {
      tmp_samp_cov <- ((t(tmp_eta[, , i]))) %*% t(t(tmp_eta[, , i])) / (nsim - 1)
      samp_cov <- samp_cov + tmp_samp_cov

    }
    samp_cov <- samp_cov  / (t - differences)
    invsamp_cov <- solve(samp_cov)

    dm <- apply(tmp_eta, c(2, 3), mean)

    out <- rep(NA, t - differences)

    for (i in 1:(t - differences)) {
      tmp <- dm[, i]
      if (alternative == "less") {
        tmp[tmp > 0] <- 0
      } else if (alternative == "greater") {
        tmp[tmp < 0] <- 0
      }
      out[i] <- as.numeric(tmp) %*% invsamp_cov %*% as.numeric(tmp)
    }

    out_pval <- 1 - pchisq(out, m)
    adjpvalue <- p.adjust(out_pval, method)

  }


  if (w > 0) {
    out <- list(
      chisq = out[-c(1:w)],
      pvalue = out_pval[-c(1:w)],
      adjpvalue = p.adjust(out_pval[-c(1:w)], method),
      differences = differences,
      w = w
    )
  } else {
    out <- list(
      chisq = out,
      pvalue = out_pval,
      adjpvalue = p.adjust(out_pval, method),
      differences = differences,
      w = w
    )
  }


  out

}


#' Global bootstrapped chi-square tests across quantiles over time
#'
#' Computes timewise (or difference-wise) \eqn{\chi^2_m} statistics that test
#' whether the vector of quantile-specific linear predictors \eqn{\eta_{\cdot,i}}
#' deviates from its overall mean (or from zero after differencing), using the
#' posterior mean vector and a pooled posterior covariance across simulations.
#'
#' @param eta 3D numeric array \code{[iterations, m, t]} of posterior draws for
#'   linear predictors (e.g., from \code{\link{getEta}}).
#' @param differences Integer \(\ge 0\). If \code{0}, test levels of
#'   \eqn{\eta_{\cdot,i}}; if \(\ge 1\), test the \code{differences}-order
#'   time differences of each quantile series.
#' @param method P-value adjustment method passed to \code{stats::p.adjust()}
#'   (e.g., \code{"holm"}, \code{"bonferroni"}, \code{"BH"}).
#' @param alternative Character, one of \code{"two.sided"}, \code{"less"},
#'   \code{"greater"}. For one-sided options, entries in the opposing direction
#'   are truncated at 0 before forming the quadratic form.
#' @param nsim Number of bootstrap samples to use (default 1000).
#'
#' @details
#' When \code{differences = 0}, the pooled covariance \eqn{\Sigma} of the
#' \eqn{m}-vectors \eqn{\eta_{\cdot,i}} is computed by averaging sample covariances
#' across time \eqn{i = 1,\dots,t}. The statistic for time \eqn{i} is
#' \deqn{Q_i = (\bar{\eta}_{\cdot,i} - \bar{\eta}_{\cdot,\cdot})^\top
#'   \Sigma^{-1} (\bar{\eta}_{\cdot,i} - \bar{\eta}_{\cdot,\cdot}),}
#' which is compared to \eqn{\chi^2_m}. For \code{differences > 0}, the same
#' is applied to the \code{differences}-order differences of \eqn{\eta}.
#'
#' @return A list with components:
#' \describe{
#'   \item{\code{chisq}}{Numeric vector of length \code{t} (or \code{t - differences})
#'     with \eqn{\chi^2} statistics.}
#'   \item{\code{pvalue}}{Unadjusted p-values (vector).}
#'   \item{\code{adjpvalue}}{Adjusted p-values via \code{p.adjust(method)}.}
#'   \item{\code{differences}}{The input \code{differences}.}
#' }
#'
#' @examples
#' \dontrun{
#'   # eta: draws x quantiles x time
#'   inf <- getChisq(eta, differences = 0, method = "holm")
#'   which(inf$adjpvalue < 0.05)
#' }
#'
#' @importFrom stats pchisq p.adjust
#' @export
getChisq_boostrap <- function(eta, eta0, differences = 0, w = 0, method = "holm", alternative = "two.sided", nsim = 1000) {

  nnsim <- nsim

  nsim <- dim(eta)[1]
  m <- dim(eta)[2]
  t <- dim(eta)[3]

  chisq_boostrapped0 <- matrix(NA, nrow = nnsim, ncol = t - differences)

  for (sim in 1:nnsim) {
    ind <- sample(1:nsim, nsim, replace = TRUE)

    eta0_tmp <- eta0[ind, , ]

    if (differences == 0) {

      gm <- apply(eta0_tmp, 2, mean)

      samp_cov <- matrix(0, nrow = m, ncol = m)

      for (i in 1:t) {
        tmp_samp_cov <- ((t(eta0_tmp[, , i]) - gm)) %*% t(t(eta0_tmp[, , i]) - gm) / (nsim - 1)
        samp_cov <- samp_cov + tmp_samp_cov

      }
      samp_cov <- samp_cov  / t

      {
        invsamp_cov <- solve(samp_cov)

        dm <- apply(eta0_tmp, c(2, 3), mean)
        dm <- dm - gm

        out <- rep(NA, t)

        for (i in 1:t) {
          tmp <- dm[, i]
          if (alternative == "less") {
            tmp[tmp > 0] <- 0
          } else if (alternative == "greater") {
            tmp[tmp < 0] <- 0
          }
          out[i] <- as.numeric(tmp) %*% invsamp_cov %*% as.numeric(tmp)

        }

        chisq_boostrapped0[sim, ] <- out

      }

    } else if (differences > 0) {

      tmp_eta <- array(NA, c(nsim, m, t - differences))

      for (j in 1:m) {
        tmp_eta[, j, ] <- t(diff(t(eta0_tmp[, j, ]), differences = differences))
      }

      samp_cov <- matrix(0, nrow = m, ncol = m)

      for (i in 1:(t - differences)) {
        tmp_samp_cov <- ((t(tmp_eta[, , i]))) %*% t(t(tmp_eta[, , i])) / (nsim - 1)
        samp_cov <- samp_cov + tmp_samp_cov

      }
      samp_cov <- samp_cov  / (t - differences)
      invsamp_cov <- solve(samp_cov)

      dm <- apply(tmp_eta, c(2, 3), mean)

      out <- rep(NA, t - differences)

      for (i in 1:(t - differences)) {
        tmp <- dm[, i]
        if (alternative == "less") {
          tmp[tmp > 0] <- 0
        } else if (alternative == "greater") {
          tmp[tmp < 0] <- 0
        }
        out[i] <- as.numeric(tmp) %*% invsamp_cov %*% as.numeric(tmp)
      }

      chisq_boostrapped0[sim, ] <- out

    }
  }

  #######################

  if (differences == 0) {

    gm <- apply(eta, 2, mean)

    samp_cov <- matrix(0, nrow = m, ncol = m)

    for (i in 1:t) {
      tmp_samp_cov <- ((t(eta[, , i]) - gm)) %*% t(t(eta[, , i]) - gm) / (nsim - 1)
      samp_cov <- samp_cov + tmp_samp_cov

    }
    samp_cov <- samp_cov  / t

    {
      invsamp_cov <- solve(samp_cov)

      dm <- apply(eta, c(2, 3), mean)
      dm <- dm - gm

      out <- rep(NA, t)

      for (i in 1:t) {
        tmp <- dm[, i]
        if (alternative == "less") {
          tmp[tmp > 0] <- 0
        } else if (alternative == "greater") {
          tmp[tmp < 0] <- 0
        }
        out[i] <- as.numeric(tmp) %*% invsamp_cov %*% as.numeric(tmp)

      }


    }


  } else if (differences > 0) {

    tmp_eta <- array(NA, c(nsim, m, t - differences))

    for (j in 1:m) {
      tmp_eta[, j, ] <- t(diff(t(eta[, j, ]), differences = differences))
    }

    samp_cov <- matrix(0, nrow = m, ncol = m)

    for (i in 1:(t - differences)) {
      tmp_samp_cov <- ((t(tmp_eta[, , i]))) %*% t(t(tmp_eta[, , i])) / (nsim - 1)
      samp_cov <- samp_cov + tmp_samp_cov

    }
    samp_cov <- samp_cov  / (t - differences)
    invsamp_cov <- solve(samp_cov)

    dm <- apply(tmp_eta, c(2, 3), mean)

    out <- rep(NA, t - differences)

    for (i in 1:(t - differences)) {
      tmp <- dm[, i]
      if (alternative == "less") {
        tmp[tmp > 0] <- 0
      } else if (alternative == "greater") {
        tmp[tmp < 0] <- 0
      }
      out[i] <- as.numeric(tmp) %*% invsamp_cov %*% as.numeric(tmp)
    }

  }

  chisq_vec <- out
  out_pval <- rep(NA, length(chisq_vec))
  for (i in 1:(t - differences)) {
    out_pval[i] <- mean(chisq_boostrapped0[, i] > chisq_vec[i])
  }


  if (w > 0) {
    out <- list(
      chisq = out[-c(1:w)],
      pvalue = out_pval[-c(1:w)],
      adjpvalue = p.adjust(out_pval[-c(1:w)], method),
      differences = differences,
      w = w
    )
  } else {
    out <- list(
      chisq = out,
      pvalue = out_pval,
      adjpvalue = p.adjust(out_pval, method),
      differences = differences,
      w = w
    )
  }


  out

}



#' Location–scale–skew–kurtosis from five quantiles
#'
#' Computes simple robust summaries from five quantiles: location \code{q50},
#' scale \code{q75 - q25}, a median-based skewness, and a tail-weight proxy.
#'
#' @param q1,q25,q50,q75,q99 Numeric vectors or scalars of the 1st, 25th, 50th,
#'   75th, and 99th percentiles, respectively.
#'
#' @return A list with elements \code{loc}, \code{scl}, \code{ske}, \code{kur}.
#'
#' @examples
#' lssk(1, 2, 3, 5, 9)
#'
#' @export
lssk <- function(q1, q25, q50, q75, q99) {

  loc <- q50
  scl <- q75 - q25
  ske <- ((q75 - q50) - (q50 - q25)) / scl
  kur <- (q99 - q1) / scl


  out <- list(
    loc = loc,
    scl = scl,
    ske = ske,
    kur = kur
  )

  out

}


#' Quick ribbon-style chart of y and posterior quantiles
#'
#' Plots the observed series \code{y} together with median trajectories of
#' posterior quantiles (\code{q1, q25, q50, q75, q99}), and marks time indices
#' with significant adjusted p-values from \code{\link{getChisq}}.
#'
#' @param y Numeric vector of length \eqn{n}.
#' @param q1,q25,q50,q75,q99 3D arrays or matrices whose columns (or the 3rd
#'   dimension) correspond to time; the function uses columnwise medians.
#' @param pval_chisq_adj Numeric vector of adjusted p-values (typically
#'   \code{getChisq(...)$adjpvalue}).
#' @param differences Integer used only to shift the vertical lines so that
#'   change points align with the differenced index.
#' @param FAP0 Significance threshold (false-alarm probability), default \code{0.05}.
#'
#' @details
#' Vertical dashed red lines are drawn at indices where
#' \code{pval_chisq_adj <= FAP0}, shifted by \code{differences}.
#'
#' @return Invisibly returns \code{NULL}; called for its plotting side-effect.
#'
#' @examples
#' \dontrun{
#'   plot_chart(y, q1, q25, q50, q75, q99, inf$adjpvalue, differences = 1)
#' }
#'
#' @importFrom graphics plot points abline
#' @export
plot_chart <- function(y, q1, q25, q50, q75, q99,
                       pval_chisq_adj, differences = 0, w = 0, FAP0 = 0.05) {

  lssk_tmp <- lssk(q1, q25, q50, q75, q99)

  n <- length(y)

  plot(c(1, n), c(min(y,  apply(q1 , 2, median)),
                  max(y, apply(q99, 2, median))), type = 'n', ylab = 'y', xlab = NA)

  points((1:n), y)


  points(1:n, apply(q1 , 2, median), type = 'l', lty = 2, col = 'blue')
  points(1:n, apply(q25, 2, median), type = 'l', lty = 2, col = 'blue')
  points(1:n, apply(q50, 2, median), type = 'l', lty = 2, col = 'blue')
  points(1:n, apply(q75, 2, median), type = 'l', lty = 2, col = 'blue')
  points(1:n, apply(q99, 2, median), type = 'l', lty = 2, col = 'blue')

  abline(v = which(pval_chisq_adj <= FAP0) - 0.5 + differences + w, lty = 2, col = 'red')

}



#' Elementwise p-values for gamma with multiple-testing adjustment
#'
#' Computes posterior \emph{z}-scores for \eqn{\gamma} and converts them to
#' (one- or two-sided) normal-approximation p-values, then applies
#' \code{p.adjust(method)} across all elements.
#'
#' @param fit \code{rstan::stanfit} that contains parameter \code{gamma}.
#' @param method Adjustment method for \code{stats::p.adjust()} (default \code{"bonferroni"}).
#' @param alternative One of \code{"two.sided"}, \code{"less"}, \code{"greater"}.
#'
#' @return A list with:
#' \describe{
#'   \item{\code{pvalue}}{Matrix \code{m x r} of raw p-values.}
#'   \item{\code{adjpvalue}}{Matrix \code{m x r} of adjusted p-values.}
#' }
#'
#' @examples
#' \dontrun{
#'   gp <- gamma_pval(fit, method = "BH", alternative = "two.sided")
#'   which(gp$adjpvalue < 0.05, arr.ind = TRUE)
#' }
#'
#' @importFrom rstan extract
#' @importFrom stats pnorm p.adjust
#' @export
gamma_pval <- function(fit, method = "bonferroni", alternative = "two.sided") {
  gamma <- extract(fit)
  gamma <- gamma$gamma
  gamma_std <- apply(gamma, c(2, 3), mean) / apply(gamma, c(2, 3), sd)


  if (alternative == "less") {
    pval <- pnorm(gamma_std)
  } else if (alternative == "greater") {
    pval <- 1 - pnorm(gamma_std)
  } else if (alternative == "two.sided") {
    tmp <- array(NA, c(2, dim(gamma)[2], dim(gamma)[3]))
    tmp[1, , ] <- pnorm(gamma_std)
    tmp[2, , ] <- 1 - pnorm(gamma_std)
    pval <- 2 * apply(tmp, c(2, 3), min)
  }

  adjpvalue <- matrix(p.adjust(pval, method), ncol = dim(gamma)[3])

  out <- list(
    pvalue = pval,
    adjpvalue = adjpvalue
  )

  out

}
