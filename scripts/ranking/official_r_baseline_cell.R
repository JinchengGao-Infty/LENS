args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: official_r_baseline_cell.R METHOD OUTPUT_DIR")
}
method <- args[[1L]]
output_dir <- args[[2L]]
if (file.exists(file.path(output_dir, "DONE"))) {
  quit(status = 0L)
}

suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(xgboost))
started <- proc.time()[["elapsed"]]

atomic_json <- function(path, payload) {
  temporary <- paste0(path, ".tmp")
  write_json(payload, temporary, auto_unbox = TRUE, pretty = TRUE, digits = NA)
  if (!file.rename(temporary, path)) stop("atomic JSON rename failed")
}

tryCatch({
  meta <- fromJSON(file.path(output_dir, "input_meta.json"), simplifyVector = FALSE)
  train <- read.csv(file.path(output_dir, "train.csv"), check.names = FALSE)
  y <- train[["y"]]
  X_train <- as.matrix(train[setdiff(names(train), "y")])
  feature_names <- colnames(X_train)
  model <- xgb.train(
    params = list(
      objective = "reg:squarederror",
      max_depth = 4L,
      eta = 0.05,
      subsample = 1.0,
      colsample_bytree = 1.0,
      nthread = 1L,
      seed = as.integer(meta$seed)
    ),
    data = xgb.DMatrix(X_train, label = y),
    nrounds = 300L,
    verbose = 0L
  )
  predict_fun <- function(object, X) {
    as.numeric(predict(object, as.matrix(X)))
  }
  truth_rows <- meta$truth_pairs
  truth_keys <- if (length(truth_rows)) {
    vapply(truth_rows, function(x) paste(sort(as.integer(x)), collapse = "_"), character(1))
  } else character()
  mass_map <- setNames(
    if (length(meta$truth_masses)) as.numeric(unlist(meta$truth_masses)) else numeric(),
    truth_keys
  )

  if (method == "hstats") {
    suppressPackageStartupMessages(library(hstats))
    reference <- read.csv(file.path(output_dir, "reference.csv"), check.names = FALSE)
    set.seed(as.integer(meta$seed))
    H <- hstats(
      model,
      X = reference,
      pred_fun = predict_fun,
      pairwise_m = ncol(reference),
      threeway_m = 0L,
      n_max = 500L,
      verbose = FALSE
    )
    native <- h2_pairwise(H, normalize = TRUE, squared = TRUE, sort = FALSE, zero = TRUE)$M
    scores <- setNames(rep(0, choose(ncol(reference), 2L)), character(choose(ncol(reference), 2L)))
    pairs <- combn(seq_len(ncol(reference)) - 1L, 2L, simplify = FALSE)
    names(scores) <- vapply(pairs, function(x) paste(x, collapse = "_"), character(1))
    if (!is.null(native)) {
      native_values <- as.numeric(native[, 1L])
      native_names <- rownames(native)
      for (idx in seq_along(native_values)) {
        pieces <- strsplit(native_names[[idx]], "[: ,]+")[[1L]]
        indices <- match(pieces, feature_names) - 1L
        if (length(indices) == 2L && all(is.finite(indices))) {
          scores[[paste(sort(indices), collapse = "_")]] <- native_values[[idx]]
        }
      }
    }
    ord <- order(-scores, names(scores))
    selected <- names(scores)[ord[seq_len(min(as.integer(meta$top_k), length(ord)))]]
    hits <- intersect(selected, truth_keys)
    labels <- as.integer(names(scores) %in% truth_keys)
    ranks <- rank(-scores, ties.method = "average")
    average_precision <- if (length(truth_keys)) {
      sum(cumsum(labels[order(ranks)]) / seq_along(labels) * labels[order(ranks)]) / sum(labels)
    } else NULL
    metrics <- list(
      recall_at_k = if (length(truth_keys)) length(hits) / length(truth_keys) else NULL,
      weighted_recall_at_k = if (sum(mass_map) > 0) sum(mass_map[hits]) / sum(mass_map) else NULL,
      average_precision = average_precision,
      true_positives_at_k = length(hits),
      false_positives_at_k = length(selected) - length(hits),
      k = as.integer(meta$top_k)
    )
    rows <- lapply(names(scores)[ord], function(key) list(
      pair = as.integer(strsplit(key, "_")[[1L]]), score = unname(scores[[key]])
    ))
    result <- list(
      schema_version = "lens-official-stage1-hstats-cell-1",
      status = "DONE", stage = 1L, method = "hstats-official",
      dgp = meta$dgp, seed = as.integer(meta$seed),
      complete_ranking = rows,
      top_k_pairs = lapply(selected, function(key) as.integer(strsplit(key, "_")[[1L]])),
      metrics = metrics,
      method_config = list(pairwise_m = ncol(reference), n_max = 500L,
                           normalize = TRUE, squared = TRUE),
      provenance = list(package = "hstats", version = as.character(packageVersion("hstats")),
                        official_code_used = TRUE, adapter_only = TRUE),
      discovery_only = TRUE,
      wall_seconds = proc.time()[["elapsed"]] - started
    )
  } else if (method == "IADT") {
    suppressPackageStartupMessages(library(IADT))
    reference <- read.csv(file.path(output_dir, "reference.csv"), check.names = FALSE)
    evaluation <- read.csv(file.path(output_dir, "evaluation.csv"), check.names = FALSE)
    candidates <- meta$candidate_pairs
    rows <- vector("list", length(candidates))
    pvalues <- numeric(length(candidates))
    for (idx in seq_along(candidates)) {
      pair <- as.integer(unlist(candidates[[idx]]))
      test <- testIAD_mpfr(
        colInd = pair + 1L,
        object = model,
        predictfun = predict_fun,
        X = reference,
        newX = evaluation,
        alternative = "twoSided",
        centering = FALSE,
        nCoresPDP = 1L,
        precBits = 106L
      )
      pvalues[[idx]] <- as.numeric(test$pValue)
      rows[[idx]] <- list(pair = pair, pvalue = pvalues[[idx]],
                          test_statistic = as.numeric(test$testStat),
                          interaction_difference = as.numeric(test$IAD))
    }
    adjusted <- p.adjust(pvalues, method = "BH")
    rejected <- adjusted <= 0.05
    for (idx in seq_along(rows)) {
      rows[[idx]]$bh_adjusted_pvalue <- adjusted[[idx]]
      rows[[idx]]$rejected_bh_0_05 <- rejected[[idx]]
      rows[[idx]]$is_syntactic_true_pair <- paste(sort(rows[[idx]]$pair), collapse = "_") %in% truth_keys
    }
    true_mask <- vapply(rows, function(x) x$is_syntactic_true_pair, logical(1))
    result <- list(
      schema_version = "lens-official-stage2-iadt-cell-1",
      status = "DONE", stage = 2L, method = "IADT-official",
      dgp = meta$dgp, seed = as.integer(meta$seed),
      candidate_results = rows,
      metrics = list(
        candidate_true_rejection_rate = if (any(true_mask)) mean(rejected[true_mask]) else NULL,
        candidate_null_rejection_rate = if (any(!true_mask)) mean(rejected[!true_mask]) else NULL,
        discoveries = sum(rejected),
        true_discoveries_under_syntactic_label = sum(rejected & true_mask),
        false_discoveries_under_syntactic_label = sum(rejected & !true_mask)
      ),
      method_config = list(reference_n = nrow(reference), evaluation_n = nrow(evaluation),
                           alternative = "twoSided", precBits = 106L, nCoresPDP = 1L),
      native_null_scope = meta$native_null_scope,
      provenance = list(package = "IADT", version = as.character(packageVersion("IADT")),
                        official_code_used = TRUE, adapter_only = TRUE),
      wall_seconds = proc.time()[["elapsed"]] - started
    )
  } else stop(paste("unknown method", method))

  atomic_json(file.path(output_dir, "result.json"), result)
  file.create(file.path(output_dir, "DONE"))
  cat(sprintf("OFFICIAL_R_CELL_DONE method=%s dgp=%s seed=%d seconds=%.3f\n",
              method, meta$dgp, as.integer(meta$seed), result$wall_seconds))
}, error = function(error) {
  atomic_json(file.path(output_dir, "FAILED_RUN.json"), list(
    status = "FAILED", error = conditionMessage(error),
    calls = vapply(sys.calls(), deparse1, character(1)),
    wall_seconds = proc.time()[["elapsed"]] - started
  ))
  stop(error)
})
