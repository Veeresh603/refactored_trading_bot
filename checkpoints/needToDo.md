Add the purged_cv_folds_train_eval_ep{ep}.json files to your training best_agent checkpoint metadata (one JSON per checkpoint or embedded in the checkpoint directory).

Add automatic experiment tagging (timestamp + git SHA) in saved metadata to make audits reproducible.

Add a small scripts/plot_purged_cv.py that reads the saved JSON files and plots fold scores over training epochs.