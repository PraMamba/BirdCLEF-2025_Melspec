import pandas as pd
import numpy as np
import sklearn

def padded_cmap(solution, submission, padding_factor=5):
    solution = solution  # .drop(['row_id'], axis=1, errors='ignore')
    submission = submission  # .drop(['row_id'], axis=1, errors='ignore')

    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = sklearn.metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average="macro",
    )
    return score


def map_score(solution, submission):
    solution = solution  # .drop(['row_id'], axis=1, errors='ignore')
    submission = submission  # .drop(['row_id'], axis=1, errors='ignore')
    score = sklearn.metrics.average_precision_score(
        solution.values,
        submission.values,
        average="micro",
    )
    return score

def roc_score(solution, submission):
    score_birds = {}
    for b in solution.columns:
        if solution[b].sum() < 1:
            score_birds[b] = np.nan
        else:
            score = sklearn.metrics.roc_auc_score(solution[b], submission[b])
            score_birds[b] = score

    score_birds = pd.Series(score_birds)
    return np.nanmean(score_birds)
