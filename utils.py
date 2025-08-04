import transformer_lens
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd
import json
import re
from typing import List, Dict, Union

def get_entropy(logits, eps=1e-20, softmaxed=False):
    if not softmaxed:
        logits = F.softmax(logits, dim=-1)
    return -torch.sum(logits * torch.log(logits+eps), dim=-1)

def get_renyi_entropy(logits, alpha, eps=1e-20, softmaxed=False):
    if not softmaxed:
        logits = F.softmax(logits, dim=-1)

    if alpha == 0:
        support = (logits > eps).float()
        return torch.log(torch.sum(support, dim=-1) + eps)
    elif alpha == 1:
        return -torch.sum(logits * torch.log(logits+eps), dim=-1)
    elif alpha == np.inf:
        return -torch.log(torch.max(logits, dim=-1).values + eps)
    else:
        return torch.log(torch.sum(logits ** alpha, dim=-1) + eps) / (1 - alpha)


def get_moment(logits, softmaxed=False, power=1):
    """
    Calcola la media dei logits elevati a una potenza specificata.
    
    Args:
        logits (torch.Tensor): Il tensore dei logits.
        softmaxed (bool): Se False, applica softmax prima del calcolo. Default: False.
        power (int): La potenza a cui elevare i logits prima di calcolare la media. Default: 1.
    
    Returns:
        torch.Tensor: Media dei logits elevati alla potenza specificata.
    """
    if not softmaxed:
        logits = F.softmax(logits, dim=-1)
    
    return torch.mean(logits ** power, dim=-1)

def kl_divergence(logits1, logits2, eps=1e-20, softmaxed=False):
    if not softmaxed:
        probs1 = F.softmax(logits1, dim=-1)
        probs2 = F.softmax(logits2, dim=-1)
    else:
        probs1 = logits1
        probs2 = logits2
    log_probs1 = torch.log(probs1 + eps)
    log_probs2 = torch.log(probs2 + eps)
    return torch.sum(probs1 * (log_probs1 - log_probs2), dim=-1)


def extract_answer(text: str) -> Union[str, None]:
    """
    Extract answer letter (A, B, C, or D) from various response formats.
    
    Examples of valid formats:
    - "A"
    - "Answer: B"
    - "The answer is C"
    - "I think D is correct"
    - "The correct answer would be A"
    
    Args:
        text: Raw response text from the model
        
    Returns:
        Extracted answer letter or None if no valid answer found
    """
    # Convert to uppercase and strip whitespace
    text = text.upper().strip()
    
    # If text is already just A, B, C, or D
    if text in ['A', 'B', 'C', 'D']:
        return text
    
    # Comprehensive patterns for answer extraction
    patterns = [
        # Basic patterns
        r'ANSWER:\s*([A-D])',  # "Answer: A"
        r'ANSWER IS\s*([A-D])',  # "Answer is B"
        r'THE ANSWER IS\s*([A-D])',  # "The answer is C"
        
        # Patterns with "correct"
        r'CORRECT ANSWER IS\s*([A-D])',  # "Correct answer is D"
        r'THE CORRECT ANSWER IS\s*([A-D])',  # "The correct answer is A"
        r'CORRECT ANSWER:\s*([A-D])',  # "Correct answer: B"
        r'([A-D])\s*IS THE CORRECT ANSWER',  # "C is the correct answer"
        r'([A-D])\s*IS CORRECT',  # "D is correct"
        
        # Patterns with "choose/choice/select"
        r'I CHOOSE\s*([A-D])',  # "I choose A"
        r'I WOULD CHOOSE\s*([A-D])',  # "I would choose B"
        r'CHOICE\s*([A-D])',  # "Choice C"
        r'CHOICE IS\s*([A-D])',  # "Choice is D"
        r'THE CHOICE IS\s*([A-D])',  # "The choice is A"
        r'I SELECT\s*([A-D])',  # "I select B"
        r'SELECTING\s*([A-D])',  # "Selecting C"
        
        # Patterns with "option"
        r'OPTION\s*([A-D])',  # "Option D"
        r'OPTION:\s*([A-D])',  # "Option: A"
        r'THE OPTION IS\s*([A-D])',  # "The option is B"
        
        # Patterns with "think/believe/conclude"
        r'I THINK\s*([A-D])',  # "I think C"
        r'I BELIEVE\s*([A-D])',  # "I believe D"
        r'I CONCLUDE\s*([A-D])',  # "I conclude A"
        
        # Patterns with "would be"
        r'WOULD BE\s*([A-D])',  # "would be B"
        r'SHOULD BE\s*([A-D])',  # "should be C"
        r'MUST BE\s*([A-D])',  # "must be D"
        
        # Response-style patterns
        r'RESPONSE:\s*([A-D])',  # "Response: A"
        r'MY RESPONSE IS\s*([A-D])',  # "My response is B"
        r'FINAL ANSWER:\s*([A-D])',  # "Final answer: C"
        r'SOLUTION:\s*([A-D])',  # "Solution: D"
        
        # Confidence-based patterns
        r'I AM CONFIDENT\s*([A-D])',  # "I am confident A"
        r'DEFINITELY\s*([A-D])',  # "Definitely B"
        r'CERTAINLY\s*([A-D])',  # "Certainly C"
        
        # Letter-based patterns
        r'LETTER\s*([A-D])',  # "Letter D"
        r'([A-D])\s*IS THE LETTER',  # "A is the letter"
        
        # Natural language patterns
        r'LET\'S GO WITH\s*([A-D])',  # "Let's go with B"
        r'I\'M GOING WITH\s*([A-D])',  # "I'm going with C"
        r'I PICK\s*([A-D])',  # "I pick D"
        r'PICKING\s*([A-D])',  # "Picking A"
        
        # Reasoning patterns
        r'THEREFORE,?\s*([A-D])',  # "Therefore, B"
        r'THUS,?\s*([A-D])',  # "Thus, C"
        r'SO,?\s*([A-D])',  # "So, D"
        
        # Based on patterns
        r'BASED ON.*,\s*([A-D])',  # "Based on the information, A"
        r'GIVEN.*,\s*([A-D])',  # "Given the context, B"
        
        # Mathematical notation patterns
        r'=\s*([A-D])',  # "= C"
        r'→\s*([A-D])',  # "→ D"
        
        # Parenthetical patterns
        r'\(([A-D])\)',  # "(A)"
        r'\[([A-D])\]',  # "[B]"

        # uses ** before the letter
        r'\*\*([A-D])',  # "**A"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    # If no matches found, look for standalone A, B, C, or D
    # but only if it appears exactly once to avoid ambiguity
    for answer in ['A', 'B', 'C', 'D']:
        if text.count(answer) == 1:
            return answer
            
    return None

def checker(pred, true):
    match = ['A', 'B', 'C', 'D']
    pred = extract_answer(pred)
    if pred == match[true]:
        return 1
    else:
        return 0

def evaluate_model_by_subject(X, y, subjects, clf=None, n_splits=10, random_state=42):
    """
    Evaluate a classifier across different subjects using ROC-AUC and statistical tests.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    subjects : array-like
        Subject identifiers for each sample
    clf : estimator object, default=None
        A scikit-learn classifier that implements fit, predict, and predict_proba.
        If None, uses KNeighborsClassifier(n_neighbors=25, metric='euclidean', weights='distance')
    n_splits : int, default=10
        Number of splits for cross-validation
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (results_df, anova_df) containing the main results DataFrame and
        ANOVA test results
    """
    from collections import defaultdict
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.dummy import DummyClassifier
    from scipy import stats
    
    # Use default classifier if none provided
    if clf is None:
        clf = KNeighborsClassifier(n_neighbors=25, metric='euclidean', weights='distance')
    
    # Initialize cross-validation
    fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize dictionaries for tracking metrics
    auc_roc_by_subject = defaultdict(list)
    baseline_auc_by_subject = defaultdict(list)
    f1_scores_by_subject = defaultdict(list)
    counts_total = defaultdict(int)
    
    # Perform cross-validation
    for train_index, test_index in fold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        subject_test = np.array(subjects)[test_index]
        
        # Fit the actual model
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        
        # Fit the baseline model (stratified random)
        baseline = DummyClassifier(strategy='stratified', random_state=random_state)
        baseline.fit(X_train, y_train)
        y_pred_baseline = baseline.predict(X_test)
        y_prob_baseline = baseline.predict_proba(X_test)
        
        # Evaluate per subject
        for subject in np.unique(subject_test):
            sub_idx = (subject_test == subject)
            y_true_sub = y_test[sub_idx]
            y_pred_sub = y_pred[sub_idx]
            y_prob_sub = y_prob[sub_idx]
            y_pred_baseline_sub = y_pred_baseline[sub_idx]
            y_prob_baseline_sub = y_prob_baseline[sub_idx]
            
            # Skip if only one class is present
            if len(np.unique(y_true_sub)) < 2:
                continue
                
            # Store F1 scores for reference
            f1_scores_by_subject[subject].append(f1_score(y_true_sub, y_pred_sub, average='macro'))
            
            # Store AUC-ROC for both model and baseline
            if len(np.unique(y_true_sub)) == 2:
                # Binary classification
                auc_roc_by_subject[subject].append(roc_auc_score(y_true_sub, y_prob_sub[:, 1]))
                baseline_auc_by_subject[subject].append(roc_auc_score(y_true_sub, y_prob_baseline_sub[:, 1]))
            else:
                # Multi-class classification
                auc_roc_by_subject[subject].append(roc_auc_score(y_true_sub, y_prob_sub, multi_class='ovr', average='macro'))
                baseline_auc_by_subject[subject].append(roc_auc_score(y_true_sub, y_prob_baseline_sub, multi_class='ovr', average='macro'))
            
            counts_total[subject] += sub_idx.sum()
    
    # Prepare results DataFrame
    results_data = []
    for subject in auc_roc_by_subject:
        # Calculate average metrics
        avg_auc = np.mean(auc_roc_by_subject[subject])
        avg_baseline_auc = np.mean(baseline_auc_by_subject[subject])
        avg_f1 = np.mean(f1_scores_by_subject[subject])
        
        # Perform t-test to compare model vs baseline
        t_stat, p_value = stats.ttest_rel(auc_roc_by_subject[subject], 
                                          baseline_auc_by_subject[subject])
        
        # Store results
        subject_results = {
            'subject': subject,
            'sample_count': counts_total[subject],
            'roc_auc': avg_auc,
            'baseline_roc_auc': avg_baseline_auc,
            'improvement': avg_auc - avg_baseline_auc,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'f1_score': avg_f1
        }
        results_data.append(subject_results)
    
    # Create and sort DataFrame
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values(by='roc_auc', ascending=False)
    
    
    return results_df


def evaluate_model_overall(X, y, clf=None, n_splits=10, random_state=42):
    """
    Evaluate a classifier's performance against a baseline dummy classifier using ROC-AUC,
    without dividing by subject.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    clf : estimator object, default=None
        A scikit-learn classifier that implements fit, predict, and predict_proba.
        If None, uses KNeighborsClassifier(n_neighbors=25, metric='euclidean', weights='distance')
    n_splits : int, default=10
        Number of splits for cross-validation
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing the evaluation results including average ROC-AUC for both models,
        improvement, p-value, and significance indicator
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.dummy import DummyClassifier
    from scipy import stats
    
    # Use default classifier if none provided
    if clf is None:
        clf = KNeighborsClassifier(n_neighbors=29, metric='euclidean', n_jobs=-1)
    
    # Initialize cross-validation
    fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize lists for tracking metrics across folds
    model_auc_scores = []
    baseline_auc_scores = []
    
    # Perform cross-validation
    for train_index, test_index in fold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Fit the actual model
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)
        
        # Fit the baseline model (stratified random)
        baseline = DummyClassifier(strategy='stratified', random_state=random_state)
        baseline.fit(X_train, y_train)
        y_prob_baseline = baseline.predict_proba(X_test)
        
        # Calculate and store AUC-ROC for both model and baseline
        if len(np.unique(y_test)) == 2:
            # Binary classification
            model_auc_scores.append(roc_auc_score(y_test, y_prob[:, 1]))
            baseline_auc_scores.append(roc_auc_score(y_test, y_prob_baseline[:, 1]))
        else:
            # Multi-class classification
            model_auc_scores.append(roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro'))
            baseline_auc_scores.append(roc_auc_score(y_test, y_prob_baseline, multi_class='ovr', average='macro'))
    
    # Calculate average metrics
    avg_model_auc = np.mean(model_auc_scores)
    avg_baseline_auc = np.mean(baseline_auc_scores)
    improvement = avg_model_auc - avg_baseline_auc
    
    # Perform t-test to compare model vs baseline
    t_stat, p_value = stats.ttest_rel(model_auc_scores, baseline_auc_scores)
    
    # Create results dictionary
    results = {
        'model_roc_auc': avg_model_auc,
        'baseline_roc_auc': avg_baseline_auc,
        'improvement': improvement,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'model_auc_by_fold': model_auc_scores,
        'baseline_auc_by_fold': baseline_auc_scores
    }
    
    return pd.DataFrame(results)

def get_entropy_thresholded(logits, eps=1e-20, th=None, softmaxed=False):
    if not softmaxed:
        logits = F.softmax(logits, dim=-1)

    if not th:
        th = logits.mean() - torch.sqrt(logits.var())
    
    mask = logits > th

    entropy_list = []

    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):

            slice_tensor = logits[i, j, :]
            valid_elements = slice_tensor[mask[i, j, :]]

            if len(valid_elements) > 0:
                entropy = -torch.sum(valid_elements * torch.log(valid_elements))
            else:
                entropy = torch.tensor(0.0)

            entropy_list.append(entropy)

    entropy_tensor = torch.stack(entropy_list).reshape(logits.shape[0], logits.shape[1])

    return entropy_tensor

def get_moment_thresholded(logits, softmaxed=False, power=1, th=None):
    if not softmaxed:
        logits = F.softmax(logits, dim=-1)

    if not th:
        th = logits.mean() - torch.sqrt(logits.var())
    
    mask = logits > th

    moment_list = []

    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):

            slice_tensor = logits[i, j, :]
            valid_elements = slice_tensor[mask[i, j, :]]

            if len(valid_elements) > 0:
                moment = torch.mean(valid_elements ** power)
            else:
                moment = torch.tensor(0.0)

            moment_list.append(moment)

    moment_tensor = torch.stack(moment_list).reshape(logits.shape[0], logits.shape[1])

    return moment_tensor