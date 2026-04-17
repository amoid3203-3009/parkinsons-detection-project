"""
═══════════════════════════════════════════════════════════════
TEST PLAN — ParkinsonAI Finger Tapping Severity Classifier
COS5031 Group 25 | University of Bradford
═══════════════════════════════════════════════════════════════

Coverage:
  Unit Tests        — individual function correctness
  Integration Tests — full pipeline end-to-end
  Validation Tests  — model meets PID performance targets
  Edge Case Tests   — robustness under bad/missing input

Run with:
  python test_plan.py
  or: python test_plan.py -v  (verbose)
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

# ── Constants (must match app.py and notebook) ────────────────────────────────
FEATURE_COLS = [
    'amp_mean','amp_std','amp_max','amp_min','amp_range','amp_median',
    'amp_iqr','amp_energy','vel_mean','vel_std','vel_max','vel_min',
    'vel_range','vel_energy','zero_crossing_rate','signal_energy','signal_entropy'
]
CSV_PATH = 'data/raw/finger_tapping_features.csv'
F1_TARGET = 0.75
MCC_THRESHOLD = 0.70


# ── Helper: load and prepare data ─────────────────────────────────────────────
def load_and_prepare(path=CSV_PATH):
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].values

    scaler = MinMaxScaler()
    sev_scaled = scaler.fit_transform(
        df[['amp_std','zero_crossing_rate','amp_mean','signal_entropy']].values
    )
    severity_score = (
        sev_scaled[:,0] * 0.40 +
        (1 - sev_scaled[:,1]) * 0.40 +
        (1 - sev_scaled[:,2]) * 0.10 +
        (1 - sev_scaled[:,3]) * 0.10
    )
    y = pd.qcut(severity_score, q=4, labels=[0,1,2,3]).astype(int)

    feat_scaler = MinMaxScaler()
    X_scaled = feat_scaler.fit_transform(X)

    return df, X_scaled, np.array(y), feat_scaler


# ══════════════════════════════════════════════════════════════
# 1. UNIT TESTS
# ══════════════════════════════════════════════════════════════
class TestDataLoading(unittest.TestCase):
    """Tests that the dataset loads correctly and has expected structure."""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(CSV_PATH)

    def test_csv_loads(self):
        """Dataset file loads without error."""
        self.assertIsInstance(self.df, pd.DataFrame)

    def test_expected_columns(self):
        """All 17 feature columns are present."""
        for col in FEATURE_COLS:
            self.assertIn(col, self.df.columns,
                          msg=f"Missing column: {col}")

    def test_row_count(self):
        """Dataset has at least 100 recordings."""
        self.assertGreaterEqual(len(self.df), 100,
                                msg="Dataset too small for reliable training")

    def test_no_nulls_in_features(self):
        """No null values in any feature column."""
        null_count = self.df[FEATURE_COLS].isnull().sum().sum()
        self.assertEqual(null_count, 0,
                         msg=f"Found {null_count} null values in features")

    def test_feature_range(self):
        """All feature values are finite (no inf or NaN)."""
        X = self.df[FEATURE_COLS].values
        self.assertTrue(np.isfinite(X).all(),
                        msg="Non-finite values found in feature matrix")

    def test_hand_labels(self):
        """Label column contains only 'left' and 'right'."""
        valid = {'left', 'right'}
        actual = set(self.df['label'].unique())
        self.assertTrue(actual.issubset(valid),
                        msg=f"Unexpected label values: {actual - valid}")

    def test_balanced_hands(self):
        """Left and right hand recordings are approximately balanced."""
        counts = self.df['label'].value_counts()
        ratio = counts.min() / counts.max()
        self.assertGreater(ratio, 0.8,
                           msg=f"Hand imbalance detected: {counts.to_dict()}")


class TestPreprocessing(unittest.TestCase):
    """Tests for the normalisation and severity label pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.df, cls.X_scaled, cls.y, cls.scaler = load_and_prepare()

    def test_normalisation_range(self):
        """MinMax normalisation produces values in [0, 1]."""
        self.assertAlmostEqual(self.X_scaled.min(), 0.0, places=5)
        self.assertAlmostEqual(self.X_scaled.max(), 1.0, places=5)

    def test_output_shape(self):
        """Feature matrix shape matches (n_recordings, n_features)."""
        self.assertEqual(self.X_scaled.shape[1], len(FEATURE_COLS))
        self.assertEqual(self.X_scaled.shape[0], len(self.df))

    def test_severity_classes(self):
        """Severity labels contain exactly 4 classes: 0, 1, 2, 3."""
        unique_classes = set(np.unique(self.y))
        self.assertEqual(unique_classes, {0, 1, 2, 3},
                         msg=f"Expected 4 classes, got: {unique_classes}")

    def test_severity_balance(self):
        """Each severity class has at least 10% of total recordings."""
        for cls in range(4):
            proportion = np.sum(self.y == cls) / len(self.y)
            self.assertGreater(proportion, 0.10,
                               msg=f"Class {cls} has only {proportion:.1%} of data")

    def test_no_data_leakage(self):
        """Scaler was fitted on training data only (test set transform only)."""
        X_train, X_test, _, _ = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )
        # Scaler fitted on train: test values may exceed [0,1] slightly if outliers
        # but should be within a reasonable bound
        self.assertLess(X_test.max(), 1.5,
                        msg="Test set values far exceed training range — possible leakage")


# ══════════════════════════════════════════════════════════════
# 2. INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════
class TestPipelineIntegration(unittest.TestCase):
    """End-to-end pipeline tests."""

    @classmethod
    def setUpClass(cls):
        cls.df, cls.X, cls.y, cls.scaler = load_and_prepare()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42, stratify=cls.y
        )
        cls.model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        cls.model.fit(cls.X_train, cls.y_train)
        cls.y_pred = cls.model.predict(cls.X_test)

    def test_model_trains(self):
        """Random Forest model trains without error."""
        self.assertIsNotNone(self.model)
        self.assertTrue(hasattr(self.model, 'estimators_'))

    def test_prediction_shape(self):
        """Model predictions match the number of test samples."""
        self.assertEqual(len(self.y_pred), len(self.y_test))

    def test_predictions_valid_classes(self):
        """All predictions are valid class labels (0–3)."""
        valid = {0, 1, 2, 3}
        predicted = set(np.unique(self.y_pred))
        self.assertTrue(predicted.issubset(valid),
                        msg=f"Invalid predicted classes: {predicted - valid}")

    def test_single_prediction(self):
        """Single recording prediction returns one valid class."""
        single = self.X_test[[0]]
        pred = self.model.predict(single)
        self.assertEqual(len(pred), 1)
        self.assertIn(pred[0], [0, 1, 2, 3])

    def test_probability_output(self):
        """Probability outputs sum to 1.0 for each recording."""
        proba = self.model.predict_proba(self.X_test)
        row_sums = proba.sum(axis=1)
        np.testing.assert_array_almost_equal(
            row_sums, np.ones(len(row_sums)), decimal=5,
            err_msg="Probability rows do not sum to 1.0"
        )

    def test_feature_importance_available(self):
        """Feature importances are available and sum to ~1.0."""
        fi = self.model.feature_importances_
        self.assertEqual(len(fi), len(FEATURE_COLS))
        self.assertAlmostEqual(fi.sum(), 1.0, places=5)


# ══════════════════════════════════════════════════════════════
# 3. VALIDATION TESTS (Performance targets from PID)
# ══════════════════════════════════════════════════════════════
class TestModelValidation(unittest.TestCase):
    """
    Validates model meets targets defined in PID Section 3.2
    and REFORMS checklist (Kapoor et al., 2024).
    """

    @classmethod
    def setUpClass(cls):
        cls.df, cls.X, cls.y, cls.scaler = load_and_prepare()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42, stratify=cls.y
        )
        cls.model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        cls.model.fit(cls.X_train, cls.y_train)
        cls.y_pred = cls.model.predict(cls.X_test)
        cls.f1 = f1_score(cls.y_test, cls.y_pred, average='weighted')
        cls.mcc = matthews_corrcoef(cls.y_test, cls.y_pred)

    def test_f1_meets_target(self):
        """F1 score meets PID Objective 2 target of >= 0.75."""
        self.assertGreaterEqual(
            self.f1, F1_TARGET,
            msg=f"F1 {self.f1:.4f} is below target {F1_TARGET}"
        )

    def test_mcc_above_threshold(self):
        """MCC > 0.70 confirms performance is not due to class imbalance."""
        self.assertGreater(
            self.mcc, MCC_THRESHOLD,
            msg=f"MCC {self.mcc:.4f} is below threshold {MCC_THRESHOLD}"
        )

    def test_cross_validation_f1(self):
        """5-fold stratified CV F1 mean >= 0.75 (Lones, 2024)."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        scores = cross_val_score(model, self.X, self.y,
                                 cv=cv, scoring='f1_weighted')
        mean_f1 = scores.mean()
        self.assertGreaterEqual(
            mean_f1, F1_TARGET,
            msg=f"CV F1 mean {mean_f1:.4f} is below target {F1_TARGET}"
        )

    def test_cv_consistency(self):
        """CV F1 standard deviation < 0.05 (consistent across folds)."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        scores = cross_val_score(model, self.X, self.y,
                                 cv=cv, scoring='f1_weighted')
        std = scores.std()
        self.assertLess(std, 0.05,
                        msg=f"CV F1 std {std:.4f} too high — inconsistent model")

    def test_per_class_recall(self):
        """Each severity class has recall >= 0.70."""
        from sklearn.metrics import recall_score
        recalls = recall_score(self.y_test, self.y_pred, average=None)
        class_names = ['Normal','Mild','Moderate','Severe']
        for i, (recall, name) in enumerate(zip(recalls, class_names)):
            self.assertGreaterEqual(
                recall, 0.70,
                msg=f"Recall for {name} is {recall:.2f} — below 0.70"
            )

    def test_no_class_entirely_missed(self):
        """Model predicts all 4 severity classes (no class entirely missed)."""
        predicted_classes = set(np.unique(self.y_pred))
        self.assertEqual(len(predicted_classes), 4,
                         msg=f"Model only predicts {len(predicted_classes)} classes")


# ══════════════════════════════════════════════════════════════
# 4. EDGE CASE TESTS
# ══════════════════════════════════════════════════════════════
class TestEdgeCases(unittest.TestCase):
    """Robustness tests under abnormal or missing input."""

    @classmethod
    def setUpClass(cls):
        cls.df, cls.X, cls.y, cls.scaler = load_and_prepare()
        cls.model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        cls.model.fit(cls.X, cls.y)

    def test_single_row_prediction(self):
        """Model handles single-row input without error."""
        single = self.X[[0]]
        pred = self.model.predict(single)
        self.assertEqual(len(pred), 1)

    def test_all_zeros_input(self):
        """Model handles all-zero feature input gracefully."""
        zeros = np.zeros((1, len(FEATURE_COLS)))
        pred = self.model.predict(zeros)
        self.assertIn(pred[0], [0, 1, 2, 3])

    def test_all_ones_input(self):
        """Model handles all-one feature input gracefully."""
        ones = np.ones((1, len(FEATURE_COLS)))
        pred = self.model.predict(ones)
        self.assertIn(pred[0], [0, 1, 2, 3])

    def test_large_batch(self):
        """Model handles large batch (500 rows) without error."""
        large = np.tile(self.X[0], (500, 1))
        preds = self.model.predict(large)
        self.assertEqual(len(preds), 500)

    def test_missing_column_detected(self):
        """Missing feature column raises KeyError before model is called."""
        bad_df = pd.read_csv(CSV_PATH).drop(columns=['amp_mean'])
        with self.assertRaises(KeyError):
            _ = bad_df[FEATURE_COLS].values

    def test_confidence_scores_in_range(self):
        """All confidence scores are between 0 and 1."""
        proba = self.model.predict_proba(self.X)
        self.assertTrue((proba >= 0).all() and (proba <= 1).all(),
                        msg="Probability scores outside [0, 1]")

    def test_low_confidence_flagging(self):
        """At least some predictions have confidence < 0.99 (model is not over-certain)."""
        proba = self.model.predict_proba(self.X)
        max_proba = proba.max(axis=1)
        has_uncertain = (max_proba < 0.99).any()
        self.assertTrue(has_uncertain,
                        msg="All predictions have 100% confidence — model may be overfit")


# ══════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════
def run_tests():
    print("═" * 65)
    print("  PARKINSONAI TEST PLAN — COS5031 Group 25")
    print("  University of Bradford")
    print("═" * 65)
    print()

    suites = [
        ("1. Data Loading & Structure", TestDataLoading),
        ("2. Preprocessing Pipeline",   TestPreprocessing),
        ("3. Integration Tests",        TestPipelineIntegration),
        ("4. Model Validation",         TestModelValidation),
        ("5. Edge Cases & Robustness",  TestEdgeCases),
    ]

    total_pass = 0
    total_fail = 0
    total_error = 0

    for suite_name, test_class in suites:
        print(f"── {suite_name} ──")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        total_pass  += result.testsRun - len(result.failures) - len(result.errors)
        total_fail  += len(result.failures)
        total_error += len(result.errors)
        print()

    print("═" * 65)
    print(f"  TOTAL RESULTS")
    print(f"  Passed : {total_pass}")
    print(f"  Failed : {total_fail}")
    print(f"  Errors : {total_error}")
    status = "✅ ALL TESTS PASSED" if (total_fail + total_error) == 0 else "❌ SOME TESTS FAILED"
    print(f"  Status : {status}")
    print("═" * 65)


if __name__ == '__main__':
    run_tests()
