"""Tests for the table generator's variant grouping.

The tables are the artefact a reader trusts, so the one bug that matters here is
silently averaging a post-paper ablation into the baseline row: that would report a
mean for a configuration nobody ran.  ``run_experiment.py`` encodes every
non-default knob in the run id, and ``variant_of`` must turn that back into a
label that keeps the rows apart.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reproduction", "scripts",
))

from make_tables import variant_of  # noqa: E402
from run_experiment import RUN_ID_EXEMPT, RUN_ID_KNOBS, run_id_for  # noqa: E402


class Options(object):
    """Stand-in for the argparse namespace, with the driver's defaults."""

    def __init__(self, **overrides):
        self.dataset = "cora"
        self.partition = "ccp"
        self.model = "GCN"
        self.seed = 4
        self.unlearn_ratio = 0.005
        self.ccp_protect_eval_nodes = "none"
        for name, default, _label in RUN_ID_KNOBS:
            setattr(self, name, default)
        for key, value in overrides.items():
            setattr(self, key, value)


class VariantLabelTest(unittest.TestCase):
    def test_baseline_run_id_is_labelled_baseline(self):
        run_id = run_id_for(Options())
        self.assertEqual(variant_of(run_id, "ccp"), "baseline")

    def test_holdout_alone_is_still_baseline(self):
        run_id = run_id_for(Options(ccp_protect_eval_nodes="test"))
        self.assertIn("holdtest", run_id)
        self.assertEqual(variant_of(run_id, "ccp"), "baseline",
                         "the hold-out protocol has its own column already")

    def test_tail_repair_is_a_distinct_variant(self):
        run_id = run_id_for(Options(ccp_protect_eval_nodes="test",
                                    ccp_tail_min_size=5))
        self.assertNotEqual(variant_of(run_id, "ccp"), "baseline")
        self.assertIn("tail=5", variant_of(run_id, "ccp"))

    def test_theta_and_cap_are_distinct_variants(self):
        theta = variant_of(run_id_for(Options(ccp_theta=10)), "ccp")
        cap = variant_of(run_id_for(Options(ccp_max_community_size=40)), "ccp")
        self.assertIn("theta=10", theta)
        self.assertIn("cap=40", cap)
        self.assertNotEqual(theta, cap)

    def test_gae_is_always_a_distinct_variant(self):
        run_id = run_id_for(Options(partition="gae"))
        self.assertIn("detector=gae", variant_of(run_id, "gae"))

    def test_every_ablation_in_run_all_gets_its_own_row(self):
        """The exact configurations run_all.sh sweeps must not collapse together."""
        configurations = [
            Options(ccp_protect_eval_nodes="test"),                        # baseline
            Options(ccp_protect_eval_nodes="test", ccp_tail_min_size=5),
            Options(ccp_protect_eval_nodes="test", ccp_theta=10),
            Options(ccp_protect_eval_nodes="test", ccp_theta=40),
            Options(ccp_protect_eval_nodes="test", ccp_max_community_size=40),
            Options(ccp_protect_eval_nodes="test", partition="gae"),
            Options(ccp_protect_eval_nodes="test", partition="gae",
                    ccp_tail_min_size=5),
            Options(ccp_protect_eval_nodes="test", agg_feat="mean"),
        ]
        run_ids = [run_id_for(options) for options in configurations]
        self.assertEqual(len(set(run_ids)), len(run_ids),
                         "two ablation cells share a run id: {}".format(run_ids))

        # agg_feat has its own column, so it is allowed to share the variant label
        # with the baseline; everything else must be distinguishable by
        # (variant, partition, agg_feat) together.
        keys = [
            (variant_of(run_id, options.partition), options.partition,
             options.agg_feat)
            for run_id, options in zip(run_ids, configurations)
        ]
        self.assertEqual(len(set(keys)), len(keys),
                         "two ablation cells share a table row: {}".format(keys))


class RunIdCoverageTest(unittest.TestCase):
    def test_knob_and_exempt_sets_do_not_overlap(self):
        knobs = {name for name, _default, _label in RUN_ID_KNOBS}
        self.assertEqual(knobs & RUN_ID_EXEMPT, set())

    def test_labels_are_unique(self):
        labels = [label for _name, _default, label in RUN_ID_KNOBS]
        self.assertEqual(len(set(labels)), len(labels))

    def test_changing_any_knob_changes_the_run_id(self):
        baseline = run_id_for(Options())
        for name, default, _label in RUN_ID_KNOBS:
            if isinstance(default, bool):
                changed = not default
            elif isinstance(default, (int, float)):
                changed = default + 7
            else:
                changed = "cpu" if name == "gae_device" else "mean"
            other = run_id_for(Options(**{name: changed}))
            self.assertNotEqual(baseline, other,
                               "{} does not affect the run id".format(name))


if __name__ == "__main__":
    unittest.main()
