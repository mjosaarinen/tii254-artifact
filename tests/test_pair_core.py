import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tii254.binary_field import BinaryField
from tii254.pair_core import recover_direct_pair_core_graph_pencil


def decode(values):
    return tuple(int(value, 16) for value in values)


class PairCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        record = json.loads((ROOT / "data/pair_core.json").read_text())
        cls.result = recover_direct_pair_core_graph_pencil(
            observer_labels=record["observer_labels"],
            physical_sum_representatives=decode(
                record["physical_sum_representatives_hex"]
            ),
            pair_core_basis=decode(record["pair_core_basis_hex"]),
            common_physical_basis=decode(record["common_physical_basis_hex"]),
            core_local_kernel_bases=tuple(
                decode(rows) for rows in record["core_local_kernel_bases_hex"]
            ),
            field=BinaryField(record["field_degree"], record["field_modulus"]),
            locator_power_rounds=record["locator_power_rounds"],
        )

    def test_dimensions(self):
        self.assertEqual(
            (
                self.result.physical_sum_dimension,
                self.result.pair_core_dimension,
                self.result.common_nuisance_dimension,
                self.result.quotient_dimension,
            ),
            (122, 80, 64, 16),
        )
        self.assertEqual(set(self.result.quotient_local_kernel_dimensions), {8})
        self.assertEqual(set(self.result.pairwise_quotient_sum_dimensions), {16})

    def test_locator_branches(self):
        locators = self.result.locators
        self.assertEqual(locators.branch_count, 8)
        self.assertTrue(locators.full_frobenius_orbit)
        self.assertTrue(locators.all_kernels_replay)
        self.assertEqual(
            locators.projective_locators_by_branch[0][:4],
            ((0, 1), (1, 0), (1, 1), (226, 1)),
        )


if __name__ == "__main__":
    unittest.main()
