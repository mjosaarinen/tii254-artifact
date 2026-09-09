import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CadoAdapterTest(unittest.TestCase):
    def test_extracts_reversed_right_generator(self):
        dimension = 128
        coefficients = []
        constant = [0] * dimension
        for column in range(64):
            constant[64 + column] = (1 << column) | (1 << (64 + column))
        coefficients.append(constant)
        linear = [0] * dimension
        for column in range(64):
            linear[column] = 1 << column
        coefficients.append(linear)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pi = directory / "basis.pi"
            relations = directory / "relations.bin"
            with pi.open("wb") as stream:
                for coefficient in coefficients:
                    for row in coefficient:
                        stream.write(
                            struct.pack(
                                "<QQ", row & ((1 << 64) - 1), row >> 64
                            )
                        )
            completed = subprocess.run(
                [
                    ROOT / "build/extract-relations",
                    pi,
                    "64",
                    "64",
                    "4",
                    relations,
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            metadata = json.loads(completed.stdout)
            self.assertEqual(metadata["selected_relation_count"], 64)
            self.assertEqual(metadata["maximum_relation_degree"], 1)
            payload = relations.read_bytes()
            self.assertEqual(len(payload), 2 * 64 * 8)
            self.assertEqual(
                struct.unpack("<64Q", payload[: 64 * 8]),
                tuple(1 << row for row in range(64)),
            )
            self.assertEqual(
                struct.unpack("<64Q", payload[64 * 8 :]), (0,) * 64
            )
