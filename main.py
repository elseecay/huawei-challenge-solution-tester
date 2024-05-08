import subprocess
import argparse
import random
import math

from typing import List, Union, Tuple, Type
from pathlib import Path
from dataclasses import dataclass

import numpy


CF_TEST_COUNT = 76


@dataclass
class Score:
    n: int      # nums count
    w: float    # weight
    p: float    # memread penalty
    c: float    # average op cost
    d: float    # data score
    e: float    # relative error
    a: float    # accuracy score
    s: float    # total score


@dataclass
class ScoreStat:
    block_index: int
    block_reads: int
    penalty_count: int
    weight: int


@dataclass
class SumSequence:
    np_type: Type
    items: List[Union[int, "SumSequence"]]


def create_solution_input(path: Path, numbers: List[float]):
    file_strings = list()
    file_strings.append(str(len(numbers)))
    for n in numbers:
        nstr = f"{n:.{200}f}".rstrip("0")
        if nstr[-1] == ".":
            nstr += "0"
        file_strings.append(nstr)
    with open(path, "w") as f:
        f.write(" ".join(file_strings))


def parse_solution_output(path: Path) -> SumSequence:

    def parse_type(c: str) -> Type:
        if c == "h":
            return numpy.float16
        if c == "s":
            return numpy.float32
        if c == "d":
            return numpy.float64
        raise Exception("Invalid type")

    def parse_sequence(s: str, cur: int) -> Tuple[int, SumSequence]:
        if s[cur] != "{":
            raise Exception("Expected {")
        cur += 1
        np_type = parse_type(s[cur])
        cur += 1
        seq = SumSequence(np_type, list())
        if s[cur] != ":":
            raise Exception("Expected :")
        cur += 1
        while True:
            if s[cur] == "{":
                cur, child = parse_sequence(s, cur)
                seq.items.append(child)
            elif ord("0") <= ord(s[cur]) <= ord("9"):
                index = 0
                while ord("0") <= ord(s[cur]) <= ord("9"):
                    index *= 10
                    index += ord(s[cur]) - ord("0")
                    cur += 1
                seq.items.append(index)
            else:
                raise Exception(f"Invalid character {s[cur]}")
            cur += 1
            if s[cur - 1] == "}":
                break
            if s[cur - 1] != ",":
                raise Exception("Expected ,")
        return cur, seq

    with open(path, "r") as f:
        content = f.read()

    return parse_sequence(content, 0)[1]


def generate_input_numbers(n: int) -> List[float]:
    numbers = list()
    for _ in range(n):
        x = random.uniform(-1e+10, +1e+10)
        numbers.append(x)
    return numbers


def calculate_accurate_sum(numbers: List[float]) -> float:
    rnums = list()
    numbers = sorted(numbers)
    i, j = 0, len(numbers) - 1
    while i <= j:
        if i != j:
            rnums.extend((numbers[i], numbers[j]))
            i, j = i + 1, j - 1
        else:
            rnums.append(numbers[i])
            i += 1
    return math.fsum(rnums)


def calculate_sequence_sum(numbers: List[float], seq: SumSequence, stat: ScoreStat) -> Union[numpy.float16, numpy.float32, numpy.float64]:
    weight = {numpy.float16: 1, numpy.float32: 2, numpy.float64: 4}[seq.np_type]
    result = seq.np_type(0.0)
    for elem in seq.items:
        if isinstance(elem, int):
            if stat.block_reads == 16:
                stat.block_reads = 0
                stat.block_index = elem
            result += seq.np_type(numbers[elem - 1])
            stat.weight += weight
            stat.block_reads += 1
            if abs(elem - stat.block_index) > 15:
                stat.penalty_count += 1
        else:
            subseq_result = calculate_sequence_sum(numbers, elem, stat)
            if elem.np_type is not seq.np_type:
                subseq_result = seq.np_type(subseq_result)
            result += subseq_result
            stat.weight += weight
    return result


def calculate_score(numbers: List[float], seq: SumSequence):
    accurate_sum = calculate_accurate_sum(numbers)
    score_stat = ScoreStat(0, 16, 0, 0)
    sequence_sum = calculate_sequence_sum(numbers, seq, score_stat).item()
    score = Score(0, 0, 0, 0, 0, 0, 0, 0)
    score.n = len(numbers)
    score.w = score_stat.weight
    score.p = ((score_stat.penalty_count + 1) * score_stat.penalty_count // 2) / 20000.0
    score.c = (score.w + score.p) / (score.n - 1)
    score.d = 10.0 / math.sqrt(score.c + 0.5)
    score.e = max(abs(sequence_sum - accurate_sum) / max(accurate_sum, 1e-200), 1e-20)
    score.a = math.pow(score.e, 0.05)
    score.s = score.d / score.a
    return score


def test_solution(args: argparse.Namespace) -> Score:
    n = 1000
    numbers = generate_input_numbers(n)
    create_solution_input(args.input, numbers)
    subprocess.run([args.executable], check=True, cwd=Path(args.executable).parent)
    seq = parse_solution_output(args.output)
    score = calculate_score(numbers, seq)
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50))
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file path")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file path")
    parser.add_argument("-e", "--executable", type=str, required=True, help="Executable file path")
    return parser.parse_args()


def main():
    args = parse_args()
    score = test_solution(args)
    print(score.s * CF_TEST_COUNT)


if __name__ == "__main__":
    main()