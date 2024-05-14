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
    g: str      # generator name
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
        raise RuntimeError("Invalid type")

    def parse_sequence(s: str, cur: int) -> Tuple[int, SumSequence]:
        if s[cur] != "{":
            raise RuntimeError("Expected {")
        cur += 1
        np_type = parse_type(s[cur])
        cur += 1
        seq = SumSequence(np_type, list())
        if s[cur] != ":":
            raise RuntimeError("Expected :")
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
                raise RuntimeError(f"Invalid character {s[cur]}")
            cur += 1
            if s[cur - 1] == "}":
                break
            if s[cur - 1] != ",":
                raise RuntimeError("Expected ,")
        return cur, seq

    with open(path, "r") as f:
        content = f.read()

    return parse_sequence(content, 0)[1]


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
    return result


def calculate_score(numbers: List[float], seq: SumSequence) -> Score:
    accurate_sum = calculate_accurate_sum(numbers)
    score_stat = ScoreStat(0, 16, 0, 0)
    sequence_sum = calculate_sequence_sum(numbers, seq, score_stat).item()
    score = Score("", 0, 0, 0, 0, 0, 0, 0, 0)
    score.n = len(numbers)
    score.w = score_stat.weight
    score.p = (score_stat.penalty_count + 1) * score_stat.penalty_count / 40000.0
    score.c = (score.w + score.p) / (score.n - 1)
    score.d = 10.0 / math.sqrt(score.c + 0.5)
    if math.isnan(sequence_sum) or math.isinf(sequence_sum):
        score.e = math.inf
        score.a = 1.0
    else:
        score.e = max(abs(sequence_sum - accurate_sum) / max(abs(accurate_sum), 1e-200), 1e-20)
        score.a = math.pow(score.e, 0.05)
    score.s = score.d / score.a
    return score


def test_solution(args: argparse.Namespace) -> List[Score]:
    input_size = args.count
    generators = args.generator.split("+")
    if "all" in generators:
        generators = [value for name, value in globals().items() if name.startswith("input_generator")]
    else:
        generators = list(set(globals()[f"input_generator_{name}"] for name in generators))
    scores = list()
    for g in generators:
        numbers = g(input_size)
        create_solution_input(args.input, numbers)
        subprocess.run([args.executable], check=True)
        seq = parse_solution_output(args.output)
        score = calculate_score(numbers, seq)
        score.g = g.__name__[16:]
        scores.append(score)
    return scores


def parse_args() -> argparse.Namespace:
    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=50)
    usage = "python3 main.py -i input.txt -o output.txt -e solution.exe"
    parser = argparse.ArgumentParser(usage=usage, formatter_class=formatter)
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file path")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file path")
    parser.add_argument("-e", "--executable", type=str, required=True, help="Executable file path")
    parser.add_argument("-g", "--generator", type=str, required=False, default="all", help="Generator name")
    parser.add_argument("-n", "--count", type=int, required=False, default=100000, help="Input size")
    args = parser.parse_args()
    args.input = Path(args.input).resolve().absolute()
    args.output = Path(args.output).resolve().absolute()
    args.executable = Path(args.executable).resolve().absolute()
    return args


def main():
    numpy.seterr("ignore")
    args = parse_args()
    scores = test_solution(args)
    for s in scores:
        print("\n", s.g, sep="")
        print(f"N = {s.n}")
        print(f"W = {s.w}")
        print(f"P = {s.p}")
        print(f"C = {s.c}")
        print(f"E = {s.e}")
        print(f"A = {s.a}")
        print(f"Score = {s.s * CF_TEST_COUNT}")


def input_generator_uniform300(n: int) -> List[float]:
    return [random.uniform(-1e+300, +1e+300) for _ in range(n)]


def input_generator_uniform37(n: int) -> List[float]:
    return [random.uniform(-1e+37, +1e+37) for _ in range(n)]


def input_generator_uniform4(n: int) -> List[float]:
    return [random.uniform(-1e+4, +1e+4) for _ in range(n)]


def input_generator_uniform1(n: int) -> List[float]:
    return [random.uniform(-1.0, 1.0) for _ in range(n)]


def input_generator_overflow32(n: int) -> List[float]:
    return [+3e+38, +3e+38, -3e+38]


def input_generator_overflow16(n: int) -> List[float]:
    return [+65000.0, +65000.0, -65000.0]


if __name__ == "__main__":
    main()
