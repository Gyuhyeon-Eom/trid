"""System prompt for explanation-style grounded cases (C hand-off cases: conclusion, basis, conditions, exceptions).
Kept apart from the value-only evaluation instruction so training targets never contradict their instruction."""

CASE_SYSTEM = ('주어진 근거만 사용하세요. 근거에 있는 결론과 그 이유·조건·예외를 빠짐없이 쓰고, 근거에 없는 내용은 '
               '추측하지 마세요. 근거가 부족하거나 시점·대상 확인이 필요하면 그렇다고 밝히세요.')
