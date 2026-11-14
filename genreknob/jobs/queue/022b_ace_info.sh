#!/usr/bin/env bash
# Job 022b: list ACE-Step checkpoints and generation options (for choosing a stronger text condition).
source ~/gk-env.sh
cd ~/gk
A=$GK_ACE_ROOT
ls -la $A/checkpoints 2>/dev/null
du -sh $A/checkpoints/* 2>/dev/null
grep -rn "acestep-v15-[a-z]*" --include=*.py -o $A/acestep | sed 's/.*://' | sort | uniq -c | sort -rn | head -20
$ACE_PY - <<'PY'
import os, sys, dataclasses, inspect
sys.path.insert(0, os.path.expanduser("~/gk-ace/ACE-Step-1.5"))
from acestep.inference import GenerationParams, GenerationConfig
for f in dataclasses.fields(GenerationParams):
    print("param", f.name, "=", f.default if f.default is not dataclasses.MISSING else "<req>")
for f in dataclasses.fields(GenerationConfig):
    print("config", f.name, "=", f.default if f.default is not dataclasses.MISSING else "<req>")
from acestep.handler import AceStepHandler
print(inspect.signature(AceStepHandler.initialize_service))
PY
grep -rn "def initialize\|init_llm\|class LLMHandler\|def initialize(" $A/acestep/*.py $A/acestep/llm*/*.py 2>/dev/null | head -20
grep -rn "Skipping LM for task_type" -B 12 $A/acestep/inference.py | head -40
