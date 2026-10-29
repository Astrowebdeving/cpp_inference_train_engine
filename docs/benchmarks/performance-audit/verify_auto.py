import json
import os
import subprocess
import sys
from pathlib import Path
root=Path('/Users/tu15/alex_cpp/CPP_Completed')
out=root/'artifacts/performance-audit'
py=root/'.venv-serving/bin/python'
binary=root/'build/bin/mini_mla_gemma'
model=root/'artifacts/gemma-4-E2B-it-4bit'
commands={}
env=os.environ.copy()
env['PYTORCH_ENABLE_MPS_FALLBACK']='0'
env['VECLIB_MAXIMUM_THREADS']='1'
def run(name,command):
    command=list(map(str,command))
    commands[name]=command
    (out/'auto-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (out/(name+'.log')).open('w') as log:
        subprocess.run(command,cwd=root,env=env,check=True,timeout=600,stdout=log,stderr=subprocess.STDOUT)
    print(name,'passed',flush=True)
run('gemma-auto',[py,root/'tests/test_gemma.py','--binary',binary,'--model',model,'--output',out/'gemma-auto.json'])
run('gemma-http-auto',[py,root/'tests/test_gemma_http.py','--binary',binary,'--model',model,'--output',out/'gemma-http-auto.json'])
run('cache-policy-audit',[py,'/private/tmp/mla_kv_policy_audit.py'])
run('serving-auto-128',[py,root/'tools/compare_serving.py','--model',model,'--output',out/'serving-auto-128','--concurrency','1','4','8','--prompt-length','128','--max-tokens','64','--warmup','1','--repetitions','5','--vllm-first'])
