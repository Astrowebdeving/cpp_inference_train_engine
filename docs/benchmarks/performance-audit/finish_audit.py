import json
import os
import subprocess
import sys
from pathlib import Path
root=Path('/Users/tu15/alex_cpp/CPP_Completed')
out=root/'artifacts/performance-audit'
python=root/'.venv-serving/bin/python'
binary=root/'build/bin/mini_mla_gemma'
model=root/'artifacts/gemma-4-E2B-it-4bit'
commands={}
env=os.environ.copy()
env['PYTORCH_ENABLE_MPS_FALLBACK']='0'
env['VECLIB_MAXIMUM_THREADS']='1'
def run(name,command,timeout=600):
    command=list(map(str,command))
    commands[name]=command
    (out/'final-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (out/(name+'.log')).open('w') as log:
        subprocess.run(command,cwd=root,env=env,check=True,timeout=timeout,stdout=log,stderr=subprocess.STDOUT)
    print(name,'passed',flush=True)
compare=[python,root/'tools/compare_serving.py','--native-only','--model',model,'--max-tokens','64','--warmup','1','--independent-kv-baseline']
run('serving-packed-1024',compare+['--output',out/'serving-packed-1024','--concurrency','1','4','8','--prompt-length','1024','--repetitions','5'])
run('serving-packed-mixed',compare+['--output',out/'serving-packed-mixed','--concurrency','4','8','--prompt-length','128','--repetitions','3','--mixed-lengths'])
run('gemma-final', [python,root/'tests/test_gemma.py','--binary',binary,'--model',model,'--output',out/'gemma-final.json'])
run('gemma-http-final',[python,root/'tests/test_gemma_http.py','--binary',binary,'--model',model,'--output',out/'gemma-http-final.json'])
run('muon-learning-final',[sys.executable,root/'tools/smoke.py','--backend','mps','--precision','fp16','--optimizer','muon','--output',out/'muon-learning-final'])
run('ubsan-build',['cmake','--build',root/'build-ubsan','-j','8'])
run('ubsan-final',['ctest','--test-dir',root/'build-ubsan','--output-on-failure'])
