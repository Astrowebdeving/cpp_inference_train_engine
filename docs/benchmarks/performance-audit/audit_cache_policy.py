import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
root=Path('/Users/tu15/alex_cpp/CPP_Completed')
sys.path.insert(0,str(root/'tools'))
from gemma_runtime import GemmaProcess
out=root/'artifacts/performance-audit/cache-engine-auto-policy.json'
rows=[]
for context in [128,1024]:
    for batch in [2,4]:
        handles=list(range(batch))
        prompts=[[2]+[500+i%97 for i in range(context-b-1)] for b in handles]
        for index,reuse in enumerate([False,True,True,False]):
            with GemmaProcess(root/'build/bin/mini_mla_gemma',root/'artifacts/gemma-4-E2B-it-4bit',context_limit=2048,sessions=8,batch_kv=True,all_batch_kv=reuse) as engine:
                [engine.call('prefill',session=b,tokens=prompts[b]) for b in handles]
                for step in range(8):
                    engine.call('decode',sessions=handles,tokens=[100000+step*8+b for b in handles])
                for b in handles:
                    engine.call('release',session=b)
                prefill={'elapsed_ms':sum(engine.call('prefill',session=b,tokens=prompts[b])['elapsed_ms'] for b in handles)}
                native,wall=[],[]
                for step in range(64):
                    start=time.perf_counter()
                    result=engine.call('decode',sessions=handles,tokens=[100000+step*8+b for b in handles])
                    wall.append((time.perf_counter()-start)*1000)
                    native.append(result['elapsed_ms'])
                status=engine.call('info')
                row={'context':context,'prompt_lengths':[len(p) for p in prompts],'batch':batch,'force_all':reuse,'trial':index,'prefill_ms':prefill['elapsed_ms'],'median_native_decode_ms':statistics.median(native),'total_native_decode_ms':sum(native),'total_wall_decode_ms':sum(wall),'native_samples_ms':native,'wall_samples_ms':wall,'engine_status':status}
                rows.append(row)
                out.write_text(json.dumps({'results':rows},indent=2)+'\n')
                print({k:v for k,v in row.items() if k not in ['native_samples_ms','wall_samples_ms','engine_status']},flush=True)
record={'recorded_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'results':rows,'method':'Serial native processes; same fixed 4-bit weights and BF16 activations, fixed decode input token IDs, unequal-position prompts, one warm prefill plus eight warm decode steps, then 64 timed steps; reversed option order. Engine completion included; prefill, protocol, and decode separately recorded. Synthetic token inputs measure cache execution, not generation quality.','source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['src/infer/gemma.cpp','tools/gemma_runtime.py']}}
out.write_text(json.dumps(record,indent=2)+'\n')
