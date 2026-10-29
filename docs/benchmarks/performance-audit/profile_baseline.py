import sys,time,json,statistics,hashlib
from pathlib import Path
root=Path('/Users/tu15/alex_cpp/CPP_Completed')
sys.path.insert(0,str(root/'tools'))
import gemma_runtime as runtime
from transformers import AutoTokenizer
model=root/'artifacts/gemma-4-E2B-it-4bit'
binary=root/'artifacts/performance-audit/baseline/build/bin/mini_mla_gemma'
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
metrics={}
call=runtime.GemmaProcess.call
token=runtime.GemmaScheduler._token

def timed_call(self,command,**fields):
 start=time.perf_counter()
 result=call(self,command,**fields)
 elapsed=(time.perf_counter()-start)*1000
 metrics.setdefault(command,[]).append({'wall_ms':elapsed,'native_ms':result.get('elapsed_ms',0)})
 return result

def timed_token(self,r,t):
 start=time.perf_counter()
 result=token(self,r,t)
 metrics.setdefault('token',[]).append({'wall_ms':(time.perf_counter()-start)*1000})
 return result
runtime.GemmaProcess.call=timed_call
runtime.GemmaScheduler._token=timed_token
scheduler=runtime.GemmaScheduler(binary,model,tokenizer,context_limit=2048,max_batch=8)
base=tokenizer.apply_chat_template([{'role':'user','content':'Explain how attention works.'}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
base=(base[:-1]+([200000]*2048))
results=[]
try:
 for length,count in [(128,1),(128,4),(128,8),(1024,8)]:
  prompt=base[:length-1]+[251143]
  warm=[scheduler.submit(prompt,64,ignore_eos=True) for _ in range(count)]
  for r in warm: r.future.result(timeout=90)
  metrics.clear()
  start=time.perf_counter()
  requests=[scheduler.submit(prompt,64,ignore_eos=True) for _ in range(count)]
  for r in requests: r.future.result(timeout=90)
  elapsed=time.perf_counter()-start
  time.sleep(.05)
  summary={command:{'calls':len(samples),'wall_total_ms':sum(s['wall_ms'] for s in samples),'wall_median_ms':statistics.median(s['wall_ms'] for s in samples),'native_total_ms':sum(s.get('native_ms',0) for s in samples)} for command,samples in metrics.items()}
  record={'prompt_length':length,'concurrency':count,'tokens_per_second':count*64/elapsed,'phases':summary}
  results.append(record)
  print(json.dumps(record),flush=True)
finally:
 scheduler.close()
(root/'artifacts/performance-audit/baseline/profile.json').write_text(json.dumps({'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'method':'Native command wall time, native engine elapsed time and scheduler token handling; one warmup then one fixed-length burst per shape; diagnostic, not a final speed comparison.','results':results},indent=2)+'\n')
