import json
import statistics
import time
from pathlib import Path
import mlx.core as mx
import numpy as np

root=Path('/Users/tu15/alex_cpp/CPP_Completed')
mx.set_default_device(mx.gpu)
weights=mx.load(root/'artifacts/gemma-4-E2B-it-4bit/model.safetensors')
prefix=next(k.split('layers.')[0]+'layers.' for k in weights if 'layers.0.mlp.gate_proj.weight' in k)
rows=[]
for layer in [0,15]:
    p=prefix+str(layer)+'.mlp.'
    def w(name):
        return [weights[p+name+suffix] for suffix in ['.weight','.scales','.biases']]
    gate,up=w('gate_proj'),w('up_proj')
    joined=[mx.concatenate([a,b],axis=0) for a,b in zip(gate,up)]
    mx.eval(gate,up,joined)
    def apply_geglu(a,b):
        s=lambda v: mx.array(v,dtype=a.dtype)
        return s(0.5)*a*(s(1)+mx.tanh(s(float(np.sqrt(2/np.pi)))*(a+s(0.044715)*mx.power(a,mx.array(3)))))*b
    geglu=mx.compile(apply_geglu,shapeless=True)
    def mul(x,v):
        return mx.quantized_matmul(x,*v,transpose=True,group_size=64,bits=4)
    def separate(x):
        return geglu(mul(x,gate),mul(x,up))
    def packed(x):
        a,b=mx.split(mul(x,joined),2,axis=-1)
        return geglu(a,b)
    for batch,length in [(1,1),(4,1),(8,1),(1,128),(8,128)]:
        x=mx.array(np.random.default_rng(1).standard_normal((batch,length,1536)).astype(np.float32)).astype(mx.bfloat16)
        a,b=separate(x),packed(x)
        mx.eval(a,b)
        exact=bool(mx.all(a==b).item())
        timing={}
        for name,fn in [('separate',separate),('packed',packed),('packed-repeat',packed),('separate-repeat',separate)]:
            samples=[]
            for i in range(53):
                start=time.perf_counter()
                mx.eval(fn(x))
                if i>=3:
                    samples.append(1000*(time.perf_counter()-start))
            timing[name]=statistics.median(samples)
        row={'layer':layer,'batch':batch,'length':length,'bitwise_equal':exact,'median_ms':timing}
        rows.append(row)
        print(row,flush=True)
(root/'artifacts/performance-audit/projection-micro.json').write_text(json.dumps(rows,indent=2)+'\n')
