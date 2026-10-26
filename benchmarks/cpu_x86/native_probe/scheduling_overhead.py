import ctypes,statistics,subprocess,time,json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import torch
from packed_encoders.cpu.attention import _openmp_runtime
threads=int(sys.argv[1]);torch.set_num_threads(threads)
directory=tempfile.TemporaryDirectory(prefix='omp-overhead-')
source=Path(__file__).with_suffix('.c')
output=Path(directory.name)/'overhead.so'
subprocess.run(['gcc','-O3','-fPIC','-fopenmp','-shared',str(source),_openmp_runtime(),'-o',str(output)],check=True)
lib=ctypes.CDLL(str(output));fn=lib.probe;fn.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p];fn.restype=None
for m in (16,32,256):
 out=torch.empty(m*16);times={0:[],1:[]}
 for rnd in range(16):
  for nowait in ((0,1) if rnd%2 else (1,0)):
   start=time.perf_counter();fn(threads,m,2000,nowait,out.data_ptr());elapsed=(time.perf_counter()-start)*1e6/2000
   if rnd:times[nowait].append(elapsed)
 print(json.dumps({'threads':threads,'m':m,'median_us':{k:statistics.median(v) for k,v in times.items()},'round_us':times}),flush=True)
