from pathlib import Path
import sys
directory=Path(sys.argv[1]) if len(sys.argv)>1 else Path("/tmp/pe-scheduling")
directory.mkdir(parents=True,exist_ok=True)
base=Path('benchmarks/cpu_x86/native_probe/scheduling_attention_baseline.c').read_text()
(directory/'static.c').write_text(base.replace('schedule(dynamic, 1)','schedule(static)'))
for split in (2,4):
 s=base.replace('float *kt, float *scores, const int length)', 'float *kt, float *scores, const int length, int begin, int end)')
 s=s.replace('for (int r0 = 0; r0 < length; r0 += 4)', 'for (int r0 = begin; r0 < end; r0 += 4)')
 s=s.replace('    int offsets[sequences], ldo',f'''    int splits = 1;
    if (heads * sequences < 2 * workers && tile >= longest && half >= longest - 1) {{
        splits = {split};
        for (int i=0;i<sequences;++i) if(lengths[i]!=16 && lengths[i]!=32) splits=1;
    }}
    int offsets[sequences], ldo''')
 s=s.replace('schedule(dynamic, 1)','schedule(static)')
 s=s.replace('task < heads * sequences','task < heads * sequences * splits')
 s=s.replace('int head = task / sequences, sequence = task % sequences, length = lengths[sequence];', 'int pair = task / splits, part = task % splits;\n        int head = pair / sequences, sequence = pair % sequences, length = lengths[sequence];')
 for n in (16,32):s=s.replace(f'ldo, kt, scores, {n});', f'ldo, kt, scores, {n}, part * ({n} / splits), (part+1) * ({n} / splits));')
 (directory/f'split{split}.c').write_text(s)
 if split==2:(directory/'split2dynamic.c').write_text(s.replace('schedule(static)','schedule(dynamic, 1)'))
base=Path('benchmarks/cpu_x86/native_probe/scheduling_int8_baseline.c').read_text()
# The final RoPE worksharing barrier duplicates the parallel-region join.
a=base.rindex('#pragma omp for schedule(static)')
s=base[:a]+base[a:].replace('#pragma omp for schedule(static)','#pragma omp for schedule(static) nowait',1)
(directory/'nowait.c').write_text(s)
# The last layer has no QKV; its final norm can also use the parallel join.
a=s.index('#pragma omp for schedule(static)',s.index('gemm(aw, w, g->wmo'))
s=s[:a]+s[a:].replace('#pragma omp for schedule(static)','#pragma omp for schedule(static) nowait',1)
s=s.replace('            qkv = g->wqkv != NULL;', '''            qkv = g->wqkv != NULL;
            if (qkv) {
#pragma omp barrier
            }''')
(directory/'nowait_last.c').write_text(s)
