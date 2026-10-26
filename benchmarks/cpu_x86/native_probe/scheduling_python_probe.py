import inspect,textwrap,types
import packed_encoders.cpu.int8 as q8module
from native_probe import scheduling_python_baseline


def variants(q8):
    original=textwrap.dedent(inspect.getsource(scheduling_python_baseline.core))
    unbind=original.replace('    for kind, segment in zip(engine.kinds, self.segments):\n        q, k, v = rope.unbind(0)', '    q, k, v = rope.unbind(0)\n    for kind, segment in zip(engine.kinds, self.segments):')
    a=original.index('    for kind, segment in zip(engine.kinds, self.segments):')
    b=original.index('    return out[0]',a)
    direct=original[:a]+'''    att=engine.attention
    assert self.x86 and att.native_local and att.native_global and att.token_major and max(lengths)<=128
    longest=max(lengths)
    workers=min(att.threads,engine.heads*len(lengths))
    needed=workers*(att.tile+64)*((longest+15)//16*16)
    if att.scratch is None or att.scratch.numel()<needed:
        att.scratch=torch.empty(needed,dtype=torch.float32)
    shape=(1,tokens,engine.heads*64)
    if att.output is None or att.output.shape!=shape:
        att.output=torch.empty(shape,dtype=torch.float32)
    ptr=self.rope.data_ptr();plane=engine.heads*tokens*64*4
    prefix=(ptr,ptr+plane,ptr+2*plane,att.output.data_ptr(),boundaries.data_ptr(),tokens,engine.heads,len(lengths),att.tile)
    suffix=(1,workers,att.scratch.data_ptr(),att.scratch.numel())
    fn=att.library.pe_local_attention
    for kind, segment in zip(engine.kinds, self.segments):
        half=att.half if kind=='sliding_attention' else longest
        status=fn(*prefix,half,*suffix)
        assert status==0,status
        if self.library.pe_q8_segment(model,ctypes.addressof(segment),buffers,att.output.data_ptr(),tokens,self.workers):
            raise PackedEncodersError('native CPU int8 layer failed')
'''+original[b:]
    result={}
    for name,source in [('unbind',unbind),('direct',direct)]:
        ns=q8module.__dict__.copy();exec(source,ns);result[name]=types.MethodType(ns['core'],q8)
    return result
