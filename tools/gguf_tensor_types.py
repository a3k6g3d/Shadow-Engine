import struct, sys
FMT = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
def rd(f, fmt):
    n = struct.calcsize(fmt); return struct.unpack(fmt, f.read(n))[0]
def rstr(f):
    n = rd(f, '<Q'); return f.read(n).decode('utf-8', 'replace')
def skipval(f, t):
    if t == 8:
        rstr(f); return
    if t == 9:
        et = rd(f, '<I'); n = rd(f, '<Q')
        if et == 8:
            for _ in range(n): rstr(f)
        else:
            f.seek(struct.calcsize(FMT[et]) * n, 1)
        return
    f.seek(struct.calcsize(FMT[t]), 1)

TYPE_NAMES = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',
 10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',16:'IQ2_XXS',17:'IQ2_XS',
 18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',24:'I8',25:'I16',
 26:'I32',27:'I64',28:'F64',29:'IQ1_M',30:'BF16',31:'Q4_0_4_4',32:'Q4_0_4_8',33:'Q4_0_8_8',
 34:'TQ1_0',35:'TQ2_0',36:'MXFP4'}

p = sys.argv[1]
want = sys.argv[2:] if len(sys.argv) > 2 else None
with open(p, 'rb') as f:
    assert f.read(4) == b'GGUF'
    ver = rd(f, '<I'); nt = rd(f, '<Q'); nk = rd(f, '<Q')
    for _ in range(nk):
        k = rstr(f); t = rd(f, '<I'); skipval(f, t)
    counts = {}
    for _ in range(nt):
        name = rstr(f)
        ndim = rd(f, '<I')
        dims = [rd(f, '<Q') for _ in range(ndim)]
        ty = rd(f, '<I')
        off = rd(f, '<Q')
        tyname = TYPE_NAMES.get(ty, f'?{ty}')
        counts[tyname] = counts.get(tyname, 0) + 1
        if want is None or any(w in name for w in want):
            print(name, dims, tyname)
    print('--- type counts ---')
    for k2, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(k2, v)
