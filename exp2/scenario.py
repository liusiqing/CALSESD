"""按调用、重复号、原始行绑定随机流；不依赖候选分块。"""
import hashlib

def seed_for(seed, call, repeat, domain, row=-1):
    value=f'exp2-train-v1:{seed}:{call}:{repeat}:{domain}:{row}'.encode()
    return int.from_bytes(hashlib.blake2b(value,digest_size=8).digest(),'little') % 2**32
