"""场景随机流：绑定调用、重复号与原始行，不绑定候选或分块。"""
import hashlib

def seed_for(seed, call, repeat, domain, row=-1):
    payload = f"exp2-v2:{seed}:{call}:{repeat}:{domain}:{row}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**32)
