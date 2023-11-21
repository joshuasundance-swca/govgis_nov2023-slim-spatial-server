def transform_emb(emb: list[float]) -> str:
    embstr = ",".join(map(str, emb))
    return f"[{embstr}]"
