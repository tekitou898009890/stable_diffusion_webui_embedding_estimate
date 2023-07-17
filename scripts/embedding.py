import modules
from modules import shared, devices
from modules.textual_inversion.textual_inversion import Embedding

def make_temp_embedding(name,vectors,cache):
    if name in cache:
        embed = cache[name]
    else:
        embed = Embedding(vectors,name)
        cache[name] = embed
    embed.vec = vectors
    embed.step = None
    shape = vectors.size()
    embed.vectors = shape[0]
    embed.shape = shape[-1]
    embed.cached_checksum = None
    embed.filename = ''
    register_embedding(name,embed)

def register_embedding(name,embedding):
    # /modules/textual_inversion/textual_inversion.py
    self = modules.sd_hijack.model_hijack.embedding_db
    model = shared.sd_model
    try:
        ids = model.cond_stage_model.tokenize([name])[0]
        first_id = ids[0]
    except:
        return
    if embedding is None:
        if self.word_embeddings[name] is None:
            return
        del self.word_embeddings[name]
    else:
        self.word_embeddings[name] = embedding
    if first_id not in self.ids_lookup:
        if embedding is None:
            return
        self.ids_lookup[first_id] = []
    save = [(ids, embedding)] if embedding is not None else []
    old = [x for x in self.ids_lookup[first_id] if x[1].name!=name]
    self.ids_lookup[first_id] = sorted(old + save, key=lambda x: len(x[0]), reverse=True)
    return embedding

def get_conds_with_caching(function, required_prompts, steps, cache):
    """
    Returns the result of calling function(shared.sd_model, required_prompts, steps)
    using a cache to store the result if the same arguments have been used before.

    cache is an array containing two elements. The first element is a tuple
    representing the previously used arguments, or None if no arguments
    have been used before. The second element is where the previously
    computed result is stored.
    """

    if cache[0] is not None and (required_prompts, steps) == cache[0]:
        return cache[1]

    with devices.autocast():
        cache[1] = function(shared.sd_model, required_prompts, steps)

    cache[0] = (required_prompts, steps)
    return cache[1]
