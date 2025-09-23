
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges:list[tuple[bytes, bytes]], special_tokens:list[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        for special_token in special_tokens:
            if special_token not in vocab.values():
                vocab[len(vocab)] = special_token.encode('utf-8')
    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None)
        