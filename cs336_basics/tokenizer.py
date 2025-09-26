import pickle
from train_bpe import get_pretokens_list
from collections import defaultdict
from collections.abc import Iterable

class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges:list[tuple[bytes, bytes]], special_tokens:list[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # 处理没有遇到过的special_token，直接加到末尾
        if special_tokens is not None:
            for special_token in special_tokens:
                special_token_encoded = special_token.encode('utf-8')
                if special_token_encoded not in vocab.values():
                    vocab[len(vocab)] = special_token_encoded

        # 创建bytes2int的反向字典，encoding时需要从bytes直接到ID
        vocab_inverse = dict()
        for ID, token in vocab.items():
            vocab_inverse[token] = ID
        self.vocab_inverse = vocab_inverse

        # 创建merges_rank字典，便于O(1)拿到rank
        merges_rank = dict()
        for i in range(len(merges)):
            merges_rank[merges[i]] = i
        self.merges_rank = merges_rank

    # 加了这个修饰器后，就可以直接用这个办法初始化Tokenizer
    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens = None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)


    def index_bytes_pair(self, pair_to_find:tuple[bytes], bytes_list:list[bytes]):
            idx = bytes_list.index(pair_to_find[0])
            while idx < len(bytes_list) - 1:
                if bytes_list[idx + 1] == pair_to_find[1]:
                    return idx
                idx = bytes_list.index(pair_to_find[0], idx + 1)
            return -1
        
    def encode(self, text:str) -> list[int]:
        pretoken_list = get_pretokens_list(text, self.special_tokens)
        encoded_list = []

        for pretoken in pretoken_list:

            if self.special_tokens and pretoken in self.special_tokens:
                encoded_list.append(self.vocab_inverse[pretoken.encode('utf-8')])
                continue

            pretoken_bytes = pretoken.encode('utf-8')
            if len(pretoken_bytes) <= 1:
                encoded_list.append(self.vocab_inverse[bytes([pretoken_bytes[0]])])
                continue

            bytes_list = []
            bytes_2_ranks = dict()

            for i in range(len(pretoken_bytes)):
                bytes_list.append(bytes([pretoken_bytes[i]]))
                if i > 0:
                    pair = (bytes_list[i-1], bytes_list[i])
                    if self.merges_rank.get(pair) is not None:
                        bytes_2_ranks[pair] = self.merges_rank[pair]
            
            while len(bytes_2_ranks) > 0:
                # 找到rank最小的byte_pair在list中的index

                best_match = min(bytes_2_ranks.items(), key = lambda x:x[1])  # 注意此处要加.items()，这样比较的才是键值对，否则只是键，不比较rank
                lowest_pair = best_match[0]
                new_bytes = lowest_pair[0] + lowest_pair[1]
                idx_in_list = self.index_bytes_pair(lowest_pair, bytes_list)
                
                bytes_list[idx_in_list] = new_bytes
                del bytes_list[idx_in_list+1]

                bytes_2_ranks = {}
                for i in range(len(bytes_list) - 1):
                    pair = (bytes_list[i], bytes_list[i+1])
                    if self.merges_rank.get(pair):
                        bytes_2_ranks[pair] = self.merges_rank[pair]
            # 得到每个bytes对应的ID,然后加到encoded_list里
            num_list = []
            for bts in bytes_list:
                num_list.append(self.vocab_inverse[bts])
            
            encoded_list.extend(num_list)
        print(self.vocab[220], self.vocab[428])
        return encoded_list
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            encoded_list = self.encode(text)
            yield from encoded_list

    def decode(self, ids:list[int]) -> str:
        decoded_list = [self.vocab[id] for id in ids]
        bytes_sequence =  b''.join(decoded_list)
        return bytes_sequence.decode('utf-8', errors = 'replace')
    

if __name__ == '__main__':
    vocab_path = 'vocab.pkl'
    merges_path = 'merges.pkl'
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=['<|endoftext|>'])
    # tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=None)

    text = " this"
    print("ready to encode: ", text)
    encoded_list = tokenizer.encode(text)
    print("encoded text is: ", encoded_list)
    decoded_text = tokenizer.decode(encoded_list)
    print("decoded ids is: ", decoded_text)

    assert text == decoded_text