import pickle
from train_bpe import get_pretokens_list
from collections import defaultdict

class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges:list[tuple[bytes, bytes]], special_tokens:list[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # 处理没有遇到过的special_token，直接加到末尾
        for special_token in special_tokens:
            if special_token not in vocab.values():
                vocab[len(vocab)] = special_token.encode('utf-8')

        # 创建bytes2int的反向字典，encoding时需要从bytes直接到ID
        vocab_inverse = dict()
        for ID, token in vocab.items():
            vocab_inverse[token] = ID
        self.vocab_inverse = vocab_inverse
    
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens = None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        cls(vocab, merges, special_tokens)


    def index_bytes_pair(self, bytes_2_ranks:dict[tuple[bytes], int], bytes_list:list[bytes]):
        lowest_rank = min(bytes_2_ranks, key = lambda x:(x[1]))
        bytes_pair = lowest_rank[0]
        pre = 0
        while pre < len(bytes_list):
            index = bytes_list.index(bytes_pair[0], pre)
            if bytes_list[index + 1] == bytes_pair[1]:
                break
            pre = index
        return index, bytes_pair
        
    def encode(self, text:str) -> list[int]:
        pretoken_list = get_pretokens_list(text, self.special_tokens)
        encoded_list = []
        for pretoken in pretoken_list:

            if pretoken in self.special_tokens:
                encoded_list.append(self.vocab_inverse[pretoken.encode('utf-8')])
                continue

            pretoken_bytes = pretoken.encode('utf-8')
            bytes_list = []
            for num in list(pretoken_bytes):
                bytes_list.append(bytes(num))
            
            bytes_2_ranks = dict()
            for i in range(len(self.merges)):
                if self.merges[i] in bytes_list:
                    bytes_2_ranks[self.merges] = i
            
            while True:
                # 找到rank最小的byte_pair在list中的index
                idx, bytes_pair = self.index_bytes_pair(bytes_2_ranks, bytes_list)
                bytes_list[idx] = bytes_list[idx] + bytes_list[idx+1]
                del bytes_list[idx+1]
                del bytes_2_ranks[bytes_pair]

                if idx < len(bytes_list):
                    if (bytes_list[idx], bytes_list[idx + 1]) in self.merges:
                        new_pair = (bytes_list[idx], bytes_list[idx + 1])
                        bytes_2_ranks[new_pair] = self.merges.index(new_pair)
                if idx > 1:
                    if (bytes_list[idx - 1], bytes_list[idx]) in self.merges:
                        new_pair = (bytes_list[idx - 1], bytes_list[idx])
                        bytes_2_ranks[new_pair] = self.merges.index(new_pair)
                
                if len(bytes_2_ranks) == 0:
                    break
            # 得到每个bytes对应的ID,然后加到encoded_list里
            num_list = []
            for bts in bytes_list:
                num_list.append(self.vocab_inverse[bts])
            encoded_list.extend(num_list)
        
        return encoded_list