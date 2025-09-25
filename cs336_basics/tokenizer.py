import pickle
from train_bpe import get_pretokens_list
from collections import defaultdict

class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges:list[tuple[bytes, bytes]], special_tokens:list[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # 处理没有遇到过的special_token，直接加到末尾
        if special_tokens is not None:
            for special_token in special_tokens:
                if special_token not in vocab.values():
                    vocab[len(vocab)] = special_token.encode('utf-8')

        # 创建bytes2int的反向字典，encoding时需要从bytes直接到ID
        vocab_inverse = dict()
        for ID, token in vocab.items():
            vocab_inverse[token] = ID
        self.vocab_inverse = vocab_inverse
    
    # 加了这个修饰器后，就可以直接用这个办法初始化Tokenizer
    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens = None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)


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
            if len(pretoken_bytes) == 1:
                encoded_list.append(self.vocab_inverse[bytes([pretoken_bytes[0]])])
                continue

            bytes_pairs = []
            for i in range(len(pretoken_bytes) - 1):
                bytes_pairs.append((bytes([pretoken_bytes[i]]), bytes([pretoken_bytes[i+1]])))
            
            bytes_2_ranks = dict()
            for i in range(len(self.merges)):
                if self.merges[i] in bytes_pairs:
                    bytes_2_ranks[self.merges[i]] = i
            
            while len(bytes_2_ranks) > 0:
                # 找到rank最小的byte_pair在list中的index
                best_match = min(bytes_2_ranks, key = lambda x:x[1])
                lowest_pair = best_match
                new_bytes = best_match[0] + best_match[1]
                idx_in_pairs = bytes_pairs.index(lowest_pair)

                if idx_in_pairs > 0:
                    pair = (bytes_pairs[idx_in_pairs - 1][0], new_bytes)
                    if pair in self.merges:
                        bytes_2_ranks[pair] = self.merges.index(pair)
                    if bytes_pairs[idx_in_pairs - 1] in self.merges:
                        del bytes_2_ranks[bytes_pairs[idx_in_pairs - 1]]
                    bytes_pairs[idx_in_pairs - 1] = pair
                    
                if idx_in_pairs < len(bytes_pairs) - 1:
                    pair = (new_bytes, bytes_pairs[idx_in_pairs + 1][1])
                    if pair in self.merges:
                        bytes_2_ranks[pair] = self.merges.index(pair)
                    if bytes_pairs[idx_in_pairs + 1] in self.merges:
                        del bytes_2_ranks[bytes_pairs[idx_in_pairs + 1]]
                    bytes_pairs[idx_in_pairs + 1] = pair
                del bytes_pairs[idx_in_pairs]
                del bytes_2_ranks[lowest_pair]

            bytes_list = [first[0] for first in bytes_pairs]
            bytes_list.append(bytes_pairs[-1][1])
            
            print(bytes_list)
            # 得到每个bytes对应的ID,然后加到encoded_list里
            num_list = []
            for bts in bytes_list:
                num_list.append(self.vocab_inverse[bts])
            
            encoded_list.extend(num_list)
        
        return encoded_list
    

if __name__ == '__main__':
    vocab_path = 'vocab.pkl'
    merges_path = 'merges.pkl'
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<s>", "</s>"])
    text = "<s>hello world, bitch.</s>"
    print("ready to encode: ", text)
    print("encoded text is: ", tokenizer.encode(text))