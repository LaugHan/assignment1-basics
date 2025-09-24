import regex as re
from collections import defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_all_indices(s:str, sub_s:str):
    indices = []
    pre = 0
    while s.find(sub_s) != -1:
        idx = s.find(sub_s)
        if idx == -1:
            break
        indices.append(idx + pre)
        pre += idx + 1
        s = s[idx+1:]
    return indices

def init_vocab(special_tokens:list[str])->dict[int, bytes]:
    vocab = {} # 用来存放ID到bytes的映射
    for i in range(256):
        vocab[i] = bytes([i])  # 注意bytes接受的是list，如果直接一个整数，会等价于一个长度为i全是0的list
    for i in range(len(special_tokens)):
        vocab[256+i] = special_tokens[i].encode('utf-8')
    return vocab

def get_pretokens_list(content:str, special_tokens:list[str]) -> list[str]:
    pretokens_list = []
    
    if not special_tokens:
        for pretoken in re.finditer(PAT, content):
            pretoken_content = pretoken.group()
            pretokens_list.append(pretoken_content)
        return pretokens_list
    
    # 单独拿掉特殊词，然后得到所有pretokens的迭代器
    escaped_tokens = [re.escape(token) for token in special_tokens]
    delimiter = '(' + '|'.join(escaped_tokens) + ')'
    content_parts = re.split(delimiter, content)

    for part in content_parts:
        if part in special_tokens:
            continue
        pre_tokens = re.finditer(PAT, part)
        for pretoken in pre_tokens:
            pretoken_content = pretoken.group()
            pretokens_list.append(pretoken_content)
    return pretokens_list

def get_pretoken_count_num(content:str, special_tokens:list[str]) -> dict[str, int]:
    pretoken_count_num = {}
    
    if not special_tokens:
        for pretoken in re.finditer(PAT, content):
            pretoken_content = pretoken.group()
            pretoken_count_num[pretoken_content] = pretoken_count_num.get(pretoken_content, 0) + 1

        return pretoken_count_num
    
    # 单独拿掉特殊词，然后得到所有pretokens的迭代器
    escaped_tokens = [re.escape(token) for token in special_tokens]
    delimiter = '(' + '|'.join(escaped_tokens) + ')'
    content_parts = re.split(delimiter, content)

    # 统计各个pretoken的个数

    for part in content_parts:
        if part in special_tokens:
            continue
        pre_tokens = re.finditer(PAT, part)
        for pretoken in pre_tokens:
            pretoken_content = pretoken.group()
            pretoken_count_num[pretoken_content] = pretoken_count_num.get(pretoken_content, 0) + 1
    return pretoken_count_num

def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    with open(input_path, 'r', encoding = 'utf-8') as f:
        content = f.read()

    pretoken_count_num = get_pretoken_count_num(content, special_tokens)

    # 初始化字典，字典中目前存放的是256个初始字节对应的字符，以及特殊字符。
    # 拿到下一次字典更新时放的索引
    vocab = init_vocab(special_tokens)
    vocab_idx = len(vocab)

    # 定义中间变量
    bytes_2_indices = defaultdict(list)
    bytes_counts = defaultdict(int)
    indices_2_linked_list = defaultdict(list)
    indices_2_freq = defaultdict(int)
    merge_sequence = []

    idx = 0

    for pretoken, freq in pretoken_count_num.items():
        pretoken_bytes = pretoken.encode('utf-8')
        linked_list = [bytes([pretoken_bytes[0]])]
        for i in range(1, len(pretoken_bytes)):
            linked_list.append(bytes([pretoken_bytes[i]]))
            b = (linked_list[-2] ,linked_list[-1])
            bytes_2_indices[b].append(idx)
            bytes_counts[b] += freq
        indices_2_linked_list[idx].append(linked_list)
        indices_2_freq[idx] = freq
        idx += 1

    while len(vocab) < vocab_size:
        max_pair = max(bytes_counts.items(), key = lambda x:(x[1], x[0]))
        max_bytes_pair = max_pair[0]
        max_freq = max_pair[1]
        new_pair = max_bytes_pair[0] + max_bytes_pair[1]


        if max_freq < 2:
            break
        vocab[vocab_idx] = new_pair
        vocab_idx += 1
        merge_sequence.append(max_bytes_pair)
        
        pre = -1
        lst = list(bytes_2_indices[max_bytes_pair])
        for i in lst:
            if pre == i:
                continue
            pre = i 
            linked_list = indices_2_linked_list[i][0]

        
            for j in range(1, len(linked_list)):
                old_b = (linked_list[j-1], linked_list[j])
                bytes_2_indices[old_b].remove(i)
                bytes_counts[old_b] -= indices_2_freq[i]

            j = 1
            while j < len(linked_list):
                old_b = (linked_list[j-1], linked_list[j])
                if old_b == max_bytes_pair:
                    linked_list[j-1] = new_pair
                    del linked_list[j]
                    j -= 1
                j += 1
            for j in range(1, len(linked_list)):
                b = (linked_list[j-1], linked_list[j])
                bytes_counts[b] += indices_2_freq[i]
                bytes_2_indices[b].append(i)
    return vocab, merge_sequence


import pickle
def save_vocab_or_merges(file, filepath:str):
    with open(filepath, 'wb') as f:
        pickle.dump(file, f)


def main():
    input_path = '../data/TinyStoriesV2-GPT4-valid.txt'
    vocab_size = 1000
    special_tokens = ['<|endoftext|>']

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    vocab_save_path = './vocab.pkl'
    merges_save_path = './merges.pkl'
    save_vocab_or_merges(vocab, vocab_save_path)
    save_vocab_or_merges(merges, merges_save_path)

if __name__ == '__main__':
    main()