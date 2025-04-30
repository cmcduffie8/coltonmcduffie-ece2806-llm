import torch.nn as nn
import torch
from model import GPTLanguageModel
import argparse
from utils import get_batch, estimate_loss, levenshtein_distance, get_stats,merge,decode_token_basic,encode_text_token_basic
from rouge import Rouge
import string
import tiktoken
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--block_size', type=int, default=256,
                        help='Size of blocks to process vocabulary')
    parser.add_argument('--vocab_size', type=int, default=2000,
                        help='Size of blocks to process vocabulary')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout')


    # model dataset
    parser.add_argument('--model', type=str, default='basic')
    parser.add_argument('--tokenization_strategy', type=str, default='BPE')
    parser.add_argument('--ckpt', type=str, default='../scratch/model/model_BPE_10epochs.pth')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of Heads in Attention Block')
    parser.add_argument('--n_layer', type=int, default=12, help='Number of Layers in Attention Block')
    parser.add_argument('--n_embd', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--loss', type=str, default='NLL')
    parser.add_argument('--testing_file_prompt', type=str, default='./test_data/test_prompt_1.txt')
    parser.add_argument('--testing_file_response', type=str, default='./test_data/test_response_1.txt')
    parser.add_argument('--testing_file_answer', type=str, default='./test_data/test_answer_1.txt')
    parser.add_argument('--training_file', type=str, default='./train_data/train.txt')
    parser.add_argument('--generate_token_number', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='train_data', choices=['shakespeare','train_data'], help='dataset')

    opt = parser.parse_args()

    return opt

def load_merges(filename):
    merges = {}
    with open(filename, 'r', encoding='utf-8') as f:
        f.seek(0)
        for line in f:
            p0, p1, i = line.strip().split()
            merges[(int(p0), int(p1))] = int(i)
        #print(f.read())
    print(f"Loaded {len(merges)} merges.")
    return merges

def main():
    opt = parse_option()
    with open(opt.testing_file_prompt, 'r', encoding='utf-8') as f:
        text_test_prompt = f.read()

    with open(opt.testing_file_answer, 'r', encoding='utf-8') as f:
        text_test_answer= f.read()

    with open(opt.training_file, 'r', encoding='utf-8') as f:
        text = f.read()



    if (opt.tokenization_strategy == ''):
        # 100 Tokens
        ascii = string.printable
        chars = list(ascii)
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
    elif(opt.tokenization_strategy == 'GPT2'):
        vocab_size = 50257

    elif opt.tokenization_strategy == 'BPE':
        merges = load_merges('./models/merges.txt')
        vocab = {i : bytes([i]) for i in range(256)}
        for (p0, p1), i in merges.items():
            vocab[i] = vocab[p0] + vocab[p1]
        vocab_size = opt.vocab_size
        print(f"Raw Text: {text_test_prompt}\n")
        # Encoding the prompt using BPE rules
        tokens = list(text_test_prompt.encode('utf-8'))  # Convert the text to a list of UTF-8 byte tokens
        print(f"Tokens: {tokens}\n")
        encoded_text = encode_text_token_basic(tokens, merges)
        print(f"Encoded Text: {encoded_text}")
        context = torch.tensor(encoded_text, device='cuda:0').unsqueeze(dim=-1)
        number_gen = opt.generate_token_number

    model = GPTLanguageModel(vocab_size, opt.n_embd, opt.block_size, opt.dropout, opt.device)
    model = model.to(opt.device)
    model.load_state_dict(torch.load(opt.ckpt))
    model.eval()
    if (opt.tokenization_strategy == ''):
        encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
        decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
        context = torch.tensor(encode(text_test_prompt), device='cuda:0').unsqueeze(dim=-1)
        number_gen = len(text_test_answer)
        response = decode(model.generate(context, max_new_tokens=number_gen,block_size=opt.block_size)[0].tolist())
    elif(opt.tokenization_strategy == 'GPT2'):
        enc = tiktoken.get_encoding('gpt2')
        context = torch.tensor(enc.encode(text_test_prompt), device='cuda:0').unsqueeze(dim=-1)
        number_gen = len(context)
        response = enc.decode(model.generate(context, max_new_tokens=number_gen, block_size=opt.block_size)[0].tolist())
    elif opt.tokenization_strategy == 'BPE':
        merges = load_merges('./models/merges.txt')
        vocab = {i : bytes([i]) for i in range(256)}
        for (p0, p1), i in merges.items():
            vocab[i] = vocab[p0] + vocab[p1]
        with torch.no_grad():
            response_ids = model.generate(context, max_new_tokens=number_gen, block_size=opt.block_size)
        print(f"Raw response_ids: {response_ids[0].tolist()}")
        response = decode_token_basic(response_ids[0].tolist(), vocab)
    with open(opt.testing_file_response, "w") as file:
        file.write(response)
    print(levenshtein_distance(text_test_answer, response))
    if response.strip():
        rouge = Rouge()
        scores = rouge.get_scores(response, text_test_answer)
        print("Rouge:", scores)
        print("Decoded response:", response)
    else:
        print("Generated response was empty.")



if __name__ == "__main__":

    main()

    
