import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTLanguageModel
import argparse
import string
from utils import get_batch, estimate_loss, levenshtein_distance, get_stats,merge,decode_token_basic,encode_text_token_basic
import tiktoken

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--target_vocab_size', type=int, default=2000,
                        help='target vocab size')
    parser.add_argument('--block_size', type=int, default=256,
                        help='Size of blocks to process vocabulary')
    
    parser.add_argument('--max_iters', type=int, default=256,
                        help='Max Iterations of the Training Process')
    parser.add_argument('--eval_iters', type=int, default=30,
                        help='Eval Iterations of the Training Process')

    parser.add_argument('--eval_interval', type=int, default=256,
                        help='Max Iterations of the Training Process')
    

    # optimization
    parser.add_argument('--learning_rate', type=float, default=.0001,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout')
    parser.add_argument('--weight_decay', type=float, default=.01)
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='basic')
    parser.add_argument('--tokenization_strategy', type=str, default='BPE')
    parser.add_argument('--save_file', type=str, default='./models/test.pth')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='Adam')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of Heads in Attention Block')
    parser.add_argument('--n_layer', type=int, default=12, help='Number of Layers in Attention Block')
    parser.add_argument('--n_embd', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--loss', type=str, default='NLL')
    parser.add_argument('--training_file', type=str, default='./train_data/train_clean.txt')
    parser.add_argument('--training_file_tokenizer', type=str, default='./train_data/train.txt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='train',choices=['shakespeare','train_data'], help='dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--save_merges_file', type=str, default='./models/merges.txt', help='File to save merges')


    opt = parser.parse_args()


    return opt

def save_merges(merges, filename):
    with open(filename, 'w', encoding='utf8') as f:
        for pair, i in merges.items():
            f.write(f"{pair[0]} {pair[1]} {i}\n")
        f.flush()
        f.close()
        print(f"Saved {len(merges)} merges to {filename}")

def main():
    opt = parse_option()

    with open(opt.training_file, 'r', encoding='utf-8') as f:
        text = f.read()

    with open(opt.training_file_tokenizer, 'r', encoding='utf-8') as f:
        text_token = f.read()

    ##################### Note that the Tokenizer is Trained Separately from the LLM ########################
    if(opt.tokenization_strategy == ''):
        ascii = string.printable
        chars = list(ascii)
        vocab_size = len(chars)
        # create a mapping from characters to integers
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
        decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
        data = torch.tensor(encode(text), dtype=torch.long)
    elif(opt.tokenization_strategy == 'BPE'):
        ######################################### We want to merge pairs of encodings together to summarize alphabet ##########
        # Start out with Unicode Encoding

        tokens = text_token.encode('utf-8')
        desired_vocab_size = opt.target_vocab_size

        # Make assumption that your Text Only Has Printable Characters in The English Language
        num_merges = desired_vocab_size - 256
        ids = list(tokens)

        merges = {}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair  = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
        save_merges(merges, opt.save_merges_file)
        
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0,p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        vocab_size = desired_vocab_size
        data = torch.tensor(encode_text_token_basic(text_token,merges), dtype=torch.long)
    elif(opt.tokenization_strategy == 'GPT2'):
        enc = tiktoken.get_encoding('gpt2')
        vocab_size = 50257
        data = torch.tensor(enc.encode(text), dtype=torch.long)

    ##################### Note that the Tokenizer is Trained Separately from the LLM ########################

    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    model = GPTLanguageModel(vocab_size, opt.n_embd, opt.block_size, opt.dropout, opt.device)
    if(opt.ckpt != ''):
        model.load_state_dict(torch.load(opt.ckpt))
    model = model.to(opt.device)


    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    for epoch in range(opt.epochs):
        print(f"Epoch {epoch + 1}/{opt.epochs}")
        for iter in range(opt.max_iters):

            # every once in a while evaluate the loss on train and val sets
            if iter % opt.eval_interval == 0:
                losses = estimate_loss(opt,model,train_data,val_data)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # sample a batch of data
            xb, yb = get_batch('train',train_data,val_data,opt)

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if iter % 255 == 0:  # print final loss
                print(f"Step {iter}, Loss: {loss.item()}")
    save_file = opt.save_file
    torch.save(model.state_dict(), save_file)


if __name__ == "__main__":
    #tokenizer_test()
    main()