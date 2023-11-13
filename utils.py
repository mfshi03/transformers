import torch

def generate_topk(model, initial_seq, max_length=100, k=50):
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(encode(initial_seq)).unsqueeze(0).to(device)
        generated = input_seq.tolist()

        for _ in range(max_length - len(initial_seq)):
            output = model(input_seq, input_seq) # [batch_size, seq_len, vocab_size]
            # Get the probabilities for the last token in the sequence
            probs = output[0, -1, :]
            # Take the top-k tokens only
            top_k_probs, top_k_idx = torch.topk(probs, k=k, dim=-1)
            # Sample from the top-k tokens instead of doing an argmax
            next_token = torch.multinomial(F.softmax(top_k_probs, dim=-1), num_samples=1).item()
            next_token = top_k_idx[next_token].item()
            generated[0].append(next_token)
            input_seq = torch.tensor(generated).to(device)

        generated_seq = decode(generated[0])
        
    return generated_seq