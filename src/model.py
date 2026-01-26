import torch
from transformers import AutoTokenizer 


H_EMBED = 128 
X_EMBED = 128
A_SIZE = H_EMBED + X_EMBED
TOKENIZER = AutoTokenizer.from_pretrained("alibayram/tr_tokenizer")
VOCAB_SIZE = len(TOKENIZER)

class RnnEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.tokenizer = TOKENIZER
        self.vocab_size = len(self.tokenizer)

        self.Wh = torch.nn.Linear(H_EMBED, 128, bias=True)
        self.Cx = torch.nn.Embedding(self.vocab_size, 128)

        self.Wl = torch.nn.Linear(256, 128, bias=False)
    
    def forward(self, x, h):
        # takes x (vocab_size, 1) 
        # takes h (128, 1) 
    
        h_1 = torch.tanh(self.Wh(h.view(1, -1)))
        x_1 = torch.tanh(self.Cx(x).view(1, -1))

        l = torch.cat((h_1, x_1), dim = 1) #(1, 256)

        yh = self.Wl(l)

        return yh

class Classifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.W1 = torch.nn.Linear(H_EMBED, 256, bias=True)
        self.W2 = torch.nn.Linear(256, 3)
    
    def forward(self, x):
        # takes x (vocab_size, 1) 
        # takes h (128, 1) 
    
        x = torch.tanh(self.W1(x.view(1, -1))) # 1, 256
        x = self.W2(x) 

        return x.view(1, -1)


if __name__ == "__main__":
    encoder = RnnEncoder()
    text = "merhaba ben geldim!!!"
    token_ids = torch.tensor(encoder.tokenizer(text)["input_ids"])
    print(token_ids[1])

    y = encoder(token_ids[0], torch.rand(128, 1))
    print(y)
    print(f"Y shape: {y.shape}")


