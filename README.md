This will be a naive implementation of the Transformer Architecture in PyTorch. Transformers address problems in natural language processing through the use of self-attention. This allows the Transformer to understand patterns in linear sequences and reason about them by learning long-range patterns in text. 

**Preliminary Plan**

- [ ] Read the following reference material on Github for implementation specifics by 9/17
    - [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master), Andrej Karpathy's GPT2 Transformer implementation with a custom vocab
    - [X-Transformers](https://github.com/lucidrains/x-transformers), A transformer library built with many custom tooling from a variety of different papers
    - [HuggingFace Transformer's BERT](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert), HugginFace's implementation for BERT


- [ ] Write a system design document for classes and necessary methods for implementation 9/20

- [ ] Build a custom text vocab dataset for training and validation. 10/1
    - [ ] Use tiktoken library to tokenize sentences and phrases. 
    - [ ] Performing web scraping and parsing 
    - [ ] Add important labels

- [ ] Complete core transformer classes and methods 10/10

- [ ] Make a dataset loader module for running experiments to train our transformer on the custom dataset 10/20

- [ ] Wrap-up experiment and document experimental results such as training performance 11/1


**Papers**

- [Attention is All you Need](https://arxiv.org/pdf/1706.03762.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- [Language Model are unsupervised learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

**Resources**

- [hyunwoongko transformer](https://github.com/hyunwoongko/transformer)
- [Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)