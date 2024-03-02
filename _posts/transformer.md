### Resources
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Encoder only vs decoder only vs encoder-decoder

#### Transformer fundamentals and code examples
- [ARENA 3.0: Transformer from Scratch](https://arena3-chapter1-transformer-interp.streamlit.app/[1.1]_Transformer_from_Scratch)
- [Visualizing Seq2seq Models with Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Let's build GPT: from scratch, in code, spelled out](https://youtu.be/kCc8FmEb1nY?si=R69O7ePGfoMg3pmc)
- [nanoGPT](https://github.com/karpathy/nanogpt)
- [picoGPT](https://github.com/jaymody/picoGPT)

  The quadratic complexity issue of attention: multiplying the Q matrix with the K matrix.
  
#### Interpretability: Thinking about what Transformer internals actually do  
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [Interpreting GPT: The Logit Lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

Andrej Karpathy also gives some intuition about the purpose of various architectural details in his youtube walkthrough.

https://www.perfectlynormal.co.uk/blog-transformer-analogy

"We can think of attention as a kind of generalized convolution. Standard convolution layers work by imposing a "prior of locality", i.e. the assumption that pixels which are close together are more likely to share information. Although language has some locality (two words next to each other are more likely to share information than two words 100 tokens apart), the picture is a lot more nuanced, because which tokens are relevant to which others depends on the context of the sentence. For instance, in the sentence "When Mary and John went to the store, John gave a drink to Mary", the names in this sentence are the most important tokens for predicting that the final token will be "Mary", and this is because of the particular context of this sentence rather than the tokens' position.

Attention layers are effectively our way of saying to the transformer, "don't impose a prior of locality, but instead develop your own algorithm to figure out which tokens are important to which other tokens in any given sequence." - ARENA 3.0

#### KV cache
- https://kipp.ly/transformer-param-count/
- https://kipp.ly/transformer-inference-arithmetic/
- [The KV Cache: Memory Usage in Transformers](https://www.youtube.com/watch?v=80bIUggRJf4)
- https://r4j4n.github.io/blogs/posts/kv/

#### Further reading
- [Transformer Circuits Thread](https://transformer-circuits.pub/) - Anthropic's ideas on Transformer interpretability
- [T5](https://arxiv.org/abs/1910.10683) - for understanding the impact of architectural details and training objectives for transformers. (nanoGPT commits also demonstrates performance as architectural pieces are added)
- [How to Train Really Large Models on Many GPUs?](https://lilianweng.github.io/posts/2021-09-25-train-large/) - various parallelism paradigms across multiple GPUs and model architecture and memory saving designs
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) - making inference faster with general network compression techniques and Transformer architecture modifications
- [The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) - Transformer architectural variants
- [Mixture-of-Experts](https://arxiv.org/abs/1701.06538) - for improving training efficiency with a form of parameter sparsity

### Exercises
- https://transformer-circuits.pub/2021/exercises/index.html
- https://github.com/jacobhilton/deep_learning_curriculum/blob/master/1-Transformers.md
  
From [Andrej Karpathy: "Let's build GPT: from scratch, in code, spelled out"](https://youtu.be/kCc8FmEb1nY?si=R69O7ePGfoMg3pmc)
Suggested exercises:
- EX1: The n-dimensional tensor mastery challenge: Combine the `Head` and `MultiHeadAttention` into one class that processes all the heads in parallel, treating the heads as another batch dimension (answer is in nanoGPT).
- EX2: Train the GPT on your own dataset of choice! What other data could be fun to blabber on about? (A fun advanced suggestion if you like: train a GPT to do addition of two numbers, i.e. a+b=c. You may find it helpful to predict the digits of c in reverse order, as the typical addition algorithm (that you're hoping it learns) would proceed right to left too. You may want to modify the data loader to simply serve random problems and skip the generation of train.bin, val.bin. You may want to mask out the loss at the input positions of a+b that just specify the problem using y=-1 in the targets (see CrossEntropyLoss ignore_index). Does your Transformer learn to add? Once you have this, swole doge project: build a calculator clone in GPT, for all of +-*/. Not an easy problem. You may need Chain of Thought traces.)
- EX3: Find a dataset that is very large, so large that you can't see a gap between train and val loss. Pretrain the transformer on this data, then initialize with that model and finetune it on tiny shakespeare with a smaller number of steps and lower learning rate. Can you obtain a lower validation loss by the use of pretraining?
- EX4: Read some transformer papers and implement one additional feature or change that people seem to use. Does it improve the performance of your GPT?

