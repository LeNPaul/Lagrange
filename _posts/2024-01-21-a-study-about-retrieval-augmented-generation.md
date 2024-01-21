---
layout: post
title: "A Study About Retrieval Augmented Generation"
author: "Mohammadreza Ghofrani"
categories: journal
tags: [rag]
image: forest.jpg
---

"Why Was Sam Altman Fired As CEO of OpenAI?", this is one of the questions that ChatGPT cannot answer
by default. In this case the response of ChatGPT is not bad, It basically says "I don't know." and
quails answering to the question.

> I'm sorry, but that statement is not accurate. As of my last knowledge update in January 2022, Sam Altman was the CEO of OpenAI. If there have been any changes or developments since then, I would not be aware of them. I recommend checking the latest and most reliable sources for the most up-to-date information on OpenAI's leadership.

However the ChatGPT isn't always good at saying "I don't know". It is possible to provide a
wrong answer to the question. For instance, here I have asked ChatGPT "When does the Habsburg Empire founded?",
In response it said:

> The Habsburg Empire, also known as the Austro-Hungarian Empire, was founded in 1804. It officially came into existence when Emperor Francis II proclaimed himself Emperor of Austria. The empire lasted until the end of World War I in 1918, after which it was dissolved.

However, we know that the Habsburg Empire lasted for nearly as 6 centeries and it is not possible to
start at 1804 and ended in 1918. This problem is known as the *hallucination*. In simple terms,
hallunication is the problem that LLMs tend to think "they *know* everything.", therefore they
answer every question asked even with the wrong or inaccurate information.

One of the solutions of the hallunication problem is to augment relevant documents alongside the
given prompt. Model uses the information provided in the documents while generating response, crafting
a better and precise response.

Retrieval Augmented Generation (RAG) as it names suggests consists of three components: Retriever,
Augmenter and Generator. Retriever fetches the most relevant information from the knowledge base,
Augmenter concats the query with the given docuemnts, and Generator that crafts the response to the
user's query.

The studies in this field can be devided into two classes: fine-grained and End-to-End. Fine-grained
studies tend to tackle with Just one of the components of the RAG architecture; However, End-to-End
methods modify the architecture as whole.

# Fine-grained

## Retreival fine-graining

Most of the ideas in this part has came from the Information Retrieval (IR) field in the NLP.
The main questions these researches try to answer are "How to effectively represent a document as a vector?",
"What is the best method to chunk the document?", or "How to effectively index the documents?".

One of the interesting ideas which can be classified here is the idea proposed in
paper "GENERATE RATHER THAN RETRIEVE", where they have proposed to utilize an LLM to generate a
document based on user query and then give generated document to the 2nd phase to augment and
generate the response.

## Augmentation fine-graining

The studies here, focus on finding the optimum qurey and best way to append the retrieved document
to craft an effective prompt.
For instance, studies show that LLMs tend to attend more to the documents who are located at first
or the last part of the given prompt. Therefore, some of the studies after finding top $K$ documents
rerank them and put the most important document as the first augmented document. Others tend to extract
main information out of the text not only to help LLM to grasp important ideas but also to face
context limitation of the studies. Some other studies try to curate best prompt for each use case,
therefore they propose a *routing* functionality to choose best prompt out of the available prompts.

## Generation fine-graining

The studies focusing on the generation are mostly include the ideas from text generation field in the
NLP. Before introduction of LLMs most researches tried fine-tuning the models such as T5,
which have a good generation capability, for the RAG. Recently due to the good performance of LLMs,
most of the researchers use the model out of box without any specific fine-tuning.

# End-to-End

Some researches like RETRO tend to train the whole RAG architecture as a whole. The idea
behind these researches is better alignment of generative model and the retrieval model. Other studies
also try to provide feedback from generator to the retrieval to imporove retrieved documents quality and
ease the task of generator.
