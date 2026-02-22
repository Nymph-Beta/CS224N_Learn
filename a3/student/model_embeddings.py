#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N Spring 2024: Assignment 3
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Vera Lin <veralin@stanford.edu>
Siyan Li <siyanli@stanford.edu>
"""

import torch.nn as nn

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layers.
        初始化源语言和目标语言的嵌入层

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        # 嵌入层大小
        self.embed_size = embed_size

        # default values
        # 源语言嵌入层
        self.source = None
        # 目标语言嵌入层
        self.target = None

        # 源语言填充词索引
        src_pad_token_idx = vocab.src['<pad>']
        # 目标语言填充词索引
        tgt_pad_token_idx = vocab.tgt['<pad>']

        ### YOUR CODE HERE (~2 Lines)
        ### TODO - Initialize the following variables:
        ###     self.source (Embedding Layer for source language)
        ###     self.target (Embedding Layer for target langauge)
        ###
        ### Note:
        ###     1. `vocab` object contains two vocabularies:
        ###            `vocab.src` for source
        ###            `vocab.tgt` for target
        ###     2. You can get the length of a specific vocabulary by running:
        ###             `len(vocab.<specific_vocabulary>)`
        ###     3. Remember to include the padding token for the specific vocabulary
        ###        when creating your Embedding.
        ###    vocab.src 和 vocab.tgt 分别是源语言和目标语言的 VocabEntry 对象
        ###    len(vocab.src) 给出源语言词汇量大小，len(vocab.tgt) 给出目标语言词汇量
        ###    src_pad_token_idx 和 tgt_pad_token_idx 已经帮你算好了（都是 0）
        ###    self.embed_size 是嵌入维度
        ###
        ### Use the following docs to properly initialize these variables:
        ###     Embedding Layer:
        ###         https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        
        # 初始化源语言嵌入层
        self.source = nn.Embedding(len(vocab.src), self.embed_size, padding_idx=src_pad_token_idx)
        # 初始化目标语言嵌入层
        self.target = nn.Embedding(len(vocab.tgt), self.embed_size,padding_idx=tgt_pad_token_idx)
        ###    padding_idx: Specifies a padding token, which is ignored by the embedding lookup.
        ### END YOUR CODE


