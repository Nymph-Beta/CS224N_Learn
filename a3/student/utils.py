#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2022-23: Homework 4
utils.py: Utility Functions
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
Siyan Li <siyanli@stanford.edu>
Moussa KB Doumbouya <moussa@stanford.edu>
"""

from typing import List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import sentencepiece as spm
nltk.download('punkt')


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
        找到 batch 中最长的句子长度，将较短句子用 pad_token 填充到相同长度
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    # 找到 batch 中最长的句子长度
    max_len = max(len(sent) for sent in sents)
    # 将较短句子用 pad_token 填充到相同长度
    for sent in sents:
        if len(sent) < max_len:
            # 将较短句子用 pad_token 填充到相同长度
            sent.extend([pad_token] * (max_len - len(sent)))
            # 将句子添加到 sents_padded 中
            sents_padded.append(sent)
        # 如果句子长度已经等于最长句子长度，则直接添加到 sents_padded 中
        else:
            sents_padded.append(sent)

    ### END YOUR CODE

    return sents_padded


def read_corpus(file_path, source, vocab_size=2500):
    """ Read file, where each sentence is dilineated by a `\n`.
    使用 sentencepiece 将句子分词，并添加 <s> 和 </s> 标签
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    @param vocab_size (int): number of unique subwords in
        vocabulary when reading and tokenizing
    """
    data = []
    # 加载 sentencepiece 模型
    sp = spm.SentencePieceProcessor()
    sp.load('{}.model'.format(source))
    # 读取文件
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            # 使用 sentencepiece 将句子分词
            subword_tokens = sp.encode_as_pieces(line)
            # 使用 sentencepiece 将句子分词，并添加 <s> 和 </s> 标签
            # only append <s> and </s> to the target sentence
            if source == 'tgt':
                subword_tokens = ['<s>'] + subword_tokens + ['</s>']
            data.append(subword_tokens)

    return data


def autograder_read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    使用 nltk 将句子分词，并添加 <s> 和 </s> 标签
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    # 使用 nltk 将句子分词，并添加 <s> 和 </s> 标签
    for line in open(file_path):
        # 使用 nltk 将句子分词
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        # 如果 source 是 target，则添加 <s> 和 </s> 标签
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        # 将句子添加到 data 中
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    按句子长度从大到小排序，生成 batch
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    # 计算 batch 数量
    batch_num = math.ceil(len(data) / batch_size)
    # 生成索引数组
    index_array = list(range(len(data)))

    if shuffle:
        # 随机打乱索引数组
        np.random.shuffle(index_array)

    for i in range(batch_num):
        # 生成 batch 索引
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        # 生成 batch 数据
        examples = [data[idx] for idx in indices]
        # 按句子长度从大到小排序
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        # 生成 batch 源句子
        src_sents = [e[0] for e in examples]
        # 生成 batch 目标句子
        tgt_sents = [e[1] for e in examples]

        # 生成 batch 数据
        yield src_sents, tgt_sents


