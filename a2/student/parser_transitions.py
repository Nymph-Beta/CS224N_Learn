#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2023-2024: Homework 2
parser_transitions.py: Algorithms for completing partial parsess.
Sahil Chopra <schopra8@stanford.edu>
Haoshen Hong <haoshen@stanford.edu>
"""

import sys

class PartialParse(object):
    def __init__(self, sentence):
        """Initializes this partial parse.

        @param sentence (list of str): The sentence to be parsed as a list of words.
                                        Your code should not modify the sentence.
        """
        # The sentence being parsed is kept for bookkeeping purposes. Do NOT alter it in your code.
        self.sentence = sentence

        ### YOUR CODE HERE (3 Lines)
        ### Your code should initialize the following fields:
        ###     self.stack: The current stack represented as a list with the top of the stack as the
        ###                 last element of the list.
        ###     self.buffer: The current buffer represented as a list with the first item on the
        ###                  buffer as the first item of the list
        ###     self.dependencies: The list of dependencies produced so far. Represented as a list of
        ###             tuples where each tuple is of the form (head, dependent).
        ###             Order for this list doesn't matter.
        ###
        ### Note: The root token should be represented with the string "ROOT"
        ### Note: If you need to use the sentence object to initialize anything, make sure to not directly 
        ###       reference the sentence object.  That is, remember to NOT modify the sentence object. 

        self.stack = ["ROOT"]
        self.buffer = list(sentence)
        self.dependencies = []

        ### END YOUR CODE


    def parse_step(self, transition):
        """Performs a single parse step by applying the given transition to this partial parse

        @param transition (str): A string that equals "S", "LA", or "RA" representing the shift,
                                left-arc, and right-arc transitions. You can assume the provided
                                transition is a legal transition.
        """
        ### YOUR CODE HERE (~7-12 Lines)
        ### TODO:
        ###     Implement a single parsing step, i.e. the logic for the following as
        ###     described in the pdf handout:
        ###         1. Shift
        ###         2. Left Arc
        ###         3. Right Arc

        # SHIFT
        if transition == "S":
            if len(self.buffer) == 0:
                return
            self.stack.append(self.buffer.pop(0))

        # LEFT-ARC
        if transition == "LA":
            if len(self.stack) < 2:
                return
            dependent = self.stack[-2]
            head = self.stack[-1]
            self.dependencies.append((head, dependent))
            # self.stack = self.stack[:-2] + [head]
            self.stack.pop(-2)

        # RIGHT-ARC
        if transition == "RA":
            if len(self.stack) < 2:
                return
            dependent = self.stack[-1]
            head = self.stack[-2]
            self.dependencies.append((head, dependent))
            # self.stack = self.stack[:-1]
            self.stack.pop(-1)


        ### END YOUR CODE

    def parse(self, transitions):
        """Applies the provided transitions to this PartialParse

        @param transitions (list of str): The list of transitions in the order they should be applied

        @return dependencies (list of string tuples): The list of dependencies produced when
                                                        parsing the sentence. Represented as a list of
                                                        tuples where each tuple is of the form (head, dependent).
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.

    @param sentences (list of list of str): A list of sentences to be parsed
                                            (each sentence is a list of words and each word is of type string)
    @param model (ParserModel): The model that makes parsing decisions. It is assumed to have a function
                                model.predict(partial_parses) that takes in a list of PartialParses as input and
                                returns a list of transitions predicted for each parse. That is, after calling
                                    transitions = model.predict(partial_parses)
                                transitions[i] will be the next transition to apply to partial_parses[i].
    @param batch_size (int): The number of PartialParses to include in each minibatch


    @return dependencies (list of dependency lists): A list where each element is the dependencies
                                                    list for a parsed sentence. Ordering should be the
                                                    same as in sentences (i.e., dependencies[i] should
                                                    contain the parse for sentences[i]).
    """
    dependencies = []

    ### YOUR CODE HERE (~8-10 Lines)
    ### TODO:
    ###     Implement the minibatch parse algorithm.  Note that the pseudocode for this algorithm is given in the pdf handout.
    ###
    ###     Note: A shallow copy (as denoted in the PDF) can be made with the "=" sign in python, e.g.
    ###                 unfinished_parses = partial_parses[:].
    ###             Here `unfinished_parses` is a shallow copy of `partial_parses`.
    ###             In Python, a shallow copied list like `unfinished_parses` does not contain new instances
    ###             of the object stored in `partial_parses`. Rather both lists refer to the same objects.
    ###             In our case, `partial_parses` contains a list of partial parses. `unfinished_parses`
    ###             contains references to the same objects. Thus, you should NOT use the `del` operator
    ###             to remove objects from the `unfinished_parses` list. This will free the underlying memory that
    ###             is being accessed by `partial_parses` and may cause your code to crash.

    # 1. 初始化 PartialParse 对象
    partial_parses = [PartialParse(sentence) for sentence in sentences]

    # 2. 初始化未完成列表（浅拷贝）
    # 这里存的是对象的引用，所以修改 unfinished_parses 里的对象，
    # partial_parses 里的对象也会跟着变。
    unfinished_parses = partial_parses[:]

    # 3. 大循环：只要还有没做完的句子
    while len(unfinished_parses) > 0:
        # A. 取出一个 minibatch
        # 注意：如果剩余不足 batch_size，切片操作会自动取到末尾，不会报错
        minibatch = unfinished_parses[:batch_size]

        # B. 让模型批量预测下一步动作
        # 假设 model.predict 接受一个 PartialParse 列表，返回一个动作字符串列表
        transitions = model.predict(minibatch)

        # C. 批量执行动作
        # zip 把句子和对应的动作配对
        for parse, transition in zip(minibatch, transitions):
            parse.parse_step(transition)

        # D. 移除已经做完的句子
        # 这是一个关键点：在遍历列表时不能直接删除元素。
        # 最简单且不易出错的方法是：重构 unfinished_parses 列表。
        # 只保留那些“没做完”的句子。
        
        # 如何判断没做完？ Stack 还没空(>1 因为有ROOT) 或者 Buffer 还没空
        unfinished_parses = [p for p in unfinished_parses if len(p.stack) > 1 or len(p.buffer) > 0]

    # E. 从 partial_parses 中收集每个句子的依赖关系列表
    # 因为 partial_parses 和 unfinished_parses 共享同一批 PartialParse 对象（浅拷贝），
    # 所以虽然 while 循环中只操作了 unfinished_parses 里的对象，
    # 但 partial_parses 里的对象也已经被更新了（它们是同一个对象）。
    # 此外，partial_parses 保持了原始顺序，保证 dependencies[i] 对应 sentences[i]。
    dependencies = [pp.dependencies for pp in partial_parses]

    ### END YOUR CODE

    return dependencies


def test_step(name, transition, stack, buf, deps,
              ex_stack, ex_buf, ex_deps):
    """Tests that a single parse step returns the expected output"""
    pp = PartialParse([])
    pp.stack, pp.buffer, pp.dependencies = stack, buf, deps

    pp.parse_step(transition)
    stack, buf, deps = (tuple(pp.stack), tuple(pp.buffer), tuple(sorted(pp.dependencies)))
    assert stack == ex_stack, \
        "{:} test resulted in stack {:}, expected {:}".format(name, stack, ex_stack)
    assert buf == ex_buf, \
        "{:} test resulted in buffer {:}, expected {:}".format(name, buf, ex_buf)
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)
    print("{:} test passed!".format(name))


def test_parse_step():
    """Simple tests for the PartialParse.parse_step function
    Warning: these are not exhaustive
    """
    test_step("SHIFT", "S", ["ROOT", "the"], ["cat", "sat"], [],
              ("ROOT", "the", "cat"), ("sat",), ())
    test_step("LEFT-ARC", "LA", ["ROOT", "the", "cat"], ["sat"], [],
              ("ROOT", "cat",), ("sat",), (("cat", "the"),))
    test_step("RIGHT-ARC", "RA", ["ROOT", "run", "fast"], [], [],
              ("ROOT", "run",), (), (("run", "fast"),))


def test_parse():
    """Simple tests for the PartialParse.parse function
    Warning: these are not exhaustive
    """
    sentence = ["parse", "this", "sentence"]
    dependencies = PartialParse(sentence).parse(["S", "S", "S", "LA", "RA", "RA"])
    dependencies = tuple(sorted(dependencies))
    expected = (('ROOT', 'parse'), ('parse', 'sentence'), ('sentence', 'this'))
    assert dependencies == expected,  \
        "parse test resulted in dependencies {:}, expected {:}".format(dependencies, expected)
    assert tuple(sentence) == ("parse", "this", "sentence"), \
        "parse test failed: the input sentence should not be modified"
    print("parse test passed!")


class DummyModel(object):
    """Dummy model for testing the minibatch_parse function
    """
    def __init__(self, mode = "unidirectional"):
        self.mode = mode

    def predict(self, partial_parses):
        if self.mode == "unidirectional":
            return self.unidirectional_predict(partial_parses)
        elif self.mode == "interleave":
            return self.interleave_predict(partial_parses)
        else:
            raise NotImplementedError()

    def unidirectional_predict(self, partial_parses):
        """First shifts everything onto the stack and then does exclusively right arcs if the first word of
        the sentence is "right", "left" if otherwise.
        """
        return [("RA" if pp.stack[1] == "right" else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]

    def interleave_predict(self, partial_parses):
        """First shifts everything onto the stack and then interleaves "right" and "left".
        """
        return [("RA" if len(pp.stack) % 2 == 0 else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]

def test_dependencies(name, deps, ex_deps):
    """Tests the provided dependencies match the expected dependencies"""
    deps = tuple(sorted(deps))
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)


def test_minibatch_parse():
    """Simple tests for the minibatch_parse function
    Warning: these are not exhaustive
    """

    # Unidirectional arcs test
    sentences = [["right", "arcs", "only"],
                 ["right", "arcs", "only", "again"],
                 ["left", "arcs", "only"],
                 ["left", "arcs", "only", "again"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'right'), ('arcs', 'only'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[1],
                      (('ROOT', 'right'), ('arcs', 'only'), ('only', 'again'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[2],
                      (('only', 'ROOT'), ('only', 'arcs'), ('only', 'left')))
    test_dependencies("minibatch_parse", deps[3],
                      (('again', 'ROOT'), ('again', 'arcs'), ('again', 'left'), ('again', 'only')))

    # Out-of-bound test
    sentences = [["right"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0], (('ROOT', 'right'),))

    # Mixed arcs test
    sentences = [["this", "is", "interleaving", "dependency", "test"]]
    deps = minibatch_parse(sentences, DummyModel(mode="interleave"), 1)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'is'), ('dependency', 'interleaving'),
                      ('dependency', 'test'), ('is', 'dependency'), ('is', 'this')))
    print("minibatch_parse test passed!")


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        raise Exception("You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")
    elif args[1] == "part_c":
        test_parse_step()
        test_parse()
    elif args[1] == "part_d":
        test_minibatch_parse()
    else:
        raise Exception("You did not provide a valid keyword. Either provide 'part_c' or 'part_d', when executing this script")
