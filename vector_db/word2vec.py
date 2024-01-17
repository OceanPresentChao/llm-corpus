import os
from gensim.models import Word2Vec
import jieba
import numpy as np
from numpy import ndarray
from typing import List, Union, Dict
from os import path

class Word2VecManager:
    """Pre-trained word2vec embedding"""
    def __init__(self, model_name_or_path: str,
                 stopwords: List[str] = None):
        """
        Init word2vec model

        Args:
            model_name_or_path: word2vec file path
                https://radimrehurek.com/gensim/models/keyedvectors.html#module-gensim.models.keyedvectors
            stopwords: list, stopwords
        """
        from gensim.models import KeyedVectors  # noqa

        self.stopwords = stopwords if stopwords else load_stopwords(default_stopwords_file)
        self.model = Word2Vec.load(model_name_or_path)
        self.w2v = self.model.wv
        self.jieba = jieba

    def __str__(self):
        return f"<Word2Vec, word count: {len(self.w2v.key_to_index)}, emb size: {self.w2v.vector_size}, " \
               f"stopwords count: {len(self.stopwords)}>"

    def encode(self, sentences: Union[List[str], str]) -> ndarray:
        """
        Encode sentences to vectors
        """
        if self.w2v is None:
            raise ValueError('No model for embed sentence')

        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        for sentence in sentences:
            emb = []
            count = 0
            for word in sentence:
                # 过滤停用词
                if word in self.stopwords:
                    continue
                # 调用词向量
                if word in self.w2v.key_to_index:
                    emb.append(self.w2v.get_vector(word, norm=True))
                    count += 1
                else:
                    if len(word) == 1:
                        continue
                    # 再切分
                    ws = self.jieba.lcut(word, cut_all=True, HMM=True)
                    for w in ws:
                        if w in self.w2v.key_to_index:
                            emb.append(self.w2v.get_vector(w, norm=True))
                            count += 1
            tensor_x = np.array(emb).sum(axis=0)  # 纵轴相加
            if count > 0:
                avg_tensor_x = np.divide(tensor_x, count)
            else:
                avg_tensor_x = np.zeros(self.w2v.vector_size, dtype=float)
            all_embeddings.append(avg_tensor_x)
        all_embeddings = np.array(all_embeddings, dtype=float)
        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings