import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from nltk.stem import WordNetLemmatizer
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def chinese_tokenizer(text):
    # 使用jieba进行中文分词
    tokens = jieba.lcut(text)
    return tokens


def chatbot_chinese(query, df):
    # 使用TfidfVectorizer构建TF-IDF模型
    tfidf = TfidfVectorizer(tokenizer=chinese_tokenizer, token_pattern=None)

    factors = tfidf.fit_transform(df['Query']).toarray()

    # 对输入的查询进行TF-IDF转换
    query_vector = tfidf.transform([query]).toarray()

    # 使用余弦相似度计算相似度
    similar_score = cosine_similarity(factors, query_vector)

    # 获取最相似的问题的索引
    index = np.argmax(similar_score)

    # 获取匹配的问题和回应
    matching_question = df.loc[index]['Query']
    response_dict = df.loc[index]['Response']
    confidence = similar_score[index][0]

    chat_dict = {'match': matching_question,
                 'response': response_dict,
                 'score': confidence}
    return chat_dict


def chatbot(query):
    legitimatize = WordNetLemmatizer()
    tfidf = TfidfVectorizer()

    factors = tfidf.fit_transform(df['Query']).toarray()
    # feature_names = tfidf.get_feature_names_out()

    # step:-1 clean
    query = legitimatize.lemmatize(query)
    # step:-2 word embedding - transform
    query_vector = tfidf.transform([query]).toarray()
    # step-3: cosine similarity
    similar_score = 1 - cosine_distances(factors, query_vector)
    index = similar_score.argmax()  # take max index position
    # searching or matching question
    matching_question = df.loc[index]['Query']
    response_dict = df.loc[index]['Response']
    confidence = similar_score[index][0]
    chat_dict = {'match': matching_question,
                 'response': response_dict,
                 'score': confidence}
    return chat_dict


if __name__ == '__main__':
    df = pd.read_csv('chatbot.txt', names=('Query', 'Response'), sep=('|'))

    while True:
        query = input('USER: ')
        if query == 'exit':
            break

        response = chatbot_chinese(query, df)  # 传递数据框作为参数
        print("Score: ", response['score'])
        print("Len:", len(query))
        # Todo 如果是英文 长度不能这么处理，如果是英文的话判断条件再改
        if response['score'] <= 0.9 and len(query) < 5:
            print('BOT: Please rephrase your Question.')
        elif response['score'] <= 0.6 and len(query) >= 5:
            print('BOT: Please rephrase your Question.')
        else:
            print('BOT: ', response['response'])
