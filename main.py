# %%
from recurrent_embedded_topic_model.utils import preprocessing
import pandas as pd
import json
import os
from nltk.corpus import stopwords
import re
from gensim.models import KeyedVectors
from gensim import models
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import numpy as np
from collections import Counter
import operator
from embedded_topic_model.models.etm import ETM
from recurrent_embedded_topic_model.models.retm import RETM


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def preprocess(texto):
    #Lower letters
    texto = (texto).lower()

    #Remove stopwords
    stop = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    texto = stop.sub('', texto) 

    #Remove punctation and numbers
    texto = re.sub('[^ña-z]+', ' ', texto)

    return(texto)

# %%
train = 1
graphs = 0
LDA = 1
files = ["movies", "email", "books", "news"]
file = files[3]

if file == "movies":
    df_mov1 = pd.read_csv("data/train_data.txt", sep=" ::: ", names=["id", "Movie", "Genre", "desc"])
    df_mov2 = pd.read_csv("data/test_data.txt", sep=" ::: ", names=["id", "Movie", "desc"])

    df = pd.concat([df_mov1[["desc"]], df_mov2[["desc"]]])[["desc"]]

    df["desc"] = df.desc.apply(preprocess)
    clean_text = df.desc.str.cat(sep=" ")
    freq_words = Counter(clean_text.split())
    vocabulary_words = list(freq_words.keys())
    avg_words = round(np.mean(df.desc.str.split().str.len()))
    print("Average number of words per row: ", avg_words)
    print("Total number of unique words: ", len(vocabulary_words))
    Probabilidad_palabra = {k : v /len(clean_text.split()) for k, v in freq_words.most_common(20)}

    x, y  = zip(*sorted(Probabilidad_palabra.items(),key=operator.itemgetter(1), reverse=True))
    fig = plt.figure(figsize=(15,5))
    plt.bar(x,y, 
            color='darkgreen',
            alpha=0.5)
    plt.xticks(rotation=90, fontsize=12)
    plt.title('Word probability occurance')
    plt.show()
    df.sample(frac=1, random_state=42)
    df = df.head(30000)
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(df, test_size=0.05, random_state=42)
    documents = df.desc.to_list()
    documents_train = X_train.desc.to_list()
    documents_test = X_test.desc.to_list()

elif file == "email":
    df = pd.read_csv("data/Spam_Ham_data.csv", encoding="latin1", names=["email", "label", "subject", "text"])
    df["desc"] = df["subject"] + " " + df["text"]
    df["desc"] = df["desc"].astype(str)
    df["desc"] = df.desc.apply(preprocess)
    clean_text = df.desc.str.cat(sep=" ")
    freq_words = Counter(clean_text.split())
    vocabulary_words = list(freq_words.keys())
    avg_words = round(np.mean(df.desc.str.split().str.len()))
    print("Average number of words per row: ", avg_words)
    print("Total number of unique words: ", len(vocabulary_words))
    Probabilidad_palabra = {k : v /len(clean_text.split()) for k, v in freq_words.most_common(20)}

    x, y  = zip(*sorted(Probabilidad_palabra.items(),key=operator.itemgetter(1), reverse=True))
    fig = plt.figure(figsize=(15,5))
    plt.bar(x,y, 
            color='darkgreen',
            alpha=0.5)
    plt.xticks(rotation=90, fontsize=12)
    plt.title('Word probability occurance')
    plt.show()
    df.sample(frac=1, random_state=42)
    df = df.head(30000)
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(df, test_size=0.05, random_state=42)
    documents = df.desc.to_list()
    documents_train = X_train.desc.to_list()
    documents_test = X_test.desc.to_list()
elif file == "books":
    df = pd.read_csv("data/booksummaries.txt", sep="\t", names=["id", "random", "name", "author", "date", "genres", "desc"])
    df["desc"] = df.desc.apply(preprocess)
    df.sample(frac=1, random_state=42)
    clean_text = df.desc.str.cat(sep=" ")
    freq_words = Counter(clean_text.split())
    vocabulary_words = list(freq_words.keys())
    avg_words = round(np.mean(df.desc.str.split().str.len()))
    print("Average number of words per row: ", avg_words)
    print("Total number of unique words: ", len(vocabulary_words))
    Probabilidad_palabra = {k : v /len(clean_text.split()) for k, v in freq_words.most_common(20)}

    x, y  = zip(*sorted(Probabilidad_palabra.items(),key=operator.itemgetter(1), reverse=True))
    fig = plt.figure(figsize=(15,5))
    plt.bar(x,y, 
            color='darkgreen',
            alpha=0.5)
    plt.xticks(rotation=90, fontsize=12)
    plt.title('Word probability occurance')
    plt.show()
    df = df.head(30000)
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(df, test_size=0.05, random_state=42)
    documents = df.desc.to_list()
    documents_train = X_train.desc.to_list()
    documents_test = X_test.desc.to_list()
elif file == "news":
    df = pd.read_csv("data/news.csv")
    df["desc"] = df.texto_completo.apply(preprocess)
    df.sample(frac=1, random_state=42)
    clean_text = df.desc.str.cat(sep=" ")
    freq_words = Counter(clean_text.split())
    vocabulary_words = list(freq_words.keys())
    avg_words = round(np.mean(df.desc.str.split().str.len()))
    print("Average number of words per row: ", avg_words)
    print("Total number of unique words: ", len(vocabulary_words))
    Probabilidad_palabra = {k : v /len(clean_text.split()) for k, v in freq_words.most_common(20)}

    x, y  = zip(*sorted(Probabilidad_palabra.items(),key=operator.itemgetter(1), reverse=True))
    fig = plt.figure(figsize=(15,5))
    plt.bar(x,y, 
            color='darkgreen',
            alpha=0.5)
    plt.xticks(rotation=90, fontsize=12)
    plt.title('Word probability occurance')
    plt.show()
    df = df.head(30000)
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(df, test_size=0.05, random_state=42)
    documents = df.desc.to_list()
    documents_train = X_train.desc.to_list()
    documents_test = X_test.desc.to_list()

# %%

# Preprocessing the dataset
vocabulary, train_dataset, test_dataset, = preprocessing.create_etm_datasets(
    documents, 
    min_df=0.01, 
    max_df=0.75, 
    train_size=0.95, #anteriormente 85
)

from embedded_topic_model.utils import embedding

# Training word2vec embeddings
if train == 1:
    embeddings_mapping = embedding.create_word2vec_embedding_from_dataset(documents, embedding_file_path="embeds", save_c_format_w2vec=True)
else:
    embeddings_mapping = models.KeyedVectors.load_word2vec_format('embeds.bin', binary=True)

# %%
retm_instance = RETM(
    vocabulary,
    embeddings=embeddings_mapping, # You can pass here the path to a word2vec file or
                                # a KeyedVectors instance
    num_topics=50,
    epochs=12,
    debug_mode=True,
    t_hidden_size=8,
    batch_size=6,
    eval_batch_size=6,
    eval_perplexity=False,
    lr=0.01,
    wdecay=1.0e-5,
    avg_words=avg_words,
    train_embeddings=False, # Optional. If True, ETM will learn word embeddings jointly with
                            # topic embeddings. By default, is False. If 'embeddings' argument
                            # is being passed, this argument must not be True
)

retm_instance.fit(train_dataset, test_dataset)



perplex = retm_instance.train_perplexity
perplex_test = retm_instance._perplexity(test_dataset)
topics, topic_values = retm_instance.get_topics(20)
topic_coherence = retm_instance.get_topic_coherence()
topic_diversity = retm_instance.get_topic_diversity()
model = "R-ETM"



with open("metrics.json", "r") as jsonFile:
        data = json.load(jsonFile)

if file not in data[model].keys():
    data[model][file] = {"perplexity_train": str(perplex), "perplexity_test":str(perplex_test), "topic_words": topics, "word_values": str(topic_values), "topic_coherence": str(topic_coherence), "topic_diversity": str(topic_diversity)}


with open("metrics.json", "w") as jsonFile:
    json.dump(data, jsonFile)


if graphs == 1:
    tsne = TSNE(n_components=2)
    word_vocab = embeddings_mapping.vocab.keys()
    X_tsne = tsne.fit_transform(embeddings_mapping[list(word_vocab)])
    df_tsne = pd.DataFrame(X_tsne, index=word_vocab, columns=['x', 'y'])

    for topic in range(len(topics)): 

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        colors = cm.rainbow(np.linspace(0, 1, len(topics[topic])))

        df_topic = df_tsne.loc[df_tsne.index.isin(topics[topic])].reindex(topics[topic])
        df_topic["importance"] = topic_values[topic]
        df_topic["importance"] = df_topic.importance * 10003
        ax.scatter(df_topic['x'], df_topic['y'], s=df_topic["importance"], color=colors)
        
        del df_topic["importance"]

        for word, pos in df_topic.iterrows():
            ax.annotate(word, pos)

        plt.show()



etm_instance = ETM(
    vocabulary,
    embeddings=embeddings_mapping, # You can pass here the path to a word2vec file or
                                # a KeyedVectors instance
    num_topics=50,
    epochs=12,
    debug_mode=True,
    t_hidden_size=8,
    batch_size=6,
    eval_batch_size=6,
    eval_perplexity=False,
    lr=0.01,
    wdecay=1.0e-5,
    train_embeddings=False, # Optional. If True, ETM will learn word embeddings jointly with
                            # topic embeddings. By default, is False. If 'embeddings' argument
                            # is being passed, this argument must not be True
)

etm_instance.fit(train_dataset, test_dataset)

#%%

perplex = etm_instance.train_perplexity
perplex_test = etm_instance._perplexity(test_dataset)
topics, topic_values = etm_instance.get_topics(20)
topic_coherence = etm_instance.get_topic_coherence()
topic_diversity = etm_instance.get_topic_diversity()
model = "ETM"

with open("metrics.json", "r") as jsonFile:
        data = json.load(jsonFile)

if file not in data[model].keys():
    data[model][file] = {"perplexity_train": str(perplex), "perplexity_test":str(perplex_test), "topic_words": topics, "word_values": str(topic_values), "topic_coherence": str(topic_coherence), "topic_diversity": str(topic_diversity)}

with open("metrics.json", "w") as jsonFile:
    json.dump(data, jsonFile)


if graphs == 1:
    tsne = TSNE(n_components=2)
    word_vocab = embeddings_mapping.vocab.keys()
    X_tsne = tsne.fit_transform(embeddings_mapping[list(word_vocab)])
    df_tsne = pd.DataFrame(X_tsne, index=word_vocab, columns=['x', 'y'])

    for topic in range(len(topics)): 

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        colors = cm.rainbow(np.linspace(0, 1, len(topics[topic])))

        df_topic = df_tsne.loc[df_tsne.index.isin(topics[topic])].reindex(topics[topic])
        df_topic["importance"] = topic_values[topic]
        df_topic["importance"] = df_topic.importance * 10000
        ax.scatter(df_topic['x'], df_topic['y'], s=df_topic["importance"], color=colors)
        
        del df_topic["importance"]

        for word, pos in df_topic.iterrows():
            ax.annotate(word, pos)

        plt.show()

#%%
if LDA ==  1:
    from sklearn.feature_extraction.text import CountVectorizer 
    from sklearn.decomposition import LatentDirichletAllocation


    cv = CountVectorizer()
    mdt_frec = cv.fit_transform(df['desc']) 
    terminos= cv.get_feature_names()
    X = pd.DataFrame(mdt_frec.todense(), 
                                index=df.index, 
                                columns=terminos)

    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(X, test_size=0.05, random_state=42)

    lda_model = LatentDirichletAllocation(n_components=50, learning_method='online', random_state=42, max_iter=1000)
    lda_model.fit(X_train)
    lda_pplxity = lda_model.perplexity(X_train)
    lda_pplxity_test = lda_model.perplexity(X_test)

    model = "LDA"

    with open("metrics.json", "r") as jsonFile:
            data = json.load(jsonFile)

    if file not in data[model].keys():
        data[model][file] = {"perplexity_train": str(lda_pplxity), "perplexity_test":str(lda_pplxity_test)}

    with open("metrics.json", "w") as jsonFile:
        json.dump(data, jsonFile)

    import pickle
    # file name, I'm using *.pickle as a file extension
    filename = "lda_summaries.pickle"

    # save model
    pickle.dump(lda_model, open(filename, "wb"))


# %%
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
import nltk
from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.evaluation.measures import CoherenceNPMI

stopwords = list(stopwords.words("english"))

sp = WhiteSpacePreprocessingStopwords(documents_train, stopwords_list=stopwords)
preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = sp.preprocess()

sp_test = WhiteSpacePreprocessingStopwords(documents_test, stopwords_list=stopwords)
preprocessed_documents_test, unpreprocessed_corpus_test, vocab_test, retained_indices_test = sp.preprocess()

tp = TopicModelDataPreparation("paraphrase-multilingual-mpnet-base-v2")
# %%
training_dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)
testing_dataset = tp.fit(text_for_contextual=unpreprocessed_corpus_test, text_for_bow=preprocessed_documents_test)
# %%
ctm = ZeroShotTM(bow_size=len(tp.vocab), contextual_size=768, n_components=50, num_epochs=20)
ctm.fit(training_dataset, validation_dataset=testing_dataset) # run the model
topics_predictions = ctm.get_thetas(training_dataset, n_samples=5) # get all the topic predictions
zero_pplxity = ctm.pplxity_train
zero_pplxity_test = ctm.pplxity_val
textos_coherence = [x.split() for x in documents]
npmi = CoherenceNPMI(texts=textos_coherence, topics=ctm.get_topic_lists())
score = npmi.score()
print(score)
with open("metrics.json", "r") as jsonFile:
        data = json.load(jsonFile)
#data["ZeroShotTM"] = {}

if file not in data["ZeroShotTM"].keys():
    data["ZeroShotTM"][file] = {"perplexity_train": str(zero_pplxity), "perplexity_test":str(zero_pplxity_test), "topic_words": ctm.get_topic_lists(), "topic_coherence": str(score)}

with open("metrics.json", "w") as jsonFile:
    json.dump(data, jsonFile)

# %%

ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=50, num_epochs=20)
ctm.fit(training_dataset, validation_dataset=testing_dataset) # run the model
topics_predictions = ctm.get_thetas(training_dataset, n_samples=5) # get all the topic predictions
zero_pplxity = ctm.pplxity_train
zero_pplxity_test = ctm.pplxity_val
textos_coherence = [x.split() for x in documents]
npmi = CoherenceNPMI(texts=textos_coherence, topics=ctm.get_topic_lists())
score = npmi.score()
print(score)
with open("metrics.json", "r") as jsonFile:
        data = json.load(jsonFile)

#data["CombinedTM"] = {}

if file not in data["CombinedTM"].keys():
    data["CombinedTM"][file] = {"perplexity_train": str(zero_pplxity), "perplexity_test":str(zero_pplxity_test), "topic_words": ctm.get_topic_lists(), "topic_coherence": str(score)}

with open("metrics.json", "w") as jsonFile:
    json.dump(data, jsonFile)

# %%
### Perplexity graphs
import seaborn as sns
with open("metrics.json", "r") as jsonFile:
    data = json.load(jsonFile)

df_pplxity = pd.DataFrame()
for file in files:
    for models in data.keys():
        if models == "LDA":
            df_new=pd.DataFrame(data={"perplexity_train":float(data[models][file]["perplexity_train"]), \
                                    "perplexity_test":float(data[models][file]["perplexity_test"]), \
                                    "file":file}, index=[models])
        else:
            df_new=pd.DataFrame(data={"perplexity_train":float(data[models][file]["perplexity_train"]), \
                                    "perplexity_test":float(data[models][file]["perplexity_test"]), \
                                    "topic_coherence":float(data[models][file]["topic_coherence"]), \
                                    "topic_diversity":float(data[models][file]["topic_diversity"]), \
                                    "file":file}, index=[models])
        df_pplxity = pd.concat([df_pplxity, df_new])


#create grouped bar chart
df_pplxity.reset_index(inplace=True)
df_pplxity.rename(columns={"index":"model"}, inplace=True)
df_pplxity.to_csv("model_metrics.csv")
sns.set(style='white')
sns.barplot(x=df_pplxity.file, y=np.log2(df_pplxity.perplexity_train), hue=df_pplxity.model, data=df_pplxity) 
plt.show()
#Test
sns.barplot(x=df_pplxity.file, y=np.log2(df_pplxity.perplexity_test), hue=df_pplxity.model, data=df_pplxity)
plt.show()
#Coherence
sns.barplot(x=df_pplxity.file, y=df_pplxity.topic_coherence, hue=df_pplxity.model, data=df_pplxity.loc[df_pplxity.model != "LDA"])
plt.show()
#Diversity
sns.barplot(x=df_pplxity.file, y=df_pplxity.topic_diversity, hue=df_pplxity.model, data=df_pplxity.loc[df_pplxity.model != "LDA"])
plt.show()



# %%
