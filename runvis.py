#!/usr/bin/python
# encoding=utf-8
from subprocess import call
from glob import glob
from nltk.corpus import stopwords
import os, struct
from tensorflow.core.example import example_pb2
import pyrouge
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import *
stemmer = PorterStemmer()

ratio = 1
duc_num = 5
cmd = '/root/miniconda2/bin/python run_summarization.py --mode=eval --single_pass=0 --coverage=True --vocab_path=finished_files/vocab --log_root=log --exp_name=myexperiment --data_path=test/temp_file'
#cmd = '/root/miniconda2/bin/python ../pointer-generator-master/run_summarization.py --mode=decode --single_pass=1 --coverage=True --vocab_path=finished_files/vocab --log_root=log --exp_name=myexperiment --data_path=test/temp_file'

generated_path = '/gttp/pointer-generator-tal/log/myexperiment/decode_test_400maxenc_4beam_35mindec_100maxdec_ckpt-238410/'
#generated_path = '/gttp/pointer-generator-master/log/myexperiment/decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/'
cmd = cmd.split()
stopwords = set(stopwords.words('english'))

max_len = 250


def pp(string):
    return ' '.join([stemmer.stem(word.decode('utf8')) for word in string.lower().split() if not word in stopwords])
    
def write_to_file(article, abstract, rel, writer):
    abstract = '<s> '+' '.join(abstract)+' </s>'
    #abstract = abstract.encode('utf8', 'ignore')
    #rel = rel.encode('utf8', 'ignore')
    #article = article.encode('utf8', 'ignore')
    tf_example = example_pb2.Example()
    tf_example.features.feature['abstract'].bytes_list.value.extend([bytes(abstract)])
    tf_example.features.feature['relevancy'].bytes_list.value.extend([bytes(rel)])
    tf_example.features.feature['article'].bytes_list.value.extend([bytes(article)])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))


def duck_iterator(i):
    duc_folder = 'duc0' + str(i) + 'tokenized/'
    for topic in os.listdir(duc_folder + 'testdata/docs/'):
        topic_folder = duc_folder + 'testdata/docs/' + topic
        if not os.path.isdir(topic_folder):
            continue
        query = ' '.join(open(duc_folder + 'queries/' + topic).readlines())
        model_files = glob(duc_folder + 'models/' + topic[:-1].upper() + '.*')

        topic_texts = [' '.join(open(topic_folder + '/' + file).readlines()).replace('\n', '') for file in
                       os.listdir(topic_folder)]

        abstracts = [' '.join(open(f).readlines()) for f in model_files]
        yield topic_texts, abstracts, query

def ones(sent, ref): return 1.

def count_score(sent, ref):
    ref = pp(ref).split()
    sent = ' '.join(pp(w) for w in sent.lower().split() if not w in stopwords)
    return sum([1. if w in ref else 0. for w in sent.split()])


def get_w2v_score_func(magic = 10):
    import gensim
    google = gensim.models.KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin', binary=True)
    def w2v_score(sent, ref):
        ref = ref.lower()
        sent = sent.lower()
        sent = [w for w in sent.split() if w in google]
        ref = [w for w in ref.split() if w in google]
        try:
            score = google.n_similarity(sent, ref)
        except:
            score = 0.
        return score * magic
    return w2v_score

def get_tfidf_score_func_glob(magic = 1):
    corpus = []
    for i in range(5, 8):
        for topic_texts, _, _ in duck_iterator(i):
            corpus += [pp(t) for t in topic_texts]

    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(corpus)

    def tfidf_score_func(sent, ref):
        #ref = [pp(s) for s in ref.split(' . ')]
        sent = pp(sent)
        v1 = vectorizer.transform([sent])
        #v2s = [vectorizer.transform([r]) for r in ref]
        #return max([cosine_similarity(v1, v2)[0][0] for v2 in v2s])
        v2 = vectorizer.transform([ref])
        return cosine_similarity(v1, v2)[0][0]

    return tfidf_score_func

tfidf_score = get_tfidf_score_func_glob()


def get_tfidf_score_func(magic = 10):
    corpus = []
    for i in range(5, 8):
        for topic_texts, _, _ in duck_iterator(i):
            corpus += [t.lower() for t in topic_texts]

    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(corpus)

    def tfidf_score_func(sent, ref):
        ref = ref.lower()
        sent = sent.lower()
        v1 = vectorizer.transform([sent])
        v2 = vectorizer.transform([ref])
        return cosine_similarity(v1, v2)[0][0]*magic
    return tfidf_score_func


def just_relevant(text, query):
    text = text.split(' . ')
    score_per_sent = [count_score(sent, query) for sent in text]
    sents_gold = list(zip(*sorted(zip(score_per_sent, text), reverse=True)))[1]
    sents_gold = sents_gold[:int(len(sents_gold)*ratio)]

    filtered_sents = []
    for s in text:
        if not s: continue
        if s in sents_gold: filtered_sents.append(s)
    return ' . '.join(filtered_sents)

class Summary:
    def __init__(self, texts, abstracts, query):
        #texts = sorted([(tfidf_score(query, text), text) for text in texts], reverse=True)
        #texts = sorted([(tfidf_score(text, ' '.join(abstracts)), text) for text in texts], reverse=True)

        #texts = [text[1] for text in texts]
        self.texts = texts
        self.abstracts = abstracts
        self.query = query
        self.summary = []
        self.words = set()
        self.length = 0

    def add_sum(self, summ):
        for sent in summ:
            self.summary.append(sent)

    def get(self):
        text = max([(len(t.split()), t) for t in  self.texts])[1]
        #text = texts[0]
        if ratio < 1: text = just_relevant(text, self.query)

        sents = text.split(' . ')
        score_per_sent = [(score_func(sent, self.query), sent) for sent in sents]
        #score_per_sent = [(count_score(sent, ' '.join(self.abstracts)), sent) for sent in sents]

        scores = []
        for score, sent in score_per_sent:
            scores += [score] * (len(sent.split()) + 1)
        scores = str(scores[:-1])
        return text, 'a', scores

def get_summaries(path):
    path = path+'decoded/'
    out = {}
    for file_name in os.listdir(path):
        index = int(file_name.split('_')[0])
        out[index] = open(path+file_name).readlines()
    return out

def rouge_eval(ref_dir, dec_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference_(\d+).txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    return r.convert_and_evaluate()

def evaluate(summaries):
    for path in ['eval/ref', 'eval/dec']:
        if os.path.exists(path): shutil.rmtree(path, True)
        os.mkdir(path)
    for i, summ in enumerate(summaries):
        for j,abs in enumerate(summ.abstracts):
            with open('eval/ref/'+str(i)+'_reference_'+str(j)+'.txt', 'w') as f:
                f.write(abs)
        with open('eval/dec/'+str(i)+'_decoded.txt', 'w') as f:
            f.write(' '.join(summ.summary))
    print rouge_eval('eval/ref/', 'eval/dec/') 


#count_score
score_func = ones#get_tfidf_score_func()

summaries = [Summary(texts, abstracts, query) for texts, abstracts, query in duck_iterator(duc_num)]

with open('test/temp_file', 'wb') as writer:
    for summ in summaries:
        article, abstract, scores = summ.get()
        write_to_file(article, abstracts, scores, writer)
call(['rm', '-r', generated_path])
call(cmd)
generated_summaries = get_summaries(generated_path)

for i in range(len(summaries)):
    summaries[i].add_sum(generated_summaries[i])

evaluate(summaries)
print duc_num
print score_func 


