#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from src.utils.UtilsOfTree import core_nlp_tree_to_json
from stanza.server import CoreNLPClient
import os
import json
from nltk import sent_tokenize
from tqdm import tqdm

def articles_to_tree(client, base_path, out_base_path):
    files = os.listdir(base_path)
    for file in tqdm(files):

        file_path = os.path.join(base_path, file)
        out_path = os.path.join(out_base_path, file)
        article_to_tree(client, file_path, out_path)


def article_to_tree(client, file_path, out_path):
    
    error = False
    error_sent = None
    with open(file_path) as file, open(out_path, 'w') as out_file:

        lines = [line.strip() for line in file.readlines()]
        for line in lines:
            for sent in sent_tokenize(line):
                try:
                    annotation = client.annotate(sent)
                    for fragment in annotation.sentence:
                        tree = core_nlp_tree_to_json(fragment.parseTree)
                        out_file.write(json.dumps(tree)+'\n')
                except:
                    continue


# In[ ]:


if __name__ == "__main__":
    with CoreNLPClient(
        annotators=['tokenize','ssplit', 'parse'],
        timeout=30000,
        memory='16G') as client:

        fake_path = "data/FakeNewsNet/politi/fake/"
        fake_out = "data/FakeNewsNet/politiTree/fake/"
        real_path = "data/FakeNewsNet/politi/real/"
        real_out = "data/FakeNewsNet/politiTree/real/"

        sent = articles_to_tree(client, fake_path, fake_out)
        sent2  = articles_to_tree(client, real_path, real_out)


# In[ ]:




