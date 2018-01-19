import tensorflow as tf
import numpy as np
# there is no rule of thumb for the amount of hidden nodes you should use, it is something you have to figure out case-specifically by trial and error

f1 = '/Users/weiliang/G_Drive/dl_study_notes/trytry/rnn/belling_the_cat.txt'
f2 = '/Users/weiliang/G_Drive/dl_study_notes/nlp_experiments/data/i_robot_asimov.txt'


def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    print(content)
    content = [x.strip() for x in content]
    content = [' '.join(map(str, content))]

    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)

    content = np.reshape(content, [-1, ])
    print(content)
    return content


t1 = read_data(f1)
t2 = read_data(f2)
