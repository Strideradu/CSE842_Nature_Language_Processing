import numpy as np

precision = np.double

class HMM(object):
    def __init__(self, state_num = 45, observe_num = None):
        # for start and end state
        self.n = state_num + 2
        self.transition = np.zeros((self.n,self.n),dtype=precision)
        if observe_num:
            self.m = observe_num
            self.emission = np.zeros( (self.n,self.m) ,dtype=precision)

        else:
            self.m = None
            self.emission = None

    def estimate(self, pos_train, smooth = 1):
        # given a tag, know the i
        tag_dict = {"START": 0, "END": 1}
        # given i, know the tag
        reverse_tag_dict = {0:"SATRT", 1:"END"}
        states_count = np.zeros((self.n,self.n),dtype=np.int)
        word_dict = {}


        for line in pos_train:
            line_tag = line.split()
            # print line_tag
            prev_state_i = 0
            for i in range(len(line_tag)/2):

                state_i = tag_dict.get(line_tag[2*i + 1], None)

                # no state found in dict, initialize this state
                if state_i is None:
                    state_i = len(tag_dict)
                    tag_dict[line_tag[2*i + 1]] = state_i
                    reverse_tag_dict[state_i] = line_tag[2*i + 1]

                states_count[prev_state_i][state_i] += 1

                word_tbl = word_dict.get(line_tag[2 * i], None)
                if word_tbl is None:
                    word_dict[line_tag[2 * i]] = np.zeros(self.n, dtype=np.int)
                word_dict[line_tag[2 * i]][state_i] += 1

                prev_state_i = state_i

            state_i = 1
            states_count[prev_state_i][state_i] += 1

        # calculte the probabilities of word given state
        word_index = {}
        index_word = {}
        word_freq_tbl = []
        unka_tbl = np.zeros(self.n, dtype=precision)
        state_sum = np.sum(states_count, axis= 0)
        total = np.sum(states_count)

        for word in word_dict:
            count = np.sum(word_dict[word])
            if count >= 5:
                freq = np.zeros(self.n, dtype=precision)
                for i in range(self.n):
                    if i >1:
                        freq[i] = np.log(float(word_dict[word][i])/state_sum[i])

                    else:
                        freq[i] = -np.inf

                word_freq_tbl.append(freq)
                word_index[word] = len(word_freq_tbl) - 1
                index_word[word_index[word]] = word
            else:
                for i in range(self.n):
                    unka_tbl[i] += word_dict[word][i]

        freq = np.zeros(self.n, dtype=precision)
        for i in range(self.n):
            if i > 1:
                freq[i] = np.log(float(unka_tbl[i]) / state_sum[i])

            else:
                freq[i] = -np.inf

        word_freq_tbl.append(freq)
        word_index["UNKA"] = len(word_freq_tbl) - 1
        index_word[word_index["UNKA"]] = "UNKA"

        self.emission = np.array(word_freq_tbl)

        for i in range(self.n):
            for j in range(self.n):
                if state_sum[i] > 0:
                    self.transition[i][j] = np.log(smooth * float(states_count[i][j])/state_sum[i] + (1-smooth)*float(state_sum[j]/total) )

                else:
                    self.transition[i][j] = np.log(float(state_sum[j]/total))

        print state_sum



if __name__ == '__main__':
    hmm = HMM()
    with open("wsj1-18.training") as f:
        lines = f.readlines()
        hmm.estimate(lines)
