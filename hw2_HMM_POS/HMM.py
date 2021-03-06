import numpy as np
np.set_printoptions(threshold='nan')

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

        self.tag_dict = None
        self.reverse_tag_dict = None
        self.word_index = None

    def estimate(self, word_list, tag_list, smooth = 1.0):
        """

        :param pos_train: list of senstences with tag
        :param smooth: smooth parameter
        :return:
        """

        states_count, word_dict = self._record_count(word_list, tag_list)

        state_sum = np.sum(states_count, axis=1)
        total = np.sum(states_count)

        self._fill_emission(word_dict, state_sum)

        self._fill_transimission(states_count, state_sum, total, smooth)
        print "Parameter estimation finished"


    def _record_count(self, parsed_word, parsed_tag):
        """
        recording the count of
        :param pos_train:
        :return:
        """
        # given a tag, know the i
        self.tag_dict = {"START": 0, "END": 1}
        # given i, know the tag
        self.reverse_tag_dict = {0:"SATRT", 1:"END"}
        states_count = np.zeros((self.n,self.n),dtype=np.int)
        word_dict = {}


        for x in range(len(parsed_word)):
            line_word = parsed_word[x]
            line_tag = parsed_tag[x]
            # print line_tag
            prev_state_i = 0
            for i in range(len(line_word)):

                state_i = self.tag_dict.get(line_tag[i], None)

                # no state found in dict, initialize this state
                if state_i is None:
                    state_i = len(self.tag_dict)
                    self.tag_dict[line_tag[i]] = state_i
                    self.reverse_tag_dict[state_i] = line_tag[i]

                states_count[prev_state_i][state_i] += 1

                word_tbl = word_dict.get(line_word[i], None)
                if word_tbl is None:
                    word_dict[line_word[i]] = np.zeros(self.n, dtype=np.int)
                word_dict[line_word[i]][state_i] += 1

                prev_state_i = state_i

            state_i = 1
            states_count[prev_state_i][state_i] += 1

        # print states_count[0]

        return states_count, word_dict

    def _fill_emission(self, word_dict, state_sum):
        """
        fill the emission array
        :param word_dict:
        :param state_sum:
        :return:
        """
        # calculte the probabilities of word given state
        self.word_index = {}
        index_word = {}
        word_freq_tbl = []
        unka_tbl = np.zeros(self.n, dtype=precision)

        for word in word_dict:
            count = np.sum(word_dict[word])
            if count >= 5:
                freq = np.zeros(self.n, dtype=precision)
                for i in range(self.n):
                    if i > 1:
                        freq[i] = np.log(float(word_dict[word][i]) / state_sum[i])

                    else:
                        freq[i] = -np.inf

                word_freq_tbl.append(freq)
                self.word_index[word] = len(word_freq_tbl) - 1
                index_word[self.word_index[word]] = word
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
        self.word_index["UNKA"] = len(word_freq_tbl) - 1
        index_word[self.word_index["UNKA"]] = "UNKA"

        self.emission = np.array(word_freq_tbl)

    def _fill_transimission(self, states_count, state_sum, total, smooth):
        """
        fill the transimisssion array
        :param states_count:
        :param state_sum:
        :param total:
        :param smooth:
        :return:
        """
        for i in range(self.n):
            for j in range(self.n):
                if state_sum[i] > 0:
                    self.transition[i][j] = np.log(smooth * float(states_count[i][j])/state_sum[i] + (1.0-smooth)*float(state_sum[j])/total)
                else:
                    self.transition[i][j] = np.log(float(state_sum[j])/total)


        #print self.transition

    def test(self, words, labels, output_num):
        total_word = 0
        right_word = 0
        length = len(words)
        output = []
        log_out = []
        for i in range(length):

            log_odds, answer = self._viterbi(words[i])
            ground_truth = labels[i]
            for j, word in enumerate(answer):
                if ground_truth[j] == word:
                    right_word += 1
            total_word += len(answer)

            if i < output_num:
                output.append(answer)
                log_out.append(log_odds)

            if i % int(0.1*length) == 0:
                print format((float(i)/length), '.2%') + "test finished"

        print "accuracy is " + str(float(right_word)/total_word)
        return output, log_out

    def infer(self, word_list):
        for parsed_line in  word_list:
            # print line
            self._viterbi(parsed_line)

    def _viterbi(self, word_seq):
        """

        delta is matrix of score of viterbi

        psi is the last best state

        :param word_seq: sequence of the word in the sentence
        :return:
        """
        # print word_seq
        delta = np.full((len(word_seq), self.n), -np.inf, dtype = precision)
        psi = np.zeros((len(word_seq), self.n), dtype=np.int)

        # the oth step is always 0
        for x in xrange(self.n):
            word_i = self.word_index[word_seq[0]]
            delta[0][x] = self.transition[0][x] + self.emission[word_i][x]
            psi[0][x] = 0

        for t in xrange(1, len(word_seq)):
            word_i = self.word_index[word_seq[t]]
            for j in xrange(self.n):
                for i in xrange(self.n):
                    if (delta[t][j] < delta[t - 1][i] + self.transition[i][j]):
                        delta[t][j] = delta[t - 1][i] + self.transition[i][j]
                        psi[t][j] = i
                delta[t][j] += self.emission[word_i][j]

        return self._backtrack(psi, delta)

    def _backtrack(self, psi, delta):
        state_result = []
        psi_max = -np.inf
        i_max = 0
        for i in xrange(self.n):
            if (psi_max < (delta[- 1][i] + self.transition[i][1])):
                psi_max = delta[ - 1][i] + self.transition[i][1]
                i_max = i
        state_result.append(self.reverse_tag_dict[i_max])

        length = delta.shape[0]
        for i in xrange(1, length):
            i_max = psi[length - i][i_max]
            state_result.append(self.reverse_tag_dict[i_max])

        # print psi_max
        return psi_max, state_result[::-1]


if __name__ == '__main__':
    hmm = HMM()