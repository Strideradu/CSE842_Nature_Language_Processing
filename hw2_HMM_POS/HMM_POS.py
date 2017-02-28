import HMM
import argparse
import sys
import os

class HMM_POS(object):
    def __init__(self, train_path, test_path, test_result):
        self.train_path = train_path
        self.test_path = test_path
        self.test_result = test_result
        self.model = HMM.HMM(45)

    def train(self, smooth):
        word, tag = self._parse_with_tag(self.train_path)
        self.model.estimate(word, tag, smooth)

    def test(self, output_num = 0):
        test_words = self._parse_wo_tag(self.test_path)
        _, test_tags = self._parse_with_tag(self.test_result)
        tag_result, log_result = self.model.test(test_words, test_tags, output_num)

        for i in range(len(tag_result)):
            output_tbl = []
            for j in range(len(tag_result[i])):
                output_tbl.append(test_words[i][j])
                output_tbl.append(tag_result[i][j])

            print str(i+1)
            print " ".join(output_tbl) + " logodds: " + str(log_result[i])

    def _parse_with_tag(self, path):
        """

        :param path: path for file
        :return: two list of word and tag sequences
        """
        with open(path) as f:
            lines = f.readlines()
            parsed_data = []
            parsed_tag = []

            for line in lines:
                parsed_word = []
                parsed_line_tag = []
                line_sp = line.strip().split()
                for i in range(len(line_sp)/2):
                    parsed_word.append(line_sp[2*i])
                    parsed_line_tag.append(line_sp[2*i + 1])

                parsed_data.append(parsed_word)
                parsed_tag.append(parsed_line_tag)

        return parsed_data, parsed_tag

    def _parse_wo_tag(self, path):
        with open(path) as f:
            lines = f.readlines()
            parsed_data = []

            for line in lines:
                parsed_word = []
                parsed_line_tag = []
                line_sp = line.strip().split()
                for i in range(len(line_sp)):
                    parsed_word.append(line_sp[i])

                parsed_data.append(parsed_word)

        return parsed_data

    def argparse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("training_path", help="the training file path for HMM POS",
                            type=str, nargs=1)
        parser.add_argument("test_path", help="the test file path for HMM POS, no tags",
                            type=str, nargs=1)
        parser.add_argument("test_truth_path", help="the test thruth file path for HMM POS",
                            type=str, nargs=1)

        parser.add_argument("-lambda", dest="smooth", help="smooth parameter", type=float)
        parser.add_argument("-k", dest="k", help="hwo many result want to output, other wise obly report accuracy", type=int)

        try:
            args = parser.parse_args()

        except:
            parser.print_help()
            sys.exit(1)

        if os.path.exists(args.training_path[0]):
            print "training file is " + str(os.path.basename(args.training_path[0]))
            self.train_path = args.training_path[0]

        if os.path.exists(args.test_path[0]):
            print "test file is " + str(os.path.basename(args.test_path[0]))
            self.test_path = args.test_path[0]

        if os.path.exists(args.test_truth_path[0]):
            print "truth file is " + str(os.path.basename(args.test_truth_path[0]))
            self.test_result = args.test_truth_path[0]

        if args.smooth:
            self.train(args.smooth)
        else:
            self.train()

        if args.k:
            self.test(args.k)
        else:
            self.test()


if __name__ == '__main__':
    model = HMM_POS(None, None, None)
    model.argparse()