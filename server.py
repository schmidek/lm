import tornado.ioloop
import tornado.web
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf

from data_utils import Vocabulary, DatasetSentence
from language_model import LM
from run_utils import run_train, run_eval

import sys
import time
import os

import numpy as np
from tensorflow.python.client import timeline

from common import CheckpointLoader
import config as cfg

num_eval_steps = 1
MAX_WORKERS = os.cpu_count() / 4
print("Using %d workers" % MAX_WORKERS)

class MainHandler(tornado.web.RequestHandler):

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    @run_on_executor
    def score(self, text):
      results = []
      for line in text:
        print(line)
        dataset = DatasetSentence(vocab, line)
        data_iterator = dataset.iterate_once(hps.batch_size * hps.num_gpus, hps.num_steps)
        loss_nom = 0.0
        loss_den = 0.0
        for i, (x, y, w) in enumerate(data_iterator):
            if i >= num_eval_steps:
                break
            time1i = time.time()
            loss = sess.run(model.loss, {model.x: x, model.y: y, model.w: w})
            print(loss)
            time2i = time.time()
            print('Loss function took %0.3f ms' % ((time2i-time1i)*1000.0))
            loss_nom += loss
            loss_den += w.mean()
            loss = loss_nom / loss_den
            sys.stdout.write("%d: %.3f (%.3f) ... " % (i, loss, np.exp(loss)))
            sys.stdout.flush()
        sys.stdout.write("\n")

        log_perplexity = loss_nom / loss_den
        print("Results: log_perplexity = %.3f perplexity = %.3f nom = %.3f den = %.3f" % (
            log_perplexity, np.exp(log_perplexity), loss_nom, loss_den))
        results.append(str(np.exp(log_perplexity)))
      return results

    @tornado.gen.coroutine
    def get(self):
      time1 = time.time()
      text = self.get_arguments("text")
      results = yield self.score(text)
      self.write("\n".join(results))
      time2 = time.time()
      print('Request function took %0.3f ms' % ((time2-time1)*1000.0))

def make_app():
    return tornado.web.Application([
        (r"/lm", MainHandler),
    ])

def setupLM():
    global sess, ckpt_loader, model
    with tf.variable_scope("model"):
        hps.num_sampled = cfg.num_sampled  #0 = Always using full softmax at evaluation.
        hps.keep_prob = 1.0
        model = LM(hps, "eval", "/cpu:0")

    if hps.average_params:
        print("Averaging parameters for evaluation.")
        saver = tf.train.Saver(model.avg_dict)
    else:
        saver = tf.train.Saver()

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=20,
                            inter_op_parallelism_threads=1)
    sess = tf.Session(config=config)
    logdir = "log"
    ckpt_loader = CheckpointLoader(saver, model.global_step, logdir + "/train")
    with sess.as_default():
      ckpt_loader.load_checkpoint()
      tf.initialize_local_variables().run()

if __name__ == "__main__":

    hps = LM.get_default_hparams().parse(cfg.hpconfig)
    hps.num_gpus = 1

    sys.stdout.write("Reading Vocab\n")

    vocab = Vocabulary.from_file("1b_word_vocab.txt")

    sys.stdout.write("Setting up model\n")
    setupLM()

    port = 8888
    sys.stdout.write("Listening on %d\n" % port)

    app = make_app()
    app.listen(port)
    tornado.ioloop.IOLoop.current().start()