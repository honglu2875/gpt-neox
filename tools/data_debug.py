from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from absl import app, flags


FLAGS = flags.FLAGS
flags.DEFINE_string('path', '/mnt/nvme/home/honglu/diff_models_megatron/data/1-deduped_train/train_text_document', 'data path(prefix)', short_name='p')


def main(argv):
    d = make_indexed_dataset(FLAGS.path, 'mmap', True)
    print(d[0])
    print(d[1])
    for i in range(10):
        print(d[i].shape)


if __name__ == '__main__':
    app.run(main)
