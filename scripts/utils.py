import tensorflow as tf

def get_ckpt(model_dir, epoch):
    if epoch is not None and epoch > 0:
        ckpts = tf.train.get_checkpoint_state(model_dir).all_model_checkpoint_paths
        ckpt = [c for c in ckpts if c.endswith('checkpoint-{}'.format(epoch))]
        assert len(ckpt) == 1
        cur_checkpoint = ckpt[0]
    else:
        cur_checkpoint = tf.train.latest_checkpoint(model_dir)
    return cur_checkpoint

