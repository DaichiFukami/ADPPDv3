import argparse

import chainer
from chainer import training
from chainer.training import extensions
from models import Discriminator
from models import Encoder
from models import Decoder
from updater import FacadeUpdater

import dataset

#from visualizer import out_image

trainQuePath ='trainQue'
trainAnsPath = 'trainAns'
testQuePath ='testQue'
testAnsPath = 'testAns'
charSize = 16*8

def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='バッチサイズ')#50
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='エポック数')#20
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPUの有無')
    parser.add_argument('--dataset', '-i', default='./facade/base',
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='result',
                        help='リザルトファイルのフォルダ')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--frequency', '-f', type=int, default=5,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    enc = Encoder(in_ch=3)
    dec = Decoder(out_ch=3)
    dis = Discriminator(in_ch=3, out_ch=3)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer
    opt_enc = make_optimizer(enc)
    opt_dec = make_optimizer(dec)
    opt_dis = make_optimizer(dis)

    train_data = dataset.DatasetPourDot(trainQuePath,trainAnsPath)
    test_data = dataset.DatasetPourDot(testQuePath,testAnsPath)
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize)

    # Set up a trainer
    updater = FacadeUpdater(
        models=(enc, dec, dis),
        iterator={
            'main': train_iter,
            'test': test_iter},
        optimizer={
            'enc': opt_enc, 'dec': opt_dec,
            'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    #途中経過の表示用の記述
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(
        filename='snapshot_iter_{.updater.iteration}.npz'),
                   trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport())
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['dis/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(['dis/accuracy'],'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'enc/loss', 'dec/loss', 'dis/loss',
    ]))
    trainer.extend(extensions.ProgressBar())
    """
    trainer.extend(
        out_image(
            updater, enc, dec,
            5, 5, args.seed, args.out),
        trigger=snapshot_interval)
    """
    #中断データの有無、あれば続きから
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    #実験開始、trainerにお任せ
    trainer.run()
    #CPUで計算できるようにしておく
    enc.to_cpu()
    dec.to_cpu()
    dis.to_cpu()
    #npz形式で書き出し
    chainer.serializers.save_npz(args.out+'/mymodel_enc.npz', enc)
    chainer.serializers.save_npz(args.out+'/mymodel_dec.npz', dec)
    chainer.serializers.save_npz(args.out+'/mymodel_dis.npz', dis)
if __name__ == '__main__':
    main()
