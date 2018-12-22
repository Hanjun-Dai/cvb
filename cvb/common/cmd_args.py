import argparse
import os

cmd_opt = argparse.ArgumentParser(description='Argparser')
cmd_opt.add_argument('-dropbox', default='../../dropbox', help='path to dropbox')
cmd_opt.add_argument('-log_file', default='log.txt', help='log file')
cmd_opt.add_argument('-init_model_dump', default=None, help='model dump')
cmd_opt.add_argument('-saved_model', default=None, help='start from existing model')
cmd_opt.add_argument('-save_dir', default=None, help='root for output')
cmd_opt.add_argument('-inner_opt', default='SGD', help='SGD/RMSprop')
cmd_opt.add_argument('-ctx', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-arch', default='cnn', help='cnn/mlp')

cmd_opt.add_argument('-seed', default=1, type=int, help='random seed')
cmd_opt.add_argument('-epoch_save', default=1, type=int, help='save every k epochs')
cmd_opt.add_argument('-num_epochs', default=100, type=int, help='num epochs')
cmd_opt.add_argument('-batch_size', default=100, type=int, help='batchsize')
cmd_opt.add_argument('-test_batch_size', default=100, type=int, help='test batchsize')
cmd_opt.add_argument('-latent_dim', default=64, type=int, help='embedding size')
cmd_opt.add_argument('-hidden_dim', type=int, default=128, help='dimension of mlp layers')
cmd_opt.add_argument('-learning_rate', default=0.001, type=float, help='learning rate')
cmd_opt.add_argument('-mda_lr', default=0.01, type=float, help='mda learning rate')
cmd_opt.add_argument('-min_mda_lr', default=0.000001, type=float, help='min mda learning rate')
cmd_opt.add_argument('-mda_decay_factor', default=0.5, type=float, help='mirror descent decay factor')
cmd_opt.add_argument('-rbm_model', type=str, default='bernoulli', help='gaussian||bernoulli')
cmd_opt.add_argument('-pcd', type=int, default=0, help='use pcd or not // 0 || 1')
cmd_opt.add_argument('-img_size', type=int, default=32, help='image size')
cmd_opt.add_argument('-gnorm_lambda', type=float, default=0.0, help='lambda for gradient norm')
cmd_opt.add_argument('-gnorm_type', type=str, default='lp1', help='type for gradient norm (lp1 || norm2)')
cmd_opt.add_argument('-data_mean', default=0.0, type=float, help='data global mean')
cmd_opt.add_argument('-data_std', default=1.0, type=float, help='data std')

cmd_opt.add_argument('-unroll_steps', type=int, default=10, help='#unroll')
cmd_opt.add_argument('-binary', type=int, default=0, help='binary image')
cmd_opt.add_argument('-test_is', type=int, default=0, help='test using importance sampling')
cmd_opt.add_argument('-vis_num', type=int, default=0, help='only do vis')
cmd_opt.add_argument('-unroll_test', type=int, default=0, help='do unroll in test')
cmd_opt.add_argument('-n_dist', type=int, default=20, help='n_dist in discrete VAE')
cmd_opt.add_argument('-n_classes', type=int, default=10, help='n_classes in discrete VAE')


cmd_args, _ = cmd_opt.parse_known_args()

if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)
print(cmd_args)
