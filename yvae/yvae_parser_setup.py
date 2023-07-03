from datetime import datetime, timezone

def add_arguments(parser):
    #classifier
    parser.add_argument("--epochs",type=int,help="training epochs", default=2)
    parser.add_argument("--test",type=bool, default=False,help='whether this is a test or not')
    parser.add_argument("--batch_size", type=int,default=1) 
    parser.add_argument("--save_img_parent",type=str,default="/home/jlb638/Desktop/vae_plus/gen_imgs/yvae/",help='root folder of where to save model')
    parser.add_argument("--name",type=str,default="yvae_{}".format(str(datetime.now(timezone.utc))))
    parser.add_argument("--save_model_parent", type=str,default="../../../../../scratch/jlb638/yvae_models/yvae/")
    parser.add_argument("--dataset_names",nargs="+",default=["jlbaker361/flickr_humans_10k", "jlbaker361/anime_faces_10k" ])
    parser.add_argument("--load", type=bool, default=False, help="whether to load previous model if possible")
    parser.add_argument("--save", type=bool, default=False, help='whether to save model')
    parser.add_argument("--image_dim",type=int, default=128)
    parser.add_argument("--interval",type=int,default=10,help='save model every interval # of epochs')
    parser.add_argument("--threshold",type=int,default=50,help='epoch threshold for when to start saving')
    parser.add_argument("--latent_dim",type=int, default=32,help='latent dim for encoding')
    parser.add_argument("--log_dir_parent",type=str,default="logs/")
    parser.add_argument("--resnet",type=bool, default=False)
    parser.add_argument("--unfreezing_epoch",type=int,default=10,help='epoch at which to unfreeze pretrained external model')
    parser.add_argument("--data_augmentation", type=bool, default=False,help="whether to do data augmentation when training")
    parser.add_argument("--node",type=str,default='unknown',help='which node this is running on for debugging on HPC')
    parser.add_argument("--initializer",type=str,default="glorot_normal",help="which initializer to use")
    parser.add_argument("--init_lr",type=float,default=0.0001,help='lr for initial adam optimizer (frozen if fine tning)')

    #classifier
    parser.add_argument("--external_name",type=str,default="",help='[classifier] if set, whether to use external pretrained model, options in yvae_model')

    #unit loop
    parser.add_argument("--kl_loss_scale",type=float,default=1.0,help='[unit vae] scale of kl_loss for optimizing')
    parser.add_argument("--disable_strategy",help="whether to use mirrored_strategy in trainer",type=bool,default=False)
    parser.add_argument("--fine_tuning",type=bool, default=False,help="[unit vae + creativity vae] whether to use fine tuning training (freezing encoder initially)")
    parser.add_argument("--use_residual",type=bool,default=False,help='[unit vae + creativity vae] whether to use residual blocks in autoencoder')
    parser.add_argument("--use_bn", type=bool, default=False, help='[unit vae + creativity vae] whether to use batch normalization in encoder/decoder')
    parser.add_argument("--use_gn",type=bool,default=False, help='[unit vae + creativity vae] whether to use group normalization')
    parser.add_argument("--pretrained_creativity_path",type=str,default='',help='[unit vae] path to load pretrained creativity encoder')
    parser.add_argument('--fid_interval', type=int, default=-1, help="interval in which to calculate fid score (default = -1 = never)")
    parser.add_argument('--vgg_interval', type=int, default=-1, help="interval in which to calculate vgg content loss (default = -1 = never)")

    #creatiivity
    parser.add_argument("--use_strategy",help="whether to use mirrored_strategy in trainer",type=bool,default=False)
    parser.add_argument("--creativity_lambda",type=float,default=0.5, help='[creativity vae] coefficient on creativity loss' )
    parser.add_argument("--pretrained_classifier_path",type=str,help="[creativity vae] path to load pretrained_classifier")

    #normal yvae
    parser.add_argument("--use_unit",help='whether to use unsupervised image to image',type=bool,default=False)
    parser.add_argument("--reconstruction_loss_function_name",type=str,default='mse',help='[yvae] reconstruction loss function between input image and reconstructed image, options in yvae_trainer')
