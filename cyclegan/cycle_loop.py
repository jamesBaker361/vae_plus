import tensorflow as tf
#from tensorflow_examples.models.pix2pix import pix2pix
from loss_functions import *
from cycle_data_helper import *
from cycle_model import *
from img_helpers import *
import time
import argparse
from datetime import datetime, timezone
import os
from random import randrange
import json

parser = argparse.ArgumentParser(description='get some args')
parser.add_argument("--epochs",type=int,help="training epochs", default=2)
parser.add_argument("--test",type=bool, default=False)
parser.add_argument("--batch_size", type=int,default=1) 
parser.add_argument("--save_img_parent",type=str,default="/home/jlb638/Desktop/vae_plus/gen_imgs/cyclegan/")
parser.add_argument("--name",type=str,default="cycle_{}".format(str(datetime.now(timezone.utc))))
parser.add_argument("--save_model_parent", type=str,default="../../../../../scratch/jlb638/yvae_models/cyclegan/")
parser.add_argument("--content_path",type=str,default="jlbaker361/flickr_humans_10k")
parser.add_argument("--style_path",type=str, default="jlbaker361/anime_faces_10k")
parser.add_argument("--load", type=bool, default=False, help="whether to load previous model if possible")
parser.add_argument("--save", type=bool, default=False, help='whether to save model')
parser.add_argument("--image_dim",type=int, default=128)
parser.add_argument("--interval",type=int,default=10,help='save model every interval # of epochs')
parser.add_argument("--threshold",type=int,default=50,help='epoch threshold for when to start saving')

args = parser.parse_args()

def objective(trial,args):
    save_folder=args.save_img_parent+args.name
    save_model_folder=args.save_model_parent+args.name
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_model_folder, exist_ok=True)

    print(args)
    OUTPUT_CHANNELS = 3
    start_epoch=0

    if args.load:
        generator_g=tf.saved_model.load(save_model_folder+"generator_g")
        generator_f=tf.saved_model.load(save_model_folder+"generator_f")
        discriminator_x=tf.saved_model.load(save_model_folder+"discriminator_x")
        discriminator_y=tf.saved_model.load(save_model_folder+"discriminator_y")
        with open(save_model_folder+"/meta_data.json","r") as src_file:
            start_epoch=json.load(src_file)["epoch"]

        print("successfully loaded from {} at epoch {}".format(save_model_folder, start_epoch))
    else:
        generator_g, generator_f, discriminator_x, discriminator_y= get_models(OUTPUT_CHANNELS, args.image_dim, 'instancenorm', False)

    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    @tf.function
    def train_step(real_x, real_y):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

            fake_y = generator_g(real_x, training=True)
            cycled_x = generator_f(fake_y, training=True)

            fake_x = generator_f(real_y, training=True)
            cycled_y = generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = generator_f(real_x, training=True)
            same_y = generator_g(real_y, training=True)


            disc_real_x = discriminator_x(real_x, training=True)
            disc_real_y = discriminator_y(real_y, training=True)


            disc_fake_x = discriminator_x(fake_x, training=True)
            disc_fake_y = discriminator_y(fake_y, training=True)


            # calculate the loss
            gen_g_loss = generator_loss(disc_fake_y)
            gen_f_loss = generator_loss(disc_fake_x)


            total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)


            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)


            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)


        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                            generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                            generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                                generator_g.trainable_variables))

        generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                                generator_f.trainable_variables))

        discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                    discriminator_x.trainable_variables))

        discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                    discriminator_y.trainable_variables))
    #end train step
    
    train_style, train_content = cycle_get_datasets_train(batch_size=args.batch_size,unit_test=args.test,content_path=args.content_path,style_path=args.style_path, image_dim=args.image_dim)
    print('train_style', train_style)
    print('train_content', train_style)
    #x = style
    #y = content
    train_content_sample=next(iter(train_content))
    content_list=[t for t in train_content][1:]
    train_style_sample=next(iter(train_style))
    style_list=[t for t in train_style][1:]
    print('train_style_sample', train_style_sample.shape)

    print("begin training")
    for epoch in range(start_epoch,args.epochs):
        start = time.time()
        n = 0
        for image_x, image_y in tf.data.Dataset.zip((train_style, train_content)):
            
            train_step(image_x, image_y)
            if n % 10 == 0:
                print ('.', end='')
            n += 1
        print ('\nTime taken for epoch {} is {} sec\n'.format(epoch + 1,time.time()-start))
        # Using a consistent image so that the progress of the model
        # is clearly visible.
        generate_images(generator_g, train_style_sample,save_folder+"/styletocontent{}.png".format(epoch))
        generate_images(generator_f, train_content_sample,save_folder+"/contenttostyle{}.png".format(epoch))
        random_content_sample=content_list[randrange(0,len(content_list))]
        random_style_sample=style_list[randrange(0,len(style_list))]
        generate_images(generator_g, random_style_sample,save_folder+"/random_styletocontent{}.png".format(epoch))
        generate_images(generator_f, random_content_sample,save_folder+"/random_contenttostyle{}.png".format(epoch))

        if epoch%args.interval==0 and epoch>args.threshold and args.save:
            
            tf.saved_model.save(generator_g,save_model_folder+"generator_g")
            tf.saved_model.save(generator_f,save_model_folder+"generator_f")
            tf.saved_model.save(discriminator_x,save_model_folder+"discriminator_x")
            tf.saved_model.save(discriminator_y,save_model_folder+"discriminator_y")
            meta_data = {"epoch":epoch}
            json_object = json.dumps(meta_data, indent=4)
 
            # Writing to sample.json
            with open(save_model_folder+"/meta_data.json", "w+") as outfile:
                outfile.write(json_object)
            print('saved at epoch {}'.format(epoch))

if __name__ == '__main__':
    print("begin!")
    print(args)
    objective(None, args)
    print('end!')