import sys
sys.path.append('yvae')
import os


from yvae_trainer import *
from yvae_data_helper import *
LOG_DIR='logs/yvae_trainer_unit_testing/'
os.makedirs(LOG_DIR, exist_ok=True)

def YVAE_Trainer_test(
    input_shape=(32,32,3),
    latent_dim=10,
    n=3,
    epochs=1):
    reconstruction_loss_function_name='mse'
    y_vae_list=get_y_vae_list(latent_dim, input_shape,n)
    dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n) for name in ["a","b","c"]}
    test_dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n) for name in ["a","b","c"]}
    optimizer=keras.optimizers.Adam(learning_rate=0.0001)
    trainer=YVAE_Trainer(y_vae_list, epochs,dataset_dict,test_dataset_dict,optimizer,reconstruction_loss_function_name,log_dir=LOG_DIR)
    trainer.train_loop()

def YVAE_Trainer_test_reconstruction(
        input_shape=(64,64,3),
    latent_dim=10,
    n=3,
    epochs=3,
    reconstruction_loss_function_name='mse'):
    y_vae_list=get_y_vae_list(latent_dim, input_shape,n)
    dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n) for name in ["a","b","c"]}
    test_dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n) for name in ["a","b","c"]}
    optimizer=keras.optimizers.Adam(learning_rate=0.0001)
    trainer=YVAE_Trainer(y_vae_list, epochs,dataset_dict,test_dataset_dict,optimizer, reconstruction_loss_function_name, log_dir=LOG_DIR)
    trainer.train_loop()

def YVAE_Trainer_generate_images_test(
    input_shape=(32,32,3),
    latent_dim=10,
    n=3,
    epochs=1):
    reconstruction_loss_function_name='mse'
    y_vae_list=get_y_vae_list(latent_dim, input_shape,n)
    dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n) for name in ["a","b","c"]}
    test_dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n) for name in ["a","b","c"]}
    optimizer=keras.optimizers.Adam(learning_rate=0.0001)
    trainer=YVAE_Trainer(y_vae_list, epochs,dataset_dict,test_dataset_dict,optimizer,reconstruction_loss_function_name, log_dir=LOG_DIR)
    trainer.generate_images(2)

def VAE_Trainer_Unit_test(input_shape=(32,32,3),
    latent_dim=10,
    n_classes=3,
    mid_name='encoder_conv_4',
    epochs=1):
    pretrained_encoder=get_encoder(input_shape, latent_dim)
    unit_list=get_unit_list(input_shape,latent_dim,n_classes,pretrained_encoder, mid_name)
    dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n_classes) for name in ["a","b","c"]}
    test_dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n_classes) for name in ["a","b","c"]}
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    kl_loss_scale=1.0
    callbacks=[]
    start_epoch=0
    trainer=VAE_Unit_Trainer(unit_list, epochs,dataset_dict,test_dataset_dict,optimizer,log_dir=LOG_DIR,
                        mirrored_strategy=None,kl_loss_scale=kl_loss_scale,callbacks=callbacks,start_epoch=start_epoch)
    print('69 nice')
    trainer.train_loop()

def VAE_Trainer_Unit_generate_imgs_test(input_shape=(32,32,3),
    latent_dim=10,
    n_classes=3,
    mid_name='encoder_conv_4',
    epochs=1):
    pretrained_encoder=get_encoder(input_shape, latent_dim)
    unit_list=get_unit_list(input_shape,latent_dim,n_classes,pretrained_encoder, mid_name)
    dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n_classes) for name in ["a","b","c"]}
    test_dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n_classes) for name in ["a","b","c"]}
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    kl_loss_scale=1.0
    callbacks=[]
    start_epoch=0
    trainer=VAE_Unit_Trainer(unit_list, epochs,dataset_dict,test_dataset_dict,optimizer,log_dir=LOG_DIR,mirrored_strategy=None,kl_loss_scale=kl_loss_scale,callbacks=callbacks,start_epoch=start_epoch)
    trainer.generate_images(2)

def VAE_Trainer_Unit_style_transfer_test(input_shape=(32,32,3),
    latent_dim=10,
    n_classes=3,
    mid_name='encoder_conv_2',
    epochs=1):
    pretrained_encoder=get_encoder(input_shape, latent_dim)
    unit_list=get_unit_list(input_shape,latent_dim,n_classes,pretrained_encoder, mid_name)
    dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n_classes) for name in ["a","b","c"]}
    test_dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n_classes) for name in ["a","b","c"]}
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    kl_loss_scale=1.0
    callbacks=[]
    start_epoch=0
    trainer=VAE_Unit_Trainer(unit_list, epochs,dataset_dict,test_dataset_dict,optimizer,log_dir=LOG_DIR,mirrored_strategy=None,kl_loss_scale=kl_loss_scale,callbacks=callbacks,start_epoch=start_epoch)
    imgs=tf.random.normal((4,*input_shape))
    transferred= trainer.style_transfer(imgs,1)
    for t in transferred:
        print(tf.shape(t))

def VAE_Creativity_Trainer_test(input_shape=(32,32,3),
                                latent_dim=10,
    n_classes=3,
    epochs=1):
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    dataset_dict={}
    test_dataset_dict={}
    batch_size=4
    dataset_list=yvae_creativity_get_dataset_train(image_dim=input_shape[1],batch_size=batch_size)
    pretrained_classifier=get_classifier_model(latent_dim,input_shape,n_classes)
    vae_list=get_y_vae_list(latent_dim, input_shape, 1)
    log_dir='./logs/unit_testing/creativity_loop/'
    trainer=VAE_Creativity_Trainer(vae_list,epochs,dataset_dict,test_dataset_dict,optimizer,dataset_list,log_dir=log_dir,mirrored_strategy=None,kl_loss_scale=1.0,callbacks=[],start_epoch=0, global_batch_size=batch_size,pretrained_classifier=pretrained_classifier, creativity_lambda=1.0,n_classes=n_classes)
    trainer.train_loop()
    trainer.generate_images(2)


if __name__ =='__main__':
    for img_dim in [32,256]:
        shape=(img_dim, img_dim, 3)
        print(shape)
        YVAE_Trainer_test(input_shape=shape)
        print('129')
        YVAE_Trainer_generate_images_test(input_shape=shape)
        print('131')
        VAE_Trainer_Unit_test(input_shape=shape)
        print('133')
        VAE_Trainer_Unit_generate_imgs_test(input_shape=shape)
        print('135')
        VAE_Creativity_Trainer_test(input_shape=shape)
        print('137')
        VAE_Trainer_Unit_style_transfer_test(input_shape=shape)
        print("kjadsfjfka")
    for loss in ['mse', 'binary_crossentropy', 'log_cosh','huber']:
        YVAE_Trainer_test_reconstruction(reconstruction_loss_function_name=loss)
    print("all done! :)")