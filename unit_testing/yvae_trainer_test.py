import sys
sys.path.append('yvae')


from yvae_trainer import *
LOG_DIR='logs/yvae_trainer_unit_testing/'

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
    start_name='encoder_conv_4',
    epochs=1):
    inputs = Input(shape=input_shape, name='encoder_input')
    pretrained_encoder=get_encoder(inputs, latent_dim)
    unit_list=get_unit_list(input_shape,latent_dim,n_classes,pretrained_encoder, start_name)
    dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n_classes) for name in ["a","b","c"]}
    test_dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n_classes) for name in ["a","b","c"]}
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    kl_loss_scale=1.0
    callbacks=[]
    start_epoch=0
    trainer=VAE_Trainer(unit_list, epochs,dataset_dict,test_dataset_dict,optimizer,log_dir=LOG_DIR,
                        mirrored_strategy=None,kl_loss_scale=kl_loss_scale,callbacks=callbacks,start_epoch=start_epoch)
    trainer.train_loop()

def VAE_Trainer_Unit_generate_imgs_test(input_shape=(32,32,3),
    latent_dim=10,
    n_classes=3,
    start_name='encoder_conv_4',
    epochs=1):
    inputs = Input(shape=input_shape, name='encoder_input')
    pretrained_encoder=get_encoder(inputs, latent_dim)
    unit_list=get_unit_list(input_shape,latent_dim,n_classes,pretrained_encoder, start_name)
    dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n_classes) for name in ["a","b","c"]}
    test_dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n_classes) for name in ["a","b","c"]}
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    kl_loss_scale=1.0
    callbacks=[]
    start_epoch=0
    trainer=VAE_Trainer(unit_list, epochs,dataset_dict,test_dataset_dict,optimizer,log_dir=LOG_DIR,mirrored_strategy=None,kl_loss_scale=kl_loss_scale,callbacks=callbacks,start_epoch=start_epoch)
    trainer.generate_images(2)

def VAE_Creativity_Trainer_test(input_shape=(32,32,3),
                                latent_dim=10,
    n_classes=3,
    start_name='encoder_conv_4',
    epochs=1):
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    dataset_dict={}
    test_dataset_dict={}
    dataset_list=yvae_creativity_get_dataset_train(image_dim=input_shape[1])
    pretrained_classifier=None
    trainer=VAE_Creativity_Trainer(vae_list,epochs,dataset_dict,test_dataset_dict,optimizer,dataset_list,log_dir='',mirrored_strategy=None,kl_loss_scale=1.0,callbacks=[],start_epoch=0, global_batch_size=4,pretrained_classifier=None, creativity_lambda=1.0)
    pass


if __name__ =='__main__':
    for img_dim in [32,256]:
        shape=(img_dim, img_dim, 3)
        print(shape)
        YVAE_Trainer_test(input_shape=shape)
        YVAE_Trainer_generate_images_test(input_shape=shape)
        VAE_Trainer_Unit_test(input_shape=shape)
        VAE_Trainer_Unit_generate_imgs_test(input_shape=shape)
    for loss in ['mse', 'binary_crossentropy', 'log_cosh','huber']:
        YVAE_Trainer_test_reconstruction(reconstruction_loss_function_name=loss)
    print("all done! :)")