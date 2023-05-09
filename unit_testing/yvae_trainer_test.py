import sys
sys.path.append('yvae')


from yvae_trainer import *

def YVAE_Trainer_test(
    input_shape=(32,32,3),
    latent_dim=10,
    n=3,
    epochs=1):
    reconstruction_loss_function_name='mse'
    y_vae_list=get_y_vae_list(latent_dim, input_shape,n)
    dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n) for name in ["a","b","c"]}
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0001)
    trainer=YVAE_Trainer(y_vae_list, epochs,dataset_dict,optimizer, reconstruction_loss_function_name)
    trainer.train_loop()

def YVAE_Trainer_test_reconstruction(
        input_shape=(64,64,3),
    latent_dim=10,
    n=3,
    epochs=3,
    reconstruction_loss_function_name='mse'):
    y_vae_list=get_y_vae_list(latent_dim, input_shape,n)
    dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n) for name in ["a","b","c"]}
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0001)
    trainer=YVAE_Trainer(y_vae_list, epochs,dataset_dict,optimizer, reconstruction_loss_function_name)
    trainer.train_loop()

def YVAE_Trainer_generate_images_test(
    input_shape=(32,32,3),
    latent_dim=10,
    n=3,
    epochs=1):
    reconstruction_loss_function_name='mse'
    y_vae_list=get_y_vae_list(latent_dim, input_shape,n)
    dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n) for name in ["a","b","c"]}
    optimizer=keras.optimizers.Adam(learning_rate=0.0001)
    trainer=YVAE_Trainer(y_vae_list, epochs,dataset_dict,optimizer,reconstruction_loss_function_name)
    trainer.generate_images(2)


if __name__ =='__main__':
    for img_dim in [32, 128, 512]:
        shape=(img_dim, img_dim, 3)
        print(shape)
        YVAE_Trainer_test(input_shape=shape)
        YVAE_Trainer_generate_images_test(input_shape=shape)
    for loss in ['mse', 'binary_crossentropy', 'log_cosh','huber']:
        YVAE_Trainer_test_reconstruction(reconstruction_loss_function_name=loss)
    print("all done! :)")