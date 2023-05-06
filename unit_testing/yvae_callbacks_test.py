import sys
sys.path.append('yvae')


from yvae_trainer import *
from yvae_callbacks import *
from yvae_data_helper import *

def YvaeImageGenerationCallback_test(
    input_shape=(32,32,3),
    latent_dim=10,
    n=2,
    epochs=2):
    y_vae_list=get_y_vae_list(latent_dim, input_shape,n)
    #dataset_dict={name:tf.data.Dataset.from_tensor_slices(tf.random.normal((8,*input_shape))).batch(4) for _ in range(n) for name in ["a","b","c"]}
    dataset_dict=yvae_get_dataset_train(batch_size=4, image_dim=input_shape[1])
    optimizer=keras.optimizers.Adam(learning_rate=0.0001)
    trainer=YVAE_Trainer(y_vae_list, epochs,dataset_dict,optimizer)
    callback=YvaeImageGenerationCallback(trainer, dataset_dict, 'exploration/', 3)
    callback(0)

def YvaeSavingCallback_test(
        input_shape=(64,64,3),
        latent_dim=10,
        n=2,
        epochs=2,
        saved_model_folder='../../../../../scratch/jlb638/yvae_models/yvae/unit_testing/'
):
    y_vae_list=get_y_vae_list(latent_dim, input_shape,n)
    dataset_dict=yvae_get_dataset_train(batch_size=4, image_dim=input_shape[1])
    optimizer=keras.optimizers.Adam(learning_rate=0.0001)
    trainer=YVAE_Trainer(y_vae_list, epochs,dataset_dict,optimizer)
    callback=YvaeSavingCallback(trainer, saved_model_folder,0,1)
    callback(10)
    encoder=tf.saved_model.load(saved_model_folder+"encoder")
    for x in range(n):
        decoder=tf.saved_model.load(saved_model_folder+"decoder_{}".format(x))


if __name__=='__main__':
    for dim in [64,128,512]:
        input_shape=(dim,dim,3)
        YvaeImageGenerationCallback_test(input_shape=input_shape)
        YvaeSavingCallback_test(input_shape=input_shape)
    print("all done :)")