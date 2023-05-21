import sys
sys.path.append('yvae')


from yvae_trainer import *
from yvae_callbacks import *
from yvae_data_helper import *
from yvae_classification_trainer import *

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

def YvaeClassifierSavingCallbackTest(
        input_shape=(64,64,3),
        latent_dim=10,
        n_classes=2,
        epochs=2,
        saved_model_folder='../../../../../scratch/jlb638/yvae_models/yvae/unit_testing/'):
    dataset_names=["jlbaker361/flickr_humans_mini", "jlbaker361/anime_faces_mini"]
    n_classes=len(dataset_names)
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0001)
    classifier_model=get_classifier_model(latent_dim,input_shape,n_classes)
    batch_size=4
    
    dataset=yvae_get_labeled_dataset_train(batch_size=batch_size, dataset_names=dataset_names,image_dim=input_shape[1])
    trainer=YVAE_Classifier_Trainer(classifier_model, epochs, optimizer, dataset)
    callback=YvaeClassifierSavingCallback(trainer,saved_model_folder,1,1)
    callback(1)
    classifier_model=tf.saved_model.load(saved_model_folder+"classifier_model")




if __name__=='__main__':
    for dim in [64]:
        input_shape=(dim,dim,3)
        YvaeClassifierSavingCallbackTest(input_shape=input_shape)
        break
        YvaeImageGenerationCallback_test(input_shape=input_shape)
        YvaeSavingCallback_test(input_shape=input_shape)
    print("all done :)")