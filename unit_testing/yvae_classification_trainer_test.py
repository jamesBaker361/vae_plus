import sys
sys.path.append('yvae')

from yvae_data_helper import *
from yvae_classification_trainer import *

def YVAE_Classifier_Trainer_test(
        input_shape=(32,32,3),
    latent_dim=10,
    epochs=1):
    dataset_names=["jlbaker361/flickr_humans_mini", "jlbaker361/anime_faces_mini"]
    n_classes=len(dataset_names)
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0001)
    classifier_model=get_classifier_model(latent_dim,input_shape,n_classes)
    batch_size=4
    
    dataset=yvae_get_labeled_dataset_train(batch_size=batch_size, dataset_names=dataset_names,image_dim=input_shape[1])
    trainer=YVAE_Classifier_Trainer(classifier_model, epochs, optimizer, dataset, log_dir='logs/yvae_classification_trainer_test')
    trainer.train_loop()
    
if __name__=='__main__':
    YVAE_Classifier_Trainer_test()