from adain_model import *
import numpy as np
import time

class AdaInTrainer():
    def __init__(self, encoder, decoder, loss_net, style_weight, optimizer, loss_fn ,callbacks,epochs,train_dataset,save_model_path):
        self.encoder = encoder
        self.decoder = decoder
        self.loss_net = loss_net
        self.style_weight = style_weight
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.callbacks=callbacks
        self.epochs=epochs
        self.train_dataset=train_dataset
        self.save_model_path=save_model_path


    def train_step(self, inputs):
        style, content = inputs

        # Initialize the content and style loss.
        loss_content = 0.0
        loss_style = 0.0

        with tf.GradientTape() as tape:
            # Encode the style and content image.
            style_encoded = self.encoder(style)
            content_encoded = self.encoder(content)

            # Compute the AdaIN target feature maps.
            t = ada_in(style=style_encoded, content=content_encoded)

            # Generate the neural style transferred image.
            reconstructed_image = self.decoder(t)

            # Compute the losses.
            reconstructed_vgg_features = self.loss_net(reconstructed_image)
            style_vgg_features = self.loss_net(style)
            loss_content = self.loss_fn(t, reconstructed_vgg_features[-1])
            for inp, out in zip(style_vgg_features, reconstructed_vgg_features):
                mean_inp, std_inp = get_mean_std(inp)
                mean_out, std_out = get_mean_std(out)
                loss_style += self.loss_fn(mean_inp, mean_out) + self.loss_fn(
                    std_inp, std_out
                )
            loss_style = self.style_weight * loss_style
            total_loss = loss_content + loss_style

        # Compute gradients and optimize the decoder.
        trainable_vars = self.decoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {
            "style_loss": loss_style,
            "content_loss": loss_content,
            "total_loss": total_loss,
        }

    def train_loop(self):
        for e in range(self.epochs):
            losses={
            "style_loss": [],
            "content_loss": [],
            "total_loss": []
            }
            start=time.time()
            for batch in self.train_dataset:
                result=self.train_step(batch)
                for k,v in result.items():
                    losses[k].append(v)
            end=time.time()
            print("epoch {} elpased {}".format(e, end-start))
            for name,val in losses.items():
                print('\t{} sum: {} mean: {} std: {}'.format(name, np.sum(val), np.mean(val), np.std(val)))
            for callback in self.callbacks:
                callback(self,e)
        return {
            "style_loss": np.mean(losses["style_loss"]),
            "content_loss": np.mean(losses["content_loss"]),
            "total_loss": np.mean(losses["total_loss"])
        }

    def test_step(self, inputs):
        style, content = inputs

        # Initialize the content and style loss.
        loss_content = 0.0
        loss_style = 0.0

        # Encode the style and content image.
        style_encoded = self.encoder(style)
        content_encoded = self.encoder(content)

        # Compute the AdaIN target feature maps.
        t = ada_in(style=style_encoded, content=content_encoded)

        # Generate the neural style transferred image.
        reconstructed_image = self.decoder(t)

        # Compute the losses.
        recons_vgg_features = self.loss_net(reconstructed_image)
        style_vgg_features = self.loss_net(style)
        loss_content = self.loss_fn(t, recons_vgg_features[-1])
        for inp, out in zip(style_vgg_features, recons_vgg_features):
            mean_inp, std_inp = get_mean_std(inp)
            mean_out, std_out = get_mean_std(out)
            loss_style += self.loss_fn(mean_inp, mean_out) + self.loss_fn(
                std_inp, std_out
            )
        loss_style = self.style_weight * loss_style
        total_loss = loss_content + loss_style

        return {
            "style_loss": loss_style,
            "content_loss": loss_content,
            "total_loss": total_loss,
        }
    
    def save_decoder(self):
        tf.saved_model.save(self.decoder, self.save_model_path+"adain_decoder")