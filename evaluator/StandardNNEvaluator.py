from evaluator.Evaluator import Evaluator


class StandardNNEvaluator(Evaluator):

    def evaluate(self, position, save_model_folder):

        decoded_architecture = self.architecture_decoder.decode(position)
        architecture_model = self.model_creator.create_model(architecture=decoded_architecture)
        architecture_model.compile(loss=self.training_params.loss,
                                   optimizer=self.training_params.optimizer,
                                   metrics=self.training_params.metrics)
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.data_loader()
        print('evaluating model of size: ' + str(len(architecture_model.layers)))
        history = architecture_model.fit(x_train, y_train, epochs=self.training_params.epochs, batch_size=self.training_params.batch_size,
                                         validation_data=(x_val, y_val))

        architecture_model.save(save_model_folder)

        evaluation_result = history.history['val_loss'][-1]

        return evaluation_result






