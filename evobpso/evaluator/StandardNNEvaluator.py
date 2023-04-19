from evobpso.evaluator.Evaluator import Evaluator


class StandardNNEvaluator(Evaluator):

    def evaluate_for_train(self, position, save_model_folder=None):

        decoded_architecture = self.architecture_decoder.decode(position)
        architecture_model = self.model_creator.create_model(architecture=decoded_architecture)
        architecture_model.compile(loss=self.training_params.loss,
                                   optimizer=self.training_params.optimizer,
                                   metrics=self.training_params.metrics)
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.data_loader()
        print('evaluating model of size: ' + str(len(architecture_model.layers)))
        history = architecture_model.fit(x_train, y_train, epochs=self.training_params.train_eval_epochs, batch_size=self.training_params.batch_size,
                                         validation_data=(x_val, y_val))

        if (save_model_folder != None):
            architecture_model.save(save_model_folder)

        evaluation_result = history.history['val_loss'][-1]

        return evaluation_result

    def evaluate_for_test(self, position, save_model_folder=None):
        decoded_architecture = self.architecture_decoder.decode(position)
        architecture_model = self.model_creator.create_model(architecture=decoded_architecture)
        architecture_model.compile(loss=self.training_params.loss,
                                   optimizer=self.training_params.optimizer,
                                   metrics=self.training_params.metrics)
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = self.data_loader()
        print('testing model of size: ' + str(len(architecture_model.layers)))

        history = architecture_model.fit(x_train, y_train, epochs=self.training_params.best_solution_training_epochs, batch_size=self.training_params.batch_size,
                                         validation_data=(x_val, y_val))

        evaluation_result = architecture_model.evaluate(x_test, y_test)

        if (save_model_folder != None):
            architecture_model.save(save_model_folder)

        return evaluation_result







