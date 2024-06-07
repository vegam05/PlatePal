from tensorflow.keras import callbacks
import numpy as np
import csv

class CustomCallback(callbacks.Callback):
    def __init__(self, test_generator, test_steps, model_name):
        self.test_generator = test_generator
        self.test_steps = test_steps
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs={}):        
        #Evaluate the model every n epochs
        if (epoch + 1) % self.test_steps == 0 and epoch != 0:
            
            loss, acc = self.model.evaluate_generator(self.test_generator)
                                
            print('\nValidation loss: {}, acc: {}\n'.format(loss, acc))
            
            writeValToCSV(self, epoch, loss, acc)
                
#writes validation metrics to csv file
def writeValToCSV(model_name, epoch, loss, acc):
    with open(model_name + '(Validation).csv', 'a', newline='') as csvFile:
        metricWriter = csv.writer(csvFile)
        metricWriter.writerow([epoch, loss, acc])