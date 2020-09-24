import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np 


def save_frozen_model(model, filename, verbose=False):
    '''
    Saves a stripped down, frozen version of the model file

    Parameters
    ----------
    model: Keras model
        The trained model to freeze and save
    
    filename: string
        The filename to save as
    
    verbose: boolean
        If True prints out each layer of the graph. Default = False

    Output
    ------
    filename.pb: Keras frozen model
        The frozen serialized model file
    filename.pbtxt: json-like text file
        The same mode, human-readable
    '''
    frozen_out_path = 'models/frozen'
    output_filename = filename

    model = model

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    if verbose:
        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 60)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)

    # Take note of these! Especially the outputs
    print("-" * 60)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph to disk
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=frozen_out_path,
                    name=f"{output_filename}.pb",
                    as_text=False)
                    
    # Save its text representation
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=frozen_out_path,
                    name=f"{output_filename}.pbtxt",
                    as_text=True)

'''
# TODO
# Graph optimizer into function
python -m tensorflow.python.tools.optimize_for_inference --input path/to/frozen_graph.pb
--output path/to/optmized_graph.pb --frozen_graph=True --input_names=x --output_names=Identity
'''

'''
# TODO
# UFF converter to function
/usr/lib/python3.6/dist-packages/uff/bin$ python convert_to_uff.py path/to/optimized_graph.pb
'''