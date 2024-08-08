import math
from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler
from hls4ml.converters.utils import parse_data_format, compute_padding_1d, compute_padding_2d

@keras_handler('BayesianDropout')
def parse_bayesian_dropout_layer(keras_layer, input_names, input_shapes, data_reader, config=None):
    assert('BayesianDropout' in keras_layer['class_name'])

    layer = parse_default_keras_layer(keras_layer, input_names)
    layer['drop_rate'] = keras_layer['config']['drop_rate']
    layer['seed'] = keras_layer['config']['seed']
    
    return layer, [shape for shape in input_shapes[0]]
