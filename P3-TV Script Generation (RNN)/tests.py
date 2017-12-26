import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

def test_get_tensors(get_tensors):
    test_graph = tf.Graph()
    with test_graph.as_default():
        test_input = tf.placeholder(tf.int32, name='input')
        test_initial_state = tf.placeholder(tf.int32, name='initial_state')
        test_final_state = tf.placeholder(tf.int32, name='final_state')
        test_probs = tf.placeholder(tf.float32, name='probs')

    input_text, initial_state, final_state, probs = get_tensors(test_graph)

    # Check correct tensor
    assert input_text == test_input,\
        'Test input is wrong tensor'
    assert initial_state == test_initial_state, \
        'Initial state is wrong tensor'
    assert final_state == test_final_state, \
        'Final state is wrong tensor'
    assert probs == test_probs, \
        'Probabilities is wrong tensor'


def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    
    # TODO: Implement Function
    input_tensor = loaded_graph.get_tensor_by_name('input: 0')
    initial_state_tensor = loaded_graph.get_tensor_by_name('initial_state: 0')
    final_state_tensor = loaded_graph.get_tensor_by_name('final_state: 0')
    probs_tensor = loaded_graph.get_tensor_by_name('probs: 0')
    

    return input_tensor, initial_state_tensor, final_state_tensor, probs_tensor

test_get_tensors(get_tensors)

"""
['_ControlDependenciesController', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_add_function', '_add_op', '_apply_device_functions', '_as_graph_def', '_as_graph_element_locked', '_attr_scope', '_attr_scope_map', '_building_function', '_check_not_finalized', '_collections', '_colocation_stack', '_container', '_control_dependencies_for_inputs', '_control_dependencies_stack', '_control_flow_context', '_current_control_dependencies', '_default_original_op', '_device_function_stack', '_finalized', '_functions', '_get_control_flow_context', '_get_function', '_gradient_override_map', '_graph_def_versions', '_handle_deleters', '_handle_feeders', '_handle_movers', '_handle_readers', '_is_function', '_kernel_label_map', '_last_id', '_lock', '_name_stack', '_names_in_use', '_next_id', '_next_id_counter', '_nodes_by_id', '_nodes_by_name', '_op_to_kernel_label_map', '_original_op', '_pop_control_dependencies_controller', '_push_control_dependencies_controller', '_record_op_seen_by_control_dependencies', '_registered_ops', '_seed', '_set_control_flow_context', '_unfeedable_tensors', '_unfetchable_ops', '_unsafe_unfinalize', '_version', 'add_to_collection', 'add_to_collections', 'as_default', 'as_graph_def', 'as_graph_element', 'building_function', 'clear_collection', 'colocate_with', 'container', 'control_dependencies', 'create_op', 'device', 'finalize', 'finalized', 'get_all_collection_keys', 'get_collection', 'get_collection_ref', 'get_operation_by_name', 'get_operations', 'get_tensor_by_name', 'gradient_override_map', 'graph_def_versions', 'is_feedable', 'is_fetchable', 'name_scope', 'prevent_feeding', 'prevent_fetching', 'seed', 'unique_name', 'version']


"""