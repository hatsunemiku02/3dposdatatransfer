import tensorflow as tf

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    latest_ckpt= tf.train.latest_checkpoint("mysave/")
    restore_saver= tf.train.import_meta_graph('./mysave/gogo.meta')
    restore_saver.restore(sess,latest_ckpt)

     # Output nodes
    output_node_names =[n.name for n in tf.get_default_graph().as_graph_def().node]


    #output_graph_def= tf.graph_util.convert_variables_to_constants(sess,sess.graph_def , ["inputs/enc_in","inputs/dec_out","linear_model/w1","linear_model/b1","linear_model/two_linear_0/w2_0","linear_model/two_linear_0/b2_0","linear_model/two_linear_0/w3_0","linear_model/two_linear_0/b3_0","linear_model/two_linear_1/w2_1","linear_model/two_linear_1/b2_1","linear_model/two_linear_1/w3_1","linear_model/two_linear_1/b3_1","linear_model/w4","linear_model/b4"])

    output_graph_def= tf.graph_util.convert_variables_to_constants(sess,sess.graph_def , ["inputs/enc_in","linear_model/MatMul","linear_model/add","linear_model/Relu","linear_model/two_linear_0/MatMul","linear_model/two_linear_0/add","linear_model/two_linear_0/Relu","linear_model/two_linear_0/MatMul_1","linear_model/two_linear_0/add_1","linear_model/two_linear_0/Relu_1","linear_model/two_linear_1/MatMul","linear_model/two_linear_1/add","linear_model/two_linear_1/Relu","linear_model/two_linear_1/MatMul_1","linear_model/two_linear_1/add_1","linear_model/two_linear_1/Relu_1","linear_model/MatMul_1","linear_model/add_1"])


    #output_graph_def= tf.graph_util.convert_variables_to_constants(sess,sess.graph_def , ["inputs/enc_in","inputs/dec_out","linear_model/MatMul"])

    tf.train.write_graph(output_graph_def,'pretrained', "graph2.pb", as_text=False)