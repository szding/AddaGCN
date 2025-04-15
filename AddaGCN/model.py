from keras import activations, initializers, constraints
from keras import regularizers
from keras.layers import Layer, Input, Dense
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

class GraphConvolution(Layer):

    def __init__(
            self,
            units,
            name=None,
            activation=None,
            use_bias=False,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs
    ):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)


        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True  
        super(GraphConvolution, self).__init__(name=name, **kwargs)

    def compute_output_shape(self, input_shapes):

        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return output_shape

    def build(self, input_shapes):

        super(GraphConvolution, self).build(input_shapes)

        features_shape = input_shapes[0]
        input_dim = features_shape[1]


        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True
            )

    def call(self, inputs):
        features, A = inputs
        output = K.dot(K.dot(A, features), self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)

        return self.activation(output)

def build_models(inp_dim, emb_dim, n_cls_source, use_bias=False,
                 loss_weight = [0.1,1], alpha_lr=[0.01,0.01]):
    inputs = Input(shape=(inp_dim,))
    G = Input(shape=(None,), sparse=True)

    x4 = GraphConvolution(1024, activation='relu', name='en0', use_bias=use_bias, kernel_regularizer=l2(5e-4))(
        [inputs, G])
    embedding = Dense(emb_dim, activation='relu', name="en1", use_bias=use_bias)(x4)

    source_classifier = Dense(n_cls_source, activation='softmax', name="mo", use_bias=use_bias)(embedding)

    domain_classifier = Dense(32, activation='linear', name="do1", use_bias=use_bias)(embedding)
    domain_classifier = Dense(2, activation='softmax', name="do", use_bias=use_bias)(domain_classifier)

    xr = Dense(1024, activation='linear', name="de1", use_bias=use_bias)(embedding)
    outputs = Dense(inp_dim, activation='linear', name="de0", use_bias=use_bias)(xr)

    domain_classification_model = Model(inputs=[inputs, G], outputs=[domain_classifier])
    domain_classification_model.compile(optimizer=Adam(learning_rate=alpha_lr[1]),
                                        loss={'do': 'categorical_crossentropy'}, metrics=['accuracy'])

    embeddings_model = Model(inputs=[inputs, G], outputs=[embedding])

    source_classification_model = Model(inputs=[inputs, G], outputs=[source_classifier])
    source_classification_model.compile(optimizer=Adam(learning_rate=alpha_lr[0]),
                                        loss={'mo': 'kld'},
                                        loss_weights={'mo': loss_weight[1]},
                                        metrics={'mo': 'mae'})

    comb_model = Model(inputs=[inputs, G], outputs=[source_classifier, domain_classifier])
    comb_model.compile(optimizer="Adam",
                       loss={'mo': 'kld',
                             'do': 'categorical_crossentropy'},
                       loss_weights={'mo': loss_weight[1],
                                     'do': loss_weight[0]},
                       metrics=['accuracy'], )


    return comb_model, source_classification_model, domain_classification_model, embeddings_model




