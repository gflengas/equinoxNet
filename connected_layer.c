#include "connected_layer.h"
#include "custom_math.h"

FC_layer init_FC_layer(int batch, int inputs, int outputs)
{
    FC_layer l = { (LAYER_TYPE)0 };
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch=batch;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;

    l.output = calloc(batch*outputs, sizeof(float));
    l.delta = calloc(batch*outputs, sizeof(float));

    l.weight_updates = calloc(inputs*outputs, sizeof(float));
    l.bias_updates = calloc(outputs, sizeof(float));

    l.weights = calloc(outputs*inputs, sizeof(float));
    l.biases = calloc(outputs, sizeof(float));

    //initialise weights based on the He initialization
    float scale = sqrt(2./(outputs*inputs));
    for(int i = 0; i < outputs*inputs; ++i) l.weights[i] = scale*rand_normal();

    for(int i = 0; i < outputs; ++i) l.biases[i] = 0;


    l.forward = FC_layer_fwd;
    l.backward = FC_layer_bwd;
    l.update = FC_update;

    return l;
}


void FC_layer_fwd(FC_layer l, network_state state)
{
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = state.input;
    float *b = l.weights;
    float *c = l.output;
    gemm_nt(m,n,k,a,k,b,k,c,n);
    //add biases
    for(int b = 0; b < l.batch; ++b){
        for(int i = 0; i < l.outputs; ++i){
            l.output[(b*n + i)*l.outputs] += l.biases[i];
        }
    }
}

void FC_layer_bwd(FC_layer l, network_state state)
{
    //l.delta = l.output;  -----------------------------------------------------------------------------------------------

    for (int i = 0; i < l.inputs*l.outputs; ++i) {
        l.weight_updates[i]=0;
    }
    for (int i = 0; i < l.outputs; ++i) {
        l.bias_updates[i]=0;
    }
    //calculate updates for biases
    for (int b = 0; b < l.batch; b++)
    {
        for (int i = 0; i < l.outputs; i++)
        {
            l.bias_updates[i] += sum_array(l.delta+(i+b*l.outputs),1);
        }

    }
    //calculate updates for weights
    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float *a = l.delta;
    float *b = state.input;
    float *c = l.weight_updates;

    gemm_tn(m,n,k,a,m,b,n,c,n);
    //calculate gradient
    for (int i = 0; i < l.outputs*l.inputs; ++i) {
        state.delta[i]=0;
    }
    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = state.delta;

    if(c) gemm_nn(m,n,k,a,k,b,n,c,n);

}
void FC_update(FC_layer l, int batch, float learning_rate, float momentum, float decay)
{
    axpy(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scale(l.outputs, momentum, l.bias_updates, 1);

    axpy(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scale(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}