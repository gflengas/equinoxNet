#include "connected_layer.h"
#include "custom_math.h"

FC_layer init_FC_layer(int batch, int inputs, int outputs)
{
    FC_layer l = {0};
    //input variables
    l.inputs = inputs;
    l.batch=batch;
    l.h = 1;
    l.w = 1;
    l.c = inputs;

    l.weight_updates = calloc(inputs*outputs, sizeof(float));
    l.bias_updates = calloc(outputs, sizeof(float));

    l.weights = calloc(outputs*inputs, sizeof(float));
    l.biases = calloc(outputs, sizeof(float));

    //output variables
    l.outputs = outputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;

    l.output = calloc(batch*outputs, sizeof(float));
    l.delta = calloc(batch*outputs, sizeof(float));

    //initialise weights based on the He initialization
    float scale = (float)sqrt(2./(outputs*inputs));
    for(int i = 0; i < outputs*inputs; ++i) l.weights[i] = scale* rand_uniform(-1,1);//rand_normal();

    for(int i = 0; i < outputs; ++i) l.biases[i] = 0;

    return l;
}


void FC_layer_fwd(FC_layer l, network net)
{
    //zero out the output values
    fill(l.outputs*l.batch, 0, l.output, 1);
    //setup gemm size parameters
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = net.input;
    float *b = l.weights;
    float *c = l.output;
    gemm_nt(m,n,k,a,k,b,k,c,n);
    //add biases
    for(int i = 0; i < l.batch; ++i){
        axpy(l.outputs, 1, l.biases, 1, l.output + i*l.outputs, 1);
    }
}

void FC_layer_bwd(FC_layer l, network net)
{
    //zero out the weight and biases updates values
    fill(l.inputs*l.outputs, 0, l.weight_updates, 1);
    fill(l.outputs, 0, l.bias_updates, 1);

    //calculate updates for biases
    for(int i = 0; i < l.batch; ++i){
        axpy(l.outputs, 1, l.delta + i*l.outputs, 1, l.bias_updates, 1);
    }
    //calculate updates for weights with gemm
    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float *a = l.delta;
    float *b = net.input;
    float *c = l.weight_updates;

    gemm_tn(m,n,k,a,m,b,n,c,n);
    //calculate gradient
    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = net.delta;

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

void free_FC_layer(FC_layer l){

    free(l.biases);
    free(l.bias_updates);

    free(l.weights);
    free(l.weight_updates);

    free(l.delta);
    free(l.output);
}