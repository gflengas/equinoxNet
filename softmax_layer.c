#include "softmax_layer.h"
#include "custom_math.h"

softmax_layer init_softmax_layer(int batch, int inputs)
{
    softmax_layer l = {0};
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));
    return l;
}

void softmax_fwd(const softmax_layer l, network net)
{
    for (int b = 0; b < l.batch; b++)
    {
        softmax(net.input+b*l.inputs,l.inputs,l.output+b*l.inputs);
    }
    //loss function

    softmax_cros_ent(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
    l.cost[0] = sum_array(l.loss, l.batch*l.inputs);

}

void softmax(float *input, int n, float *output)
{
//    printf("sfm\n");
    float sum = 0;
    float largest = -FLT_MAX;
    for(int i = 0; i < n; ++i){
        if(input[i] > largest) largest = input[i];
    }

    for(int i = 0; i < n; ++i){
        float e = exp(input[i] - largest);
        sum += e;
        output[i] = e;
    }

    for(int i = 0; i < n; ++i){
        output[i] /= sum;
    }
}

void softmax_cros_ent(int n, float *pred, float *truth, float *delta, float *error)
{
    for(int i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];

        p+= 1e-9; //small value is added just in case we have pred 0 to avoid log 0
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;

//        if(t==1){
//            printf("%d truth %f pred %f error %f delta %f\n",i,t,p,error[i],delta[i]);
//        }
    }
}

void softmax_bwd(const softmax_layer l, network net)
{
    axpy(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);

}