#include "network.h"


network make_network(int n)
{
    network net = {0};
    net.n = n;
    net.layers = (layer*)calloc(net.n, sizeof(layer));
    return net;
}

void forward_network(network net, network_state state)
{
    state.workspace = net.workspace;
    for(int i = 0; i < net.n; ++i){
        state.index = i;
        layer l = net.layers[i];
        if(l.delta){
            scale(l.outputs * l.batch, 0, l.delta, 1);
        }
        l.forward(l, state);
        state.input = l.output;
        /*
        float avg_val = 0;
        int k;
        for (k = 0; k < l.outputs; ++k) avg_val += l.output[k];
        printf(" i: %d - avg_val = %f \n", i, avg_val / l.outputs);
        */
    }
}

void update_network(network net)
{
    for(int i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update){
            l.update(l, net.batch, net.learning_rate, net.momentum, net.decay);
        }
    }
}

void backward_network(network net, network_state state)
{
    int i;
    float *original_input = state.input;
    float *original_delta = state.delta;
    state.workspace = net.workspace;
    for(i = net.n-1; i >= 0; --i){
        state.index = i;
        if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }else{
            layer prev = net.layers[i-1];
            state.input = prev.output;
            state.delta = prev.delta;
        }
        layer l = net.layers[i];
        l.backward(l, state);
    }
}
float train_network_datum(network net, float *x, float *y)
{
    network_state state={0};
    state.index = 0;
    state.net = net;
    state.input = x;
    state.delta = 0;
    state.truth = y;
    forward_network(net, state);
    backward_network(net, state);
    float error = get_network_cost(net);
    update_network(net);
    return error;
}

float train_network_sgd(network net, data d, int n)
{
    int batch = net.batch;
    float* X = (float*)calloc(batch * d.X.cols, sizeof(float));
    float* y = (float*)calloc(batch * d.y.cols, sizeof(float));
    float b_acc,t_acc=0;
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_random_batch(d, batch, X, y);
        float err = train_network_datum(net, X, y);
        sum += err;
        b_acc= batch_acc(batch,10,net.layers[3].output,y);
        t_acc+=b_acc;
        printf("------Batch :%d-----\n", i);
        printf("loss: %f accuracy: %.2f %c \n", err/batch, b_acc * 100, '%');
    }
    printf("average loss: %f accuracy: %.2f %c \n",sum/(batch*n),t_acc/n*100,'%');
    free(X);
    free(y);
    return (float)sum/(n*batch);
}

float get_network_cost(network net)
{
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    return sum/count;
}