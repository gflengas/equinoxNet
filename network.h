#ifndef DEVILTEST_NETWORK_H
#define DEVILTEST_NETWORK_H
#include "core.h"
#include "data.h"
#include "conv_layer.h"
#include "maxpool_layer.h"
#include "connected_layer.h"
#include "softmax_layer.h"
network make_network(int n);
void forward_network(network net, network_state state);
void update_network(network net);
void backward_network(network net, network_state state);
float train_network_datum(network net, float *x, float *y);
float train_network_sgd(network net, data d, int n);
float get_network_cost(network net);
#endif //DEVILTEST_NETWORK_H
