#include "data.h"
#include "network.h"
#include <time.h>

int main() {
    //load data
    clock_t start = clock();
    char *train_images = "C:\\Users\\giorgosflg\\CLionProjects\\deviltest\\mnist_train.csv";
    data d = load_categorical_data_csv(train_images, 0, 10);
    float b_acc,t_acc,sum;
    normalize_data_rows(d);
    //set network parameters
    network net=make_network(4);
    net.cost = calloc(1, sizeof(float));
    net.batch=4;
    net.w=28;
    net.h=28;
    net.c=1;
    net.momentum=0.9;
    net.decay=0.00005;
    net.learning_rate=0.01;
    size_t workspace_size = 0;
    net.inputs = net.h * net.w * net.c;
    net.input = calloc(net.inputs*net.batch, sizeof(float));
    net.truths = 10;
    net.truth = calloc(net.truths*net.batch, sizeof(float));
    conv_layer conv1 = init_convolutional_layer(net.batch, net.h, net.w, net.c, 12, 3, 1 , 1);
    //initialize the layers of the network
    //1 conv layer
    net.layers[0]=conv1;
    if(conv1.workspace_size > workspace_size) workspace_size = conv1.workspace_size;
    net.workspace = calloc(1, workspace_size);
    //2 maxpool layer
    maxpool_layer MP1 = init_maxpool_layer(net.batch, conv1.out_h, conv1.out_w, conv1.out_c, 2, 1, 1);
    net.layers[1]=MP1;
    //3 FC layer
    FC_layer fc = init_FC_layer(net.batch, MP1.outputs, 10);
    net.layers[2]=fc;
    //4 softmax
    softmax_layer sfm = init_softmax_layer(net.batch,10);
    net.layers[3]=sfm;
    //training start
    assert(d.X.rows % net.batch == 0);
    for (int e = 0; e < 10 ; ++e) {
        train_network_sgd(net,d,200);
    }

    clock_t stop = clock();
    double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
    printf("\nTime elapsed: %.5f\n", elapsed);
}
