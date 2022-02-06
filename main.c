#include "data.h"
#include "conv_layer.h"
#include "maxpool_layer.h"
#include "connected_layer.h"
#include "softmax_layer.h"
#include <time.h>

int main(void) {

    //load data
    clock_t start = clock();
    char *train_images = "mnist_test.csv";
    data d = load_categorical_data_csv(train_images, 0, 10);
    float b_acc, t_acc = 0, sum = 0;
    normalize_data_rows(d);

    //set network parameters
    network net = {0};
    net.cost = calloc(1, sizeof(float));
    net.batch = 8;
    net.w = 28;
    net.h = 28;
    net.c = 1;
    net.momentum = 0.9;
    net.decay = 0.00005;
    net.learning_rate = 0.01;
    net.inputs = net.h * net.w * net.c;
    net.input = calloc(net.inputs * net.batch, sizeof(float));
    net.truths = 10;
    net.truth = calloc(net.truths * net.batch, sizeof(float));

    //initialize the layers of the network
    //1 conv layer
    conv_layer conv1 = init_convolutional_layer(net.batch, net.h, net.w, net.c, 12,
                                                3, 1, 1);


    //2 maxpool layer
    maxpool_layer MP1 = init_maxpool_layer(net.batch, conv1.out_h, conv1.out_w, conv1.out_c,
                                           2, 1, 1);

    //3 FC layer
    FC_layer fc = init_FC_layer(net.batch, MP1.outputs, 10);

    //4 softmax
    softmax_layer sfm = init_softmax_layer(net.batch, 10);

    //training start
    assert(d.X.rows % net.batch == 0);
    int n = 200;// d.X.rows / net.batch;

    for (int e = 0; e < 1; ++e) {

        printf("------Epoch :%d-----\n", e + 1);
        for (int i = 0; i < n; i++) {
            //get the next batch that will be processed
            //get_next_batch(d,net.batch,i*net.batch, net.input, net.truth);
            get_random_batch(d, net.batch, net.input, net.truth);

            //forward

            //save original input and delta for backprop on conv
            float *original_input = net.input;
            float *original_delta = net.delta;

            if (conv1.delta) {
                fill(conv1.outputs * conv1.batch, 0, conv1.delta, 1);
            }
            conv_fwd(conv1, net);
            net.input = conv1.output;

            if (MP1.delta) {
                fill(MP1.outputs * MP1.batch, 0, MP1.delta, 1);
            }
            maxpool_fwd(MP1, net);
            net.input = MP1.output;

            if (fc.delta) {
                fill(fc.outputs * fc.batch, 0, fc.delta, 1);
            }
            FC_layer_fwd(fc, net);
            net.input = fc.output;

            if (sfm.delta) {
                fill(sfm.outputs * sfm.batch, 0, sfm.delta, 1);
            }
            softmax_fwd(sfm, net);

            b_acc = batch_acc(net.batch, sfm.outputs, sfm.output, net.truth);
            t_acc += b_acc;
//            printf("------Batch :%d-----\n", (i+1)*(e+1));
//            printf("loss: %f accuracy: %.2f %c \n", sfm.cost[0] / net.batch, b_acc * 100, '%');
            sum += sfm.cost[0]; // ++error

            //backward
            //3
            net.input = fc.output;
            net.delta = fc.delta;
            softmax_bwd(sfm, net);

            //2
            net.input = MP1.output;
            net.delta = MP1.delta;
            FC_layer_bwd(fc, net);

            //1
            net.input = conv1.output;
            net.delta = conv1.delta;
            maxpool_bwd(MP1, net);

            //0
            net.input = original_input;
            net.delta = original_delta;
            conv_bwd(conv1, net);

            // //upgrade network
            update_conv_layer(conv1, net.batch, net.learning_rate,
                              net.momentum, net.decay);
            FC_update(fc, net.batch, net.learning_rate, net.momentum, net.decay);
        }
        printf("average loss: %f accuracy: %.2f %c \n", sum / (net.batch * n * (e + 1)),
               t_acc / (n * (e + 1)) * 100,'%');
        float avg_val = 0;
        for (int k = 0; k < conv1.c * conv1.n * conv1.size * conv1.size; ++k) avg_val += conv1.weights[k];
        printf("Conv - avg_val = %f \n", avg_val / (conv1.c * conv1.n * conv1.size * conv1.size));
        avg_val = 0;
        for (int k = 0; k < fc.outputs * fc.inputs; ++k) avg_val += fc.weights[k];
        printf("fc - avg_val = %f \n", avg_val / (fc.outputs * fc.inputs));
    }
    clock_t stop = clock();
    double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
    printf("\nTime elapsed: %.5f\n", elapsed);
    //free layers
    free_conv_layer(conv1);
    free_maxpool_layer(MP1);
    free_FC_layer(fc);
    free_softmax_layer(sfm);
    //free data
    free_data(d);
    //free network related parameters
    free(net.input);
    free(net.truth);
    free(net.cost);
    return 0;
}