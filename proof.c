#include "nn.h"

int main(void) {
  Neuron_Layer layers[] = {
    create_layer(.randomize = false, .inputs = 1, .outputs = 1),
    create_layer(.randomize = false, .inputs = 1, .outputs = 1),
    create_layer(.randomize = false, .inputs = 1, .outputs = 1),
  };

  Neuron_Network nn = neuron_create(layers, ARRAY_LEN(layers));

  NMatrix X = mat_init(1, 1, (float[]) { 4.0 });
  mat_copy(nn.w[0], mat_init(1, 1, (float[]){ 1.0 }));
  mat_copy(nn.w[1], mat_init(1, 1, (float[]){ 0.5 }));
  mat_copy(nn.w[2], mat_init(1, 1, (float[]){ 0.5 }));

  NMatrix* h = create_outputs(nn);

  forward(&nn, h, X);

  // x = 4
  // w[0] = 1
  // b[0] = 0
  // h[0] = sig(4*1 + 0) = 0.98201379
  // =============
  // x = 0.98201379
  // w[1] = 0.5
  // b[1] = 0
  // h[1] = sig(0.98201379*0.5 + 0) = 0.6203436024
  // =============
  // x = 0.6203436024
  // w[2] = 0.5
  // b[2] = 0
  // h[2] = sig(0.6203436024*0.5 + 0) = 0.5769271953

  mat_print(X);
  printf(" = ");
  mat_print(h[0]);
  mat_print(h[1]);
  mat_print(h[2]);
  printf("\n");

  NMatrix T = mat_init(1, 1, (float[]) { 0.7 });
  Neuron_Network grad = neuron_clone(nn);
  NMatrix* dL_dh = create_outputs(nn);
  NMatrix* dL_dz = create_outputs(nn);

  backward(&nn, h, &grad, dL_dh, dL_dz, X, T);

  // dL/dh[2] = 2*(0.5769271953 - 0.7) = -0.2461456094
  // =====================
  // sig'(h[2]) = sig'(0.5769271953) = 0.2440822066
  // dL/dz[2] = sig'(h[2]) * dL/dh[2] = 0.2440822066 * -0.2461456094 = -0.06007976349
  // dL/dh[1] = dL/dz[2] * w[2] = -0.06007976349 * 0.5 = -0.03003988174
  // =====================
  // sig'(h[1]) = sig'(0.6203436024) = 0.2355174174
  // dL/dz[1] = sig'(h[1]) * dL/dh[1] = 0.2355174174 * -0.03003988174 = -0.007074915366
  // dL/dh[0] = dL/dz[1] * w[1] = -0.007074915366 * 0.5 = -0.003537457683
  // =====================
  // sig'(h[0]) = sig'(0.98201379) = 0.01766270625
  // dL/dz[0] = sig'(h[0]) * dL/dh[0] = 0.01766270625 * -0.003537457683 = -0.00006248107593

  printf("dL/dh = ");
  mat_print(dL_dh[0]);
  mat_print(dL_dh[1]);
  mat_print(dL_dh[2]);
  printf("\n");
  printf("dL/dz = ");
  mat_print(dL_dz[0]);
  mat_print(dL_dz[1]);
  mat_print(dL_dz[2]);
  printf("\n");
  printf("w = ");
  mat_print(grad.w[0]);
  mat_print(grad.w[1]);
  mat_print(grad.w[2]);
  printf("\n");
  printf("b = ");
  mat_print(grad.b[0]);
  mat_print(grad.b[1]);
  mat_print(grad.b[2]);
  printf("\n");

  /*for (int i = 0; i < 1; i++) {
    neuron_zero(&grad);

    for (int i = 0; i < X.rows; i++) {
      NMatrix input = mat_row(X, i);
      NMatrix target = mat_row(T, i);

      forward(&nn, outputs, input);
      backward(&nn, outputs, &delta_grad, errors, deltas, input, target);

      neuron_add(&grad, &delta_grad);
    }

    neuron_weighted_add(&nn, &grad, -0.5);
  }*/

  return 0;
}
