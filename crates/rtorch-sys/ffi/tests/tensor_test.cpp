#include "rtorch.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

int owner_releases = 0;
void release_vector(void* owner) {
  ++owner_releases;
  delete static_cast<std::vector<float>*>(owner);
}

void assert_close(float actual, float expected) {
  assert(std::fabs(actual - expected) < 1.0e-6F);
}

}  // namespace

int main() {
  const int64_t owned_shape[] = {2};
  auto* owned_values = new std::vector<float>{4.0F, 5.0F};
  auto* owned = rt_tensor_from_owned_data(owned_values->data(), 2, RT_SCALAR_F32,
      owned_shape, 1, owned_values, release_vector);
  assert(owned != nullptr && owner_releases == 0);
  auto* view = rt_tensor_reshape(owned, owned_shape, 1);
  rt_tensor_free(owned);
  assert(owner_releases == 0);
  float read_values[2] = {};
  assert(rt_tensor_copy_f32(view, read_values, 2));
  assert_close(read_values[0], 4.0F);
  rt_tensor_free(view);
  assert(owner_releases == 1);
  auto* invalid_values = new std::vector<float>{1.0F};
  assert(rt_tensor_from_owned_data(invalid_values->data(), 1, RT_SCALAR_F32,
      owned_shape, 1, invalid_values, release_vector) == nullptr);
  assert(owner_releases == 2);
  auto* empty_values = new std::vector<float>;
  const int64_t empty_shape[] = {0};
  auto* empty = rt_tensor_from_owned_data(empty_values->data(), 0, RT_SCALAR_F32,
      empty_shape, 1, empty_values, release_vector);
  assert(empty != nullptr && owner_releases == 3);
  rt_tensor_free(empty);
  assert(owner_releases == 3);

  const float values[] = {1.0F, 2.0F, 3.0F};
  const int64_t vector_shape[] = {3};
  RtTensor* input = rt_tensor_from_data(
      values,
      3,
      RT_SCALAR_F32,
      vector_shape,
      1,
      true);
  assert(input != nullptr);
  assert(rt_tensor_numel(input) == 3);
  assert(rt_tensor_scalar_type(input) == RT_SCALAR_F32);
  assert(rt_tensor_requires_grad(input));

  RtTensor* doubled = rt_tensor_add(input, input);
  assert(doubled != nullptr);
  RtTensor* total = rt_tensor_sum(doubled);
  assert(total != nullptr);
  assert(rt_tensor_backward(total));

  RtTensor* gradient = rt_tensor_grad(input);
  assert(gradient != nullptr);
  float gradient_values[3] = {};
  assert(rt_tensor_copy_f32(gradient, gradient_values, 3));
  for (const auto value : gradient_values) {
    assert_close(value, 2.0F);
  }

  const float matrix_values[] = {1.0F, 2.0F, 3.0F, 4.0F};
  const int64_t matrix_shape[] = {2, 2};
  RtTensor* matrix = rt_tensor_from_data(
      matrix_values,
      4,
      RT_SCALAR_F32,
      matrix_shape,
      2,
      false);
  RtTensor* ones =
      rt_tensor_ones(matrix_shape, 2, RT_SCALAR_F32, false);
  assert(matrix != nullptr);
  assert(ones != nullptr);

  RtTensor* product = rt_tensor_matmul(matrix, ones);
  assert(product != nullptr);
  float product_values[4] = {};
  assert(rt_tensor_copy_f32(product, product_values, 4));
  assert_close(product_values[0], 3.0F);
  assert_close(product_values[1], 3.0F);
  assert_close(product_values[2], 7.0F);
  assert_close(product_values[3], 7.0F);

  const uint16_t unsigned_values[] = {10, 20, 30};
  RtStatistics statistics{};
  assert(rt_statistics(
      unsigned_values,
      3,
      RT_SCALAR_U16,
      &statistics));
  assert_close(static_cast<float>(statistics.mean), 20.0F);

  rt_tensor_free(product);
  rt_tensor_free(ones);
  rt_tensor_free(matrix);
  rt_tensor_free(gradient);
  rt_tensor_free(total);
  rt_tensor_free(doubled);
  rt_tensor_free(input);
  return 0;
}
