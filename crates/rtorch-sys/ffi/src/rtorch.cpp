#include "rtorch.h"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct SampleRecord {
  fs::path path;
  int64_t label;
};

struct TransformPipelineImpl {
  bool to_float = false;
  bool has_resize = false;
  bool has_crop = false;
  int64_t resize_h = 0;
  int64_t resize_w = 0;
  int64_t crop_h = 0;
  int64_t crop_w = 0;
  std::vector<float> mean;
  std::vector<float> std_values;

  torch::Tensor apply(const torch::Tensor& input) const {
    torch::Tensor value = input;

    if (to_float && value.scalar_type() != torch::kFloat32) {
      value = value.to(torch::kFloat32);
    }

    if (has_resize) {
      if (value.dim() != 3) {
        throw std::runtime_error("resize expects a 3D tensor shaped CxHxW");
      }
      auto resized = torch::nn::functional::interpolate(
          value.unsqueeze(0),
          torch::nn::functional::InterpolateFuncOptions()
              .size(std::vector<int64_t>{resize_h, resize_w})
              .mode(torch::kBilinear)
              .align_corners(false));
      value = resized.squeeze(0);
    }

    if (has_crop) {
      if (value.dim() != 3) {
        throw std::runtime_error("center_crop expects a 3D tensor shaped CxHxW");
      }

      auto height = value.size(1);
      auto width = value.size(2);
      if (crop_h > height || crop_w > width) {
        throw std::runtime_error("center_crop dimensions exceed tensor size");
      }

      auto top = (height - crop_h) / 2;
      auto left = (width - crop_w) / 2;
      value = value.slice(1, top, top + crop_h).slice(2, left, left + crop_w);
    }

    if (!mean.empty() || !std_values.empty()) {
      if (mean.size() != std_values.size()) {
        throw std::runtime_error("normalize mean/std length mismatch");
      }
      if (value.dim() != 3) {
        throw std::runtime_error("normalize expects a 3D tensor shaped CxHxW");
      }
      if (static_cast<int64_t>(mean.size()) != value.size(0)) {
        throw std::runtime_error("normalize mean/std length must match channel count");
      }

      auto mean_tensor =
          torch::tensor(mean, torch::TensorOptions().dtype(torch::kFloat32)).view({-1, 1, 1});
      auto std_tensor =
          torch::tensor(std_values, torch::TensorOptions().dtype(torch::kFloat32))
              .view({-1, 1, 1});
      value = (value - mean_tensor) / std_tensor;
    }

    return value.contiguous();
  }
};

struct TensorDatasetImpl {
  std::vector<SampleRecord> samples;
  std::vector<std::string> class_names;
};

struct BatchStorage {
  std::vector<float> data;
  std::vector<int64_t> shape;
  std::vector<int64_t> labels;
  RtStatistics statistics{};
};

struct DataLoaderImpl {
  TensorDatasetImpl* dataset = nullptr;
  TransformPipelineImpl* pipeline = nullptr;
  size_t batch_size = 1;
  std::vector<size_t> indices;
  size_t cursor = 0;
  bool shuffle = false;
  std::mt19937 rng{std::random_device{}()};

  void reset() {
    indices.resize(dataset->samples.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      indices[i] = i;
    }
    if (shuffle) {
      std::shuffle(indices.begin(), indices.end(), rng);
    }
    cursor = 0;
  }
};

thread_local std::string g_last_error;

template <typename T, typename Fn>
T protect(Fn&& fn, T failure) {
  try {
    g_last_error.clear();
    return fn();
  } catch (const c10::Error& err) {
    g_last_error = err.what_without_backtrace();
  } catch (const std::exception& err) {
    g_last_error = err.what();
  } catch (...) {
    g_last_error = "unknown C++ exception";
  }
  return failure;
}

void scan_dataset_root(TensorDatasetImpl& dataset, const fs::path& root) {
  if (!fs::exists(root)) {
    throw std::runtime_error("dataset root does not exist");
  }
  if (!fs::is_directory(root)) {
    throw std::runtime_error("dataset root is not a directory");
  }

  std::vector<fs::path> class_dirs;
  for (const auto& entry : fs::directory_iterator(root)) {
    if (entry.is_directory()) {
      class_dirs.push_back(entry.path());
    }
  }

  std::sort(class_dirs.begin(), class_dirs.end());
  for (size_t class_index = 0; class_index < class_dirs.size(); ++class_index) {
    const auto& class_dir = class_dirs[class_index];
    dataset.class_names.push_back(class_dir.filename().string());

    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(class_dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".pt") {
        files.push_back(entry.path());
      }
    }
    std::sort(files.begin(), files.end());

    for (const auto& path : files) {
      SampleRecord record;
      record.path = path;
      record.label = static_cast<int64_t>(class_index);
      dataset.samples.push_back(std::move(record));
    }
  }

  if (dataset.samples.empty()) {
    throw std::runtime_error("dataset contains no .pt files");
  }
}

torch::Tensor load_tensor_from_file(const fs::path& path) {
  torch::Tensor tensor;
  torch::load(tensor, path.string());
  return tensor;
}

RtStatistics calculate_statistics(const torch::Tensor& input) {
  if (input.numel() == 0) {
    throw std::runtime_error("statistics require at least one value");
  }

  auto source =
      input
          .to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64))
          .contiguous();
  const auto count = static_cast<size_t>(source.numel());

  RtStatistics statistics{};
  statistics.count = count;
  statistics.sum = source.sum().item<double>();
  statistics.mean = source.mean().item<double>();
  auto deviations = source - statistics.mean;
  const auto squared_deviation_sum =
      (deviations * deviations).sum().item<double>();
  statistics.population_std =
      std::sqrt(squared_deviation_sum / static_cast<double>(count));
  statistics.sample_std =
      count > 1
          ? std::sqrt(squared_deviation_sum / static_cast<double>(count - 1))
          : 0.0;
  statistics.min = source.min().item<double>();
  statistics.max = source.max().item<double>();
  return statistics;
}

c10::ScalarType scalar_type_from_ffi(int scalar_type) {
  switch (scalar_type) {
    case RT_SCALAR_BOOL:
      return c10::kBool;
    case RT_SCALAR_U8:
      return c10::kByte;
    case RT_SCALAR_I8:
      return c10::kChar;
    case RT_SCALAR_I16:
      return c10::kShort;
    case RT_SCALAR_I32:
      return c10::kInt;
    case RT_SCALAR_I64:
      return c10::kLong;
    case RT_SCALAR_U16:
      return c10::kUInt16;
    case RT_SCALAR_U32:
      return c10::kUInt32;
    case RT_SCALAR_U64:
      return c10::kUInt64;
    case RT_SCALAR_F16:
      return c10::kHalf;
    case RT_SCALAR_BF16:
      return c10::kBFloat16;
    case RT_SCALAR_F32:
      return c10::kFloat;
    case RT_SCALAR_F64:
      return c10::kDouble;
    default:
      throw std::runtime_error("unsupported statistics scalar type");
  }
}

}  // namespace

struct RtTransformPipeline {
  TransformPipelineImpl inner;
};

struct RtTensorDataset {
  TensorDatasetImpl inner;
};

struct RtDataLoader {
  DataLoaderImpl inner;
};

struct RtOwnedBatch {
  BatchStorage inner;
};

extern "C" char* rt_last_error_message(void) {
  char* message = static_cast<char*>(std::malloc(g_last_error.size() + 1));
  if (message == nullptr) {
    return nullptr;
  }
  std::memcpy(message, g_last_error.c_str(), g_last_error.size() + 1);
  return message;
}

extern "C" void rt_last_error_free(char* message) {
  std::free(message);
}

extern "C" bool rt_statistics(
    const void* data,
    size_t len,
    int scalar_type,
    RtStatistics* out) {
  return protect<bool>([&]() {
    if (len == 0) {
      throw std::runtime_error("statistics require at least one value");
    }
    if (data == nullptr || out == nullptr) {
      throw std::runtime_error("statistics data or output pointer was null");
    }
    if (len > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      throw std::runtime_error("statistics input is too large");
    }

    auto values = torch::from_blob(
        const_cast<void*>(data),
        {static_cast<int64_t>(len)},
        torch::TensorOptions().dtype(scalar_type_from_ffi(scalar_type)));
    *out = calculate_statistics(values);
    return true;
  }, false);
}

extern "C" RtTransformPipeline* rt_transform_pipeline_new(
    bool to_float,
    int64_t resize_h,
    int64_t resize_w,
    int64_t crop_h,
    int64_t crop_w,
    const float* mean,
    size_t mean_len,
    const float* std_values,
    size_t std_values_len) {
  return protect<RtTransformPipeline*>([&]() {
    auto pipeline = std::make_unique<RtTransformPipeline>();
    pipeline->inner.to_float = to_float;
    pipeline->inner.has_resize = resize_h > 0 && resize_w > 0;
    pipeline->inner.has_crop = crop_h > 0 && crop_w > 0;
    pipeline->inner.resize_h = resize_h;
    pipeline->inner.resize_w = resize_w;
    pipeline->inner.crop_h = crop_h;
    pipeline->inner.crop_w = crop_w;
    if (mean_len > 0) {
      if (mean == nullptr) {
        throw std::runtime_error("normalize mean pointer was null");
      }
      pipeline->inner.mean.assign(mean, mean + mean_len);
    }
    if (std_values_len > 0) {
      if (std_values == nullptr) {
        throw std::runtime_error("normalize std pointer was null");
      }
      pipeline->inner.std_values.assign(
          std_values,
          std_values + std_values_len);
    }
    if (std::any_of(
            pipeline->inner.std_values.begin(),
            pipeline->inner.std_values.end(),
            [](float value) { return value == 0.0F; })) {
      throw std::runtime_error("normalize std values must be non-zero");
    }
    return pipeline.release();
  }, nullptr);
}

extern "C" void rt_transform_pipeline_free(RtTransformPipeline* pipeline) {
  delete pipeline;
}

extern "C" RtTensorDataset* rt_tensor_dataset_new(const char* root) {
  return protect<RtTensorDataset*>([&]() {
    if (root == nullptr) {
      throw std::runtime_error("dataset root pointer was null");
    }
    auto dataset = std::make_unique<RtTensorDataset>();
    scan_dataset_root(dataset->inner, fs::path(root));
    return dataset.release();
  }, nullptr);
}

extern "C" void rt_tensor_dataset_free(RtTensorDataset* dataset) {
  delete dataset;
}

extern "C" size_t rt_tensor_dataset_len(const RtTensorDataset* dataset) {
  return protect<size_t>([&]() {
    if (dataset == nullptr) {
      throw std::runtime_error("dataset pointer was null");
    }
    return dataset->inner.samples.size();
  }, 0);
}

extern "C" size_t rt_tensor_dataset_class_count(const RtTensorDataset* dataset) {
  return protect<size_t>([&]() {
    if (dataset == nullptr) {
      throw std::runtime_error("dataset pointer was null");
    }
    return dataset->inner.class_names.size();
  }, 0);
}

extern "C" const char* rt_tensor_dataset_class_name(
    const RtTensorDataset* dataset,
    size_t class_index) {
  return protect<const char*>([&]() {
    if (dataset == nullptr) {
      throw std::runtime_error("dataset pointer was null");
    }
    if (class_index >= dataset->inner.class_names.size()) {
      throw std::runtime_error("class index out of range");
    }
    return dataset->inner.class_names[class_index].c_str();
  }, nullptr);
}

extern "C" RtDataLoader* rt_data_loader_new(
    RtTensorDataset* dataset,
    RtTransformPipeline* pipeline,
    size_t batch_size,
    bool shuffle) {
  return protect<RtDataLoader*>([&]() {
    if (dataset == nullptr) {
      throw std::runtime_error("dataset pointer was null");
    }
    if (batch_size == 0) {
      throw std::runtime_error("batch_size must be greater than zero");
    }

    auto loader = std::make_unique<RtDataLoader>();
    loader->inner.dataset = &dataset->inner;
    loader->inner.pipeline = pipeline == nullptr ? nullptr : &pipeline->inner;
    loader->inner.batch_size = batch_size;
    loader->inner.shuffle = shuffle;
    loader->inner.reset();
    return loader.release();
  }, nullptr);
}

extern "C" void rt_data_loader_free(RtDataLoader* loader) {
  delete loader;
}

extern "C" bool rt_data_loader_reset(RtDataLoader* loader) {
  return protect<bool>([&]() {
    if (loader == nullptr) {
      throw std::runtime_error("loader pointer was null");
    }
    loader->inner.reset();
    return true;
  }, false);
}

extern "C" int rt_data_loader_next(RtDataLoader* loader, RtOwnedBatch** out) {
  return protect<int>([&]() {
    if (loader == nullptr || out == nullptr) {
      throw std::runtime_error("loader or batch output pointer was null");
    }
    if (loader->inner.cursor >= loader->inner.indices.size()) {
      *out = nullptr;
      return RT_STATUS_END;
    }

    const auto remaining = loader->inner.indices.size() - loader->inner.cursor;
    const auto take = std::min(loader->inner.batch_size, remaining);

    std::vector<torch::Tensor> batch_tensors;
    std::vector<int64_t> labels;
    batch_tensors.reserve(take);
    labels.reserve(take);

    for (size_t i = 0; i < take; ++i) {
      const auto sample_index = loader->inner.indices[loader->inner.cursor + i];
      const auto& sample = loader->inner.dataset->samples[sample_index];
      auto tensor = load_tensor_from_file(sample.path);
      if (loader->inner.pipeline != nullptr) {
        tensor = loader->inner.pipeline->apply(tensor);
      }
      batch_tensors.push_back(tensor.contiguous());
      labels.push_back(sample.label);
    }

    auto stacked = torch::stack(batch_tensors);
    auto owned =
        stacked
            .to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32))
            .contiguous();
    auto batch = std::make_unique<RtOwnedBatch>();
    batch->inner.shape.assign(owned.sizes().begin(), owned.sizes().end());
    batch->inner.labels = std::move(labels);

    const auto count = static_cast<size_t>(owned.numel());
    batch->inner.statistics = calculate_statistics(owned);

    batch->inner.data.resize(count);
    std::memcpy(batch->inner.data.data(), owned.data_ptr<float>(), count * sizeof(float));

    loader->inner.cursor += take;
    *out = batch.release();
    return RT_STATUS_OK;
  }, RT_STATUS_ERR);
}

extern "C" const float* rt_owned_batch_data(const RtOwnedBatch* batch) {
  return batch == nullptr ? nullptr : batch->inner.data.data();
}

extern "C" size_t rt_owned_batch_data_len(const RtOwnedBatch* batch) {
  return batch == nullptr ? 0 : batch->inner.data.size();
}

extern "C" const int64_t* rt_owned_batch_shape(const RtOwnedBatch* batch) {
  return batch == nullptr ? nullptr : batch->inner.shape.data();
}

extern "C" size_t rt_owned_batch_shape_len(const RtOwnedBatch* batch) {
  return batch == nullptr ? 0 : batch->inner.shape.size();
}

extern "C" const int64_t* rt_owned_batch_labels(const RtOwnedBatch* batch) {
  return batch == nullptr ? nullptr : batch->inner.labels.data();
}

extern "C" size_t rt_owned_batch_labels_len(const RtOwnedBatch* batch) {
  return batch == nullptr ? 0 : batch->inner.labels.size();
}

extern "C" bool rt_owned_batch_statistics(
    const RtOwnedBatch* batch,
    RtStatistics* out) {
  if (batch == nullptr || out == nullptr) {
    g_last_error = "batch or statistics output pointer was null";
    return false;
  }
  *out = batch->inner.statistics;
  return true;
}

extern "C" void rt_owned_batch_free(RtOwnedBatch* batch) {
  if (batch == nullptr) {
    return;
  }
  delete batch;
}
