#include "pytorch_loader.h"

#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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
  std::vector<float> std;

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

    if (!mean.empty() || !std.empty()) {
      if (mean.size() != std.size()) {
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
          torch::tensor(std, torch::TensorOptions().dtype(torch::kFloat32)).view({-1, 1, 1});
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

}  // namespace

struct PtlTransformPipeline {
  TransformPipelineImpl inner;
};

struct PtlTensorDataset {
  TensorDatasetImpl inner;
};

struct PtlDataLoader {
  DataLoaderImpl inner;
};

struct PtlOwnedBatch {
  BatchStorage inner;
};

extern "C" const char* ptl_last_error_message(void) {
  return protect<const char*>([]() {
    char* message = static_cast<char*>(std::malloc(g_last_error.size() + 1));
    if (message == nullptr) {
      throw std::bad_alloc();
    }
    std::memcpy(message, g_last_error.c_str(), g_last_error.size() + 1);
    return message;
  }, nullptr);
}

extern "C" void ptl_last_error_free(char* message) {
  std::free(message);
}

extern "C" PtlTransformPipeline* ptl_transform_pipeline_new(
    bool to_float,
    int64_t resize_h,
    int64_t resize_w,
    int64_t crop_h,
    int64_t crop_w,
    const float* mean,
    size_t mean_len,
    const float* std,
    size_t std_len) {
  return protect<PtlTransformPipeline*>([&]() {
    auto pipeline = std::make_unique<PtlTransformPipeline>();
    pipeline->inner.to_float = to_float;
    pipeline->inner.has_resize = resize_h > 0 && resize_w > 0;
    pipeline->inner.has_crop = crop_h > 0 && crop_w > 0;
    pipeline->inner.resize_h = resize_h;
    pipeline->inner.resize_w = resize_w;
    pipeline->inner.crop_h = crop_h;
    pipeline->inner.crop_w = crop_w;
    pipeline->inner.mean.assign(mean, mean + mean_len);
    pipeline->inner.std.assign(std, std + std_len);
    return pipeline.release();
  }, nullptr);
}

extern "C" void ptl_transform_pipeline_free(PtlTransformPipeline* pipeline) {
  delete pipeline;
}

extern "C" PtlTensorDataset* ptl_tensor_dataset_new(const char* root) {
  return protect<PtlTensorDataset*>([&]() {
    if (root == nullptr) {
      throw std::runtime_error("dataset root pointer was null");
    }
    auto dataset = std::make_unique<PtlTensorDataset>();
    scan_dataset_root(dataset->inner, fs::path(root));
    return dataset.release();
  }, nullptr);
}

extern "C" void ptl_tensor_dataset_free(PtlTensorDataset* dataset) {
  delete dataset;
}

extern "C" size_t ptl_tensor_dataset_len(const PtlTensorDataset* dataset) {
  return protect<size_t>([&]() {
    if (dataset == nullptr) {
      throw std::runtime_error("dataset pointer was null");
    }
    return dataset->inner.samples.size();
  }, 0);
}

extern "C" size_t ptl_tensor_dataset_class_count(const PtlTensorDataset* dataset) {
  return protect<size_t>([&]() {
    if (dataset == nullptr) {
      throw std::runtime_error("dataset pointer was null");
    }
    return dataset->inner.class_names.size();
  }, 0);
}

extern "C" const char* ptl_tensor_dataset_class_name(
    const PtlTensorDataset* dataset,
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

extern "C" PtlDataLoader* ptl_data_loader_new(
    PtlTensorDataset* dataset,
    PtlTransformPipeline* pipeline,
    size_t batch_size,
    bool shuffle) {
  return protect<PtlDataLoader*>([&]() {
    if (dataset == nullptr) {
      throw std::runtime_error("dataset pointer was null");
    }
    if (batch_size == 0) {
      throw std::runtime_error("batch_size must be greater than zero");
    }

    auto loader = std::make_unique<PtlDataLoader>();
    loader->inner.dataset = &dataset->inner;
    loader->inner.pipeline = pipeline == nullptr ? nullptr : &pipeline->inner;
    loader->inner.batch_size = batch_size;
    loader->inner.shuffle = shuffle;
    loader->inner.reset();
    return loader.release();
  }, nullptr);
}

extern "C" void ptl_data_loader_free(PtlDataLoader* loader) {
  delete loader;
}

extern "C" bool ptl_data_loader_reset(PtlDataLoader* loader) {
  return protect<bool>([&]() {
    if (loader == nullptr) {
      throw std::runtime_error("loader pointer was null");
    }
    loader->inner.reset();
    return true;
  }, false);
}

extern "C" int ptl_data_loader_next(PtlDataLoader* loader, PtlOwnedBatch** out) {
  return protect<int>([&]() {
    if (loader == nullptr || out == nullptr) {
      throw std::runtime_error("loader or batch output pointer was null");
    }
    if (loader->inner.cursor >= loader->inner.indices.size()) {
      *out = nullptr;
      return PTL_STATUS_END;
    }

    const auto remaining = loader->inner.indices.size() - loader->inner.cursor;
    const auto take = std::min(loader->inner.batch_size, remaining);

    std::vector<torch::Tensor> batch_tensors;
    std::vector<int64_t> labels;
    batch_tensors.reserve(take);
    labels.reserve(take);

    for (size_t i = 0; i < take; ++i) {
      const auto sample_index = loader->inner.indices[loader->inner.cursor++];
      const auto& sample = loader->inner.dataset->samples[sample_index];
      auto tensor = load_tensor_from_file(sample.path);
      if (loader->inner.pipeline != nullptr) {
        tensor = loader->inner.pipeline->apply(tensor);
      }
      batch_tensors.push_back(tensor.contiguous());
      labels.push_back(sample.label);
    }

    auto stacked = torch::stack(batch_tensors).contiguous().to(torch::kFloat32);
    auto batch = std::make_unique<PtlOwnedBatch>();
    batch->inner.shape.assign(stacked.sizes().begin(), stacked.sizes().end());
    batch->inner.labels = std::move(labels);

    const auto count = static_cast<size_t>(stacked.numel());
    batch->inner.data.resize(count);
    std::memcpy(batch->inner.data.data(), stacked.data_ptr<float>(), count * sizeof(float));

    *out = batch.release();
    return PTL_STATUS_OK;
  }, PTL_STATUS_ERR);
}

extern "C" const float* ptl_owned_batch_data(const PtlOwnedBatch* batch) {
  return batch == nullptr ? nullptr : batch->inner.data.data();
}

extern "C" size_t ptl_owned_batch_data_len(const PtlOwnedBatch* batch) {
  return batch == nullptr ? 0 : batch->inner.data.size();
}

extern "C" const int64_t* ptl_owned_batch_shape(const PtlOwnedBatch* batch) {
  return batch == nullptr ? nullptr : batch->inner.shape.data();
}

extern "C" size_t ptl_owned_batch_shape_len(const PtlOwnedBatch* batch) {
  return batch == nullptr ? 0 : batch->inner.shape.size();
}

extern "C" const int64_t* ptl_owned_batch_labels(const PtlOwnedBatch* batch) {
  return batch == nullptr ? nullptr : batch->inner.labels.data();
}

extern "C" size_t ptl_owned_batch_labels_len(const PtlOwnedBatch* batch) {
  return batch == nullptr ? 0 : batch->inner.labels.size();
}

extern "C" void ptl_owned_batch_free(PtlOwnedBatch* batch) {
  if (batch == nullptr) {
    return;
  }
  delete batch;
}
