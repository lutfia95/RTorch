#include "rtorch.h"

#include <torch/serialize.h>
#include <torch/torch.h>

#ifdef RTORCH_WITH_HDF5
#include <hdf5.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

enum class SampleFormat {
  TorchFile,
  Hdf5Dataset,
};

struct Hdf5DatasetState;

struct SampleRecord {
  fs::path path;
  int64_t label;
  SampleFormat format = SampleFormat::TorchFile;
  std::string hdf5_dataset;
  std::shared_ptr<Hdf5DatasetState> hdf5_state;
  uint64_t hdf5_offset = 0;
  uint64_t hdf5_length = 0;
  bool hdf5_has_slice = false;
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
  torch::Tensor tensor;
  torch::Tensor host_f32;
  std::vector<int64_t> shape;
  std::vector<int64_t> labels;
  std::optional<RtStatistics> statistics;
};

struct DataLoaderImpl {
  TensorDatasetImpl* dataset = nullptr;
  TransformPipelineImpl* pipeline = nullptr;
  size_t batch_size = 1;
  std::vector<size_t> indices;
  size_t cursor = 0;
  bool shuffle = false;
  bool drop_last = false;
  std::mt19937 rng{std::random_device{}()};

  size_t len() const {
    if (batch_size == 0) {
      return 0;
    }
    const auto sample_count = indices.empty() && dataset != nullptr
        ? dataset->samples.size()
        : indices.size();
    if (drop_last) {
      return sample_count / batch_size;
    }
    return (sample_count + batch_size - 1) / batch_size;
  }

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

bool is_hdf5_path(const fs::path& path) {
  const auto extension = path.extension().string();
  return extension == ".h5" || extension == ".hdf5";
}

bool is_tensor_data_path(const fs::path& path) {
  return path.extension() == ".pt" || is_hdf5_path(path);
}

#ifdef RTORCH_WITH_HDF5
struct Hdf5Handle {
  hid_t id = -1;
  herr_t (*close_fn)(hid_t) = nullptr;

  Hdf5Handle() = default;
  Hdf5Handle(hid_t id, herr_t (*close_fn)(hid_t)) : id(id), close_fn(close_fn) {}
  Hdf5Handle(const Hdf5Handle&) = delete;
  Hdf5Handle& operator=(const Hdf5Handle&) = delete;
  Hdf5Handle(Hdf5Handle&& other) noexcept : id(other.id), close_fn(other.close_fn) {
    other.id = -1;
    other.close_fn = nullptr;
  }
  Hdf5Handle& operator=(Hdf5Handle&& other) noexcept {
    if (this != &other) {
      close();
      id = other.id;
      close_fn = other.close_fn;
      other.id = -1;
      other.close_fn = nullptr;
    }
    return *this;
  }
  ~Hdf5Handle() {
    close();
  }

  void close() {
    if (id >= 0 && close_fn != nullptr) {
      close_fn(id);
    }
    id = -1;
    close_fn = nullptr;
  }

  operator hid_t() const {
    return id;
  }
};

struct Hdf5DatasetState {
  fs::path path;
  std::string dataset_name;
  Hdf5Handle file;
  Hdf5Handle dataset;
};

struct Hdf5StructureSpan {
  int64_t offset;
  int32_t length;
};

bool hdf5_dataset_exists(hid_t file, const std::string& dataset_name) {
  return H5Lexists(file, dataset_name.c_str(), H5P_DEFAULT) > 0;
}

std::string resolve_hdf5_dataset_name(hid_t file, const std::string& requested) {
  if (!requested.empty()) {
    if (!hdf5_dataset_exists(file, requested)) {
      throw std::runtime_error("HDF5 dataset does not exist: " + requested);
    }
    return requested;
  }

  const std::vector<std::string> candidates{"tensor", "data", "positions"};
  for (const auto& candidate : candidates) {
    if (hdf5_dataset_exists(file, candidate)) {
      return candidate;
    }
  }

  throw std::runtime_error(
      "HDF5 file does not contain a default tensor dataset; expected tensor, data, or positions");
}

Hdf5Handle open_hdf5_file(const fs::path& path) {
  auto file = H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    throw std::runtime_error("failed to open HDF5 file: " + path.string());
  }
  return Hdf5Handle(file, H5Fclose);
}

std::vector<hsize_t> hdf5_dataset_dims(hid_t dataset) {
  Hdf5Handle space(H5Dget_space(dataset), H5Sclose);
  if (space.id < 0) {
    throw std::runtime_error("failed to open HDF5 dataset dataspace");
  }
  const auto rank = H5Sget_simple_extent_ndims(space);
  if (rank < 0) {
    throw std::runtime_error("failed to inspect HDF5 dataset rank");
  }

  std::vector<hsize_t> dims(static_cast<size_t>(rank));
  if (rank > 0 && H5Sget_simple_extent_dims(space, dims.data(), nullptr) < 0) {
    throw std::runtime_error("failed to inspect HDF5 dataset shape");
  }
  return dims;
}

std::vector<Hdf5StructureSpan> read_hdf5_structure_spans(hid_t file) {
  if (!hdf5_dataset_exists(file, "structures")) {
    return {};
  }

  Hdf5Handle dataset(H5Dopen2(file, "structures", H5P_DEFAULT), H5Dclose);
  if (dataset.id < 0) {
    throw std::runtime_error("failed to open HDF5 structures dataset");
  }

  auto dims = hdf5_dataset_dims(dataset);
  if (dims.size() != 1) {
    throw std::runtime_error("HDF5 structures dataset must be one-dimensional");
  }

  Hdf5Handle memory_type(
      H5Tcreate(H5T_COMPOUND, sizeof(Hdf5StructureSpan)),
      H5Tclose);
  if (memory_type.id < 0 ||
      H5Tinsert(memory_type, "offset", HOFFSET(Hdf5StructureSpan, offset), H5T_NATIVE_LLONG) < 0 ||
      H5Tinsert(memory_type, "length", HOFFSET(Hdf5StructureSpan, length), H5T_NATIVE_INT) < 0) {
    throw std::runtime_error("failed to create HDF5 structures memory type");
  }

  std::vector<Hdf5StructureSpan> spans(static_cast<size_t>(dims[0]));
  if (H5Dread(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, spans.data()) < 0) {
    throw std::runtime_error("failed to read HDF5 structures offsets");
  }
  return spans;
}

size_t hdf5_element_count(const std::vector<hsize_t>& dims) {
  size_t count = 1;
  for (const auto dim : dims) {
    if (dim > static_cast<hsize_t>(std::numeric_limits<size_t>::max())) {
      throw std::runtime_error("HDF5 dataset dimension exceeds size_t");
    }
    const auto value = static_cast<size_t>(dim);
    if (value != 0 && count > std::numeric_limits<size_t>::max() / value) {
      throw std::runtime_error("HDF5 dataset element count overflowed size_t");
    }
    count *= value;
  }
  return count;
}

std::vector<int64_t> hdf5_torch_shape(const std::vector<hsize_t>& dims) {
  std::vector<int64_t> shape;
  shape.reserve(dims.size());
  for (const auto dim : dims) {
    if (dim > static_cast<hsize_t>(std::numeric_limits<int64_t>::max())) {
      throw std::runtime_error("HDF5 dataset dimension exceeds int64_t");
    }
    shape.push_back(static_cast<int64_t>(dim));
  }
  return shape;
}

template <typename T>
torch::Tensor read_hdf5_values(
    hid_t dataset,
    hid_t native_type,
    c10::ScalarType torch_type,
    const SampleRecord& sample) {
  auto file_dims = hdf5_dataset_dims(dataset);
  if (sample.hdf5_has_slice && file_dims.empty()) {
    throw std::runtime_error("cannot slice a scalar HDF5 dataset");
  }

  std::vector<hsize_t> read_dims = file_dims;
  Hdf5Handle file_space(H5Dget_space(dataset), H5Sclose);
  Hdf5Handle memory_space;

  if (sample.hdf5_has_slice) {
    const auto offset = static_cast<hsize_t>(sample.hdf5_offset);
    const auto length = static_cast<hsize_t>(sample.hdf5_length);
    if (offset > file_dims[0] || length > file_dims[0] - offset) {
      throw std::runtime_error("HDF5 slice exceeds dataset bounds");
    }

    std::vector<hsize_t> starts(file_dims.size(), 0);
    read_dims[0] = length;
    starts[0] = offset;

    if (H5Sselect_hyperslab(
            file_space,
            H5S_SELECT_SET,
            starts.data(),
            nullptr,
            read_dims.data(),
            nullptr) < 0) {
      throw std::runtime_error("failed to select HDF5 hyperslab");
    }

    memory_space = Hdf5Handle(
        H5Screate_simple(static_cast<int>(read_dims.size()), read_dims.data(), nullptr),
        H5Sclose);
    if (memory_space.id < 0) {
      throw std::runtime_error("failed to create HDF5 memory dataspace");
    }
  } else {
    memory_space = Hdf5Handle(H5S_ALL, nullptr);
  }

  auto shape = hdf5_torch_shape(read_dims);
  auto count = hdf5_element_count(read_dims);
  std::vector<T> values(count);
  auto memory_id = sample.hdf5_has_slice ? memory_space.id : H5S_ALL;
  auto file_id = sample.hdf5_has_slice ? file_space.id : H5S_ALL;
  if (H5Dread(dataset, native_type, memory_id, file_id, H5P_DEFAULT, values.data()) < 0) {
    throw std::runtime_error("failed to read HDF5 tensor data");
  }

  return torch::from_blob(
             values.data(),
             shape,
             torch::TensorOptions().dtype(torch_type))
      .clone();
}

torch::Tensor read_hdf5_tensor(const SampleRecord& sample) {
  if (sample.hdf5_state == nullptr || sample.hdf5_state->dataset.id < 0) {
    throw std::runtime_error("HDF5 sample has no open dataset");
  }
  const auto dataset = sample.hdf5_state->dataset.id;
  const auto& dataset_name = sample.hdf5_state->dataset_name;

  Hdf5Handle type(H5Dget_type(dataset), H5Tclose);
  if (type.id < 0) {
    throw std::runtime_error("failed to inspect HDF5 dataset type: " + dataset_name);
  }
  const auto type_class = H5Tget_class(type);
  const auto size = H5Tget_size(type);
  if (type_class == H5T_FLOAT) {
    if (size <= sizeof(float)) {
      return read_hdf5_values<float>(dataset, H5T_NATIVE_FLOAT, torch::kFloat32, sample);
    }
    return read_hdf5_values<double>(dataset, H5T_NATIVE_DOUBLE, torch::kFloat64, sample);
  }

  if (type_class == H5T_INTEGER) {
    const auto sign = H5Tget_sign(type);
    if (sign == H5T_SGN_NONE) {
      if (size == sizeof(uint8_t)) {
        return read_hdf5_values<uint8_t>(dataset, H5T_NATIVE_UCHAR, torch::kUInt8, sample);
      }
#ifdef RTORCH_HAS_EXTENDED_UNSIGNED_TYPES
      if (size <= sizeof(uint16_t)) {
        return read_hdf5_values<uint16_t>(dataset, H5T_NATIVE_USHORT, c10::kUInt16, sample);
      }
      if (size <= sizeof(uint32_t)) {
        return read_hdf5_values<uint32_t>(dataset, H5T_NATIVE_UINT, c10::kUInt32, sample);
      }
      return read_hdf5_values<uint64_t>(dataset, H5T_NATIVE_ULLONG, c10::kUInt64, sample);
#else
      if (size >= sizeof(uint64_t)) {
        throw std::runtime_error(
            "this LibTorch version cannot represent 64-bit unsigned HDF5 values");
      }
      return read_hdf5_values<int64_t>(dataset, H5T_NATIVE_LLONG, torch::kInt64, sample);
#endif
    }
    if (size <= sizeof(int8_t)) {
      return read_hdf5_values<int8_t>(dataset, H5T_NATIVE_SCHAR, torch::kInt8, sample);
    }
    if (size <= sizeof(int16_t)) {
      return read_hdf5_values<int16_t>(dataset, H5T_NATIVE_SHORT, torch::kInt16, sample);
    }
    if (size <= sizeof(int32_t)) {
      return read_hdf5_values<int32_t>(dataset, H5T_NATIVE_INT, torch::kInt32, sample);
    }
    return read_hdf5_values<int64_t>(dataset, H5T_NATIVE_LLONG, torch::kInt64, sample);
  }

  throw std::runtime_error("HDF5 dataset type is not numeric");
}

void append_hdf5_samples(
    TensorDatasetImpl& dataset,
    const fs::path& path,
    int64_t label,
    const std::string& dataset_name) {
  auto state = std::make_shared<Hdf5DatasetState>();
  state->path = path;
  state->file = open_hdf5_file(path);
  state->dataset_name =
      resolve_hdf5_dataset_name(state->file, dataset_name);
  state->dataset = Hdf5Handle(
      H5Dopen2(state->file, state->dataset_name.c_str(), H5P_DEFAULT),
      H5Dclose);
  if (state->dataset.id < 0) {
    throw std::runtime_error(
        "failed to open HDF5 dataset: " + state->dataset_name);
  }
  auto value_dims = hdf5_dataset_dims(state->dataset);
  auto spans = read_hdf5_structure_spans(state->file);

  if (!spans.empty() && !value_dims.empty()) {
    for (const auto& span : spans) {
      if (span.offset < 0 || span.length < 0) {
        throw std::runtime_error("HDF5 structures contain negative offset or length");
      }
      SampleRecord record;
      record.path = path;
      record.label = label;
      record.format = SampleFormat::Hdf5Dataset;
      record.hdf5_dataset = state->dataset_name;
      record.hdf5_state = state;
      record.hdf5_offset = static_cast<uint64_t>(span.offset);
      record.hdf5_length = static_cast<uint64_t>(span.length);
      record.hdf5_has_slice = true;
      dataset.samples.push_back(std::move(record));
    }
    return;
  }

  SampleRecord record;
  record.path = path;
  record.label = label;
  record.format = SampleFormat::Hdf5Dataset;
  record.hdf5_dataset = state->dataset_name;
  record.hdf5_state = std::move(state);
  dataset.samples.push_back(std::move(record));
}
#else
void append_hdf5_samples(
    TensorDatasetImpl&,
    const fs::path&,
    int64_t,
    const std::string&) {
  throw std::runtime_error(
      "RTorch was built without HDF5 support; enable the hdf5 Cargo feature and rebuild");
}

torch::Tensor read_hdf5_tensor(const SampleRecord&) {
  throw std::runtime_error(
      "RTorch was built without HDF5 support; enable the hdf5 Cargo feature and rebuild");
}
#endif

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
      if (entry.is_regular_file() && is_tensor_data_path(entry.path())) {
        files.push_back(entry.path());
      }
    }
    std::sort(files.begin(), files.end());

    for (const auto& path : files) {
      if (is_hdf5_path(path)) {
        append_hdf5_samples(dataset, path, static_cast<int64_t>(class_index), "");
      } else {
        SampleRecord record;
        record.path = path;
        record.label = static_cast<int64_t>(class_index);
        record.format = SampleFormat::TorchFile;
        dataset.samples.push_back(std::move(record));
      }
    }
  }

  if (dataset.samples.empty()) {
    throw std::runtime_error("dataset contains no .pt, .h5, or .hdf5 files");
  }
}

std::vector<char> read_file_bytes(const fs::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("failed to open tensor file: " + path.string());
  }
  const auto end = input.tellg();
  if (end < 0) {
    throw std::runtime_error("failed to determine tensor file size: " + path.string());
  }
  const auto unsigned_size = static_cast<uintmax_t>(end);
  if (unsigned_size > static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error("tensor file is too large to read: " + path.string());
  }
  if (unsigned_size >
      static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("tensor file exceeds stream size limits: " + path.string());
  }
  std::vector<char> bytes(static_cast<size_t>(unsigned_size));
  input.seekg(0, std::ios::beg);
  if (!bytes.empty() &&
      !input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("failed to read tensor file: " + path.string());
  }
  return bytes;
}

torch::Tensor load_python_pickled_tensor(const fs::path& path) {
  auto value = torch::pickle_load(read_file_bytes(path));
  if (!value.isTensor()) {
    throw std::runtime_error(
        "Python torch.save file does not contain a single tensor");
  }
  return value.toTensor();
}

torch::Tensor load_tensor_from_sample(const SampleRecord& sample) {
  if (sample.format == SampleFormat::Hdf5Dataset) {
    return read_hdf5_tensor(sample);
  }

  std::string cpp_archive_error;
  try {
    torch::Tensor tensor;
    torch::load(tensor, sample.path.string());
    return tensor;
  } catch (const c10::Error& error) {
    cpp_archive_error = error.what_without_backtrace();
  } catch (const std::exception& error) {
    cpp_archive_error = error.what();
  }

  try {
    return load_python_pickled_tensor(sample.path);
  } catch (const c10::Error& error) {
    throw std::runtime_error(
        "failed to load .pt tensor as a LibTorch C++ archive (" +
        cpp_archive_error + ") or Python torch.save file (" +
        error.what_without_backtrace() + ")");
  } catch (const std::exception& error) {
    throw std::runtime_error(
        "failed to load .pt tensor as a LibTorch C++ archive (" +
        cpp_archive_error + ") or Python torch.save file (" + error.what() +
        ")");
  }
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
#ifdef RTORCH_HAS_EXTENDED_UNSIGNED_TYPES
    case RT_SCALAR_U16:
      return c10::kUInt16;
    case RT_SCALAR_U32:
      return c10::kUInt32;
    case RT_SCALAR_U64:
      return c10::kUInt64;
#endif
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

int scalar_type_to_ffi(c10::ScalarType scalar_type) {
  switch (scalar_type) {
    case c10::kBool:
      return RT_SCALAR_BOOL;
    case c10::kByte:
      return RT_SCALAR_U8;
    case c10::kChar:
      return RT_SCALAR_I8;
    case c10::kShort:
      return RT_SCALAR_I16;
    case c10::kInt:
      return RT_SCALAR_I32;
    case c10::kLong:
      return RT_SCALAR_I64;
#ifdef RTORCH_HAS_EXTENDED_UNSIGNED_TYPES
    case c10::kUInt16:
      return RT_SCALAR_U16;
    case c10::kUInt32:
      return RT_SCALAR_U32;
    case c10::kUInt64:
      return RT_SCALAR_U64;
#endif
    case c10::kHalf:
      return RT_SCALAR_F16;
    case c10::kBFloat16:
      return RT_SCALAR_BF16;
    case c10::kFloat:
      return RT_SCALAR_F32;
    case c10::kDouble:
      return RT_SCALAR_F64;
    default:
      throw std::runtime_error("unsupported tensor scalar type");
  }
}

std::vector<int64_t> copy_shape(const int64_t* shape, size_t shape_len) {
  if (shape_len == 0) {
    return {};
  }
  if (shape == nullptr) {
    throw std::runtime_error("tensor shape pointer was null");
  }
  std::vector<int64_t> result(shape, shape + shape_len);
  if (std::any_of(
          result.begin(),
          result.end(),
          [](int64_t value) { return value < 0; })) {
    throw std::runtime_error("tensor shape dimensions must be non-negative");
  }
  return result;
}

size_t shape_numel(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (const auto dimension : shape) {
    const auto value = static_cast<size_t>(dimension);
    if (value != 0 && count > std::numeric_limits<size_t>::max() / value) {
      throw std::runtime_error("tensor shape element count overflowed");
    }
    count *= value;
  }
  return count;
}

template <typename T>
torch::Tensor unsigned_statistics_tensor(const void* data, size_t len) {
  const auto* source = static_cast<const T*>(data);
  std::vector<double> converted(source, source + len);
  return torch::from_blob(
             converted.data(),
             {static_cast<int64_t>(len)},
             torch::TensorOptions().dtype(torch::kFloat64))
      .clone();
}

torch::Tensor statistics_tensor_from_data(
    const void* data,
    size_t len,
    int scalar_type) {
#ifndef RTORCH_HAS_EXTENDED_UNSIGNED_TYPES
  switch (scalar_type) {
    case RT_SCALAR_U16:
      return unsigned_statistics_tensor<uint16_t>(data, len);
    case RT_SCALAR_U32:
      return unsigned_statistics_tensor<uint32_t>(data, len);
    case RT_SCALAR_U64:
      return unsigned_statistics_tensor<uint64_t>(data, len);
    default:
      break;
  }
#endif
  return torch::from_blob(
      const_cast<void*>(data),
      {static_cast<int64_t>(len)},
      torch::TensorOptions().dtype(scalar_type_from_ffi(scalar_type)));
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

struct RtTensor {
  torch::Tensor inner;
};

std::unique_ptr<RtOwnedBatch> make_owned_batch(
    const torch::Tensor& tensor,
    std::vector<int64_t> labels) {
  auto batch = std::make_unique<RtOwnedBatch>();
  batch->inner.tensor = tensor.contiguous();
  batch->inner.shape.assign(
      batch->inner.tensor.sizes().begin(),
      batch->inner.tensor.sizes().end());
  batch->inner.labels = std::move(labels);
  return batch;
}

void materialize_host_f32(BatchStorage& batch) {
  if (!batch.host_f32.defined()) {
    batch.host_f32 =
        batch.tensor
            .detach()
            .to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32))
            .contiguous();
  }
}

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

    auto values = statistics_tensor_from_data(data, len, scalar_type);
    *out = calculate_statistics(values);
    return true;
  }, false);
}

extern "C" RtTensor* rt_tensor_from_data(
    const void* data,
    size_t len,
    int scalar_type,
    const int64_t* shape,
    size_t shape_len,
    bool requires_grad) {
  return protect<RtTensor*>([&]() {
    const auto dimensions = copy_shape(shape, shape_len);
    if (shape_numel(dimensions) != len) {
      throw std::runtime_error("tensor data length does not match shape");
    }
    if (len > 0 && data == nullptr) {
      throw std::runtime_error("tensor data pointer was null");
    }

    const auto options =
        torch::TensorOptions().dtype(scalar_type_from_ffi(scalar_type));
    torch::Tensor tensor;
    if (len == 0) {
      tensor = torch::empty(dimensions, options);
    } else {
      tensor =
          torch::from_blob(const_cast<void*>(data), dimensions, options).clone();
    }
    tensor.set_requires_grad(requires_grad);

    auto result = std::make_unique<RtTensor>();
    result->inner = std::move(tensor);
    return result.release();
  }, nullptr);
}

extern "C" RtTensor* rt_tensor_zeros(
    const int64_t* shape,
    size_t shape_len,
    int scalar_type,
    bool requires_grad) {
  return protect<RtTensor*>([&]() {
    const auto dimensions = copy_shape(shape, shape_len);
    auto result = std::make_unique<RtTensor>();
    result->inner = torch::zeros(
        dimensions,
        torch::TensorOptions()
            .dtype(scalar_type_from_ffi(scalar_type))
            .requires_grad(requires_grad));
    return result.release();
  }, nullptr);
}

extern "C" RtTensor* rt_tensor_ones(
    const int64_t* shape,
    size_t shape_len,
    int scalar_type,
    bool requires_grad) {
  return protect<RtTensor*>([&]() {
    const auto dimensions = copy_shape(shape, shape_len);
    auto result = std::make_unique<RtTensor>();
    result->inner = torch::ones(
        dimensions,
        torch::TensorOptions()
            .dtype(scalar_type_from_ffi(scalar_type))
            .requires_grad(requires_grad));
    return result.release();
  }, nullptr);
}

extern "C" RtTensor* rt_tensor_clone(const RtTensor* tensor) {
  return protect<RtTensor*>([&]() {
    if (tensor == nullptr) {
      throw std::runtime_error("tensor pointer was null");
    }
    auto result = std::make_unique<RtTensor>();
    result->inner = tensor->inner;
    return result.release();
  }, nullptr);
}

extern "C" void rt_tensor_free(RtTensor* tensor) {
  delete tensor;
}

extern "C" size_t rt_tensor_numel(const RtTensor* tensor) {
  return protect<size_t>([&]() {
    if (tensor == nullptr) {
      throw std::runtime_error("tensor pointer was null");
    }
    return static_cast<size_t>(tensor->inner.numel());
  }, 0);
}

extern "C" size_t rt_tensor_shape_len(const RtTensor* tensor) {
  return protect<size_t>([&]() {
    if (tensor == nullptr) {
      throw std::runtime_error("tensor pointer was null");
    }
    return static_cast<size_t>(tensor->inner.dim());
  }, 0);
}

extern "C" bool rt_tensor_copy_shape(
    const RtTensor* tensor,
    int64_t* out,
    size_t out_len) {
  return protect<bool>([&]() {
    if (tensor == nullptr) {
      throw std::runtime_error("tensor pointer was null");
    }
    const auto sizes = tensor->inner.sizes();
    if (out_len != static_cast<size_t>(sizes.size())) {
      throw std::runtime_error("tensor shape output length mismatch");
    }
    if (out_len > 0 && out == nullptr) {
      throw std::runtime_error("tensor shape output pointer was null");
    }
    if (out_len > 0) {
      std::copy(sizes.begin(), sizes.end(), out);
    }
    return true;
  }, false);
}

extern "C" int rt_tensor_scalar_type(const RtTensor* tensor) {
  return protect<int>([&]() {
    if (tensor == nullptr) {
      throw std::runtime_error("tensor pointer was null");
    }
    return scalar_type_to_ffi(tensor->inner.scalar_type());
  }, -1);
}

extern "C" bool rt_tensor_requires_grad(const RtTensor* tensor) {
  return tensor != nullptr && tensor->inner.requires_grad();
}

extern "C" bool rt_tensor_set_requires_grad(
    RtTensor* tensor,
    bool requires_grad) {
  return protect<bool>([&]() {
    if (tensor == nullptr) {
      throw std::runtime_error("tensor pointer was null");
    }
    tensor->inner.set_requires_grad(requires_grad);
    return true;
  }, false);
}

extern "C" bool rt_tensor_copy_f32(
    const RtTensor* tensor,
    float* out,
    size_t out_len) {
  return protect<bool>([&]() {
    if (tensor == nullptr) {
      throw std::runtime_error("tensor pointer was null");
    }
    const auto source =
        tensor->inner
            .detach()
            .to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32))
            .contiguous();
    const auto count = static_cast<size_t>(source.numel());
    if (out_len != count) {
      throw std::runtime_error("tensor data output length mismatch");
    }
    if (count > 0 && out == nullptr) {
      throw std::runtime_error("tensor data output pointer was null");
    }
    if (count > 0) {
      std::memcpy(out, source.data_ptr<float>(), count * sizeof(float));
    }
    return true;
  }, false);
}

extern "C" RtTensor* rt_tensor_add(
    const RtTensor* left,
    const RtTensor* right) {
  return protect<RtTensor*>([&]() {
    if (left == nullptr || right == nullptr) {
      throw std::runtime_error("tensor pointer was null");
    }
    auto result = std::make_unique<RtTensor>();
    result->inner = left->inner + right->inner;
    return result.release();
  }, nullptr);
}

extern "C" RtTensor* rt_tensor_matmul(
    const RtTensor* left,
    const RtTensor* right) {
  return protect<RtTensor*>([&]() {
    if (left == nullptr || right == nullptr) {
      throw std::runtime_error("tensor pointer was null");
    }
    auto result = std::make_unique<RtTensor>();
    result->inner = torch::matmul(left->inner, right->inner);
    return result.release();
  }, nullptr);
}

extern "C" RtTensor* rt_tensor_reshape(
    const RtTensor* tensor,
    const int64_t* shape,
    size_t shape_len) {
  return protect<RtTensor*>([&]() {
    if (tensor == nullptr) {
      throw std::runtime_error("tensor pointer was null");
    }
    const auto dimensions = copy_shape(shape, shape_len);
    auto result = std::make_unique<RtTensor>();
    result->inner = tensor->inner.reshape(dimensions);
    return result.release();
  }, nullptr);
}

extern "C" RtTensor* rt_tensor_sum(const RtTensor* tensor) {
  return protect<RtTensor*>([&]() {
    if (tensor == nullptr) {
      throw std::runtime_error("tensor pointer was null");
    }
    auto result = std::make_unique<RtTensor>();
    result->inner = tensor->inner.sum();
    return result.release();
  }, nullptr);
}

extern "C" bool rt_tensor_backward(RtTensor* tensor) {
  return protect<bool>([&]() {
    if (tensor == nullptr) {
      throw std::runtime_error("tensor pointer was null");
    }
    tensor->inner.backward();
    return true;
  }, false);
}

extern "C" RtTensor* rt_tensor_grad(const RtTensor* tensor) {
  return protect<RtTensor*>([&]() {
    if (tensor == nullptr) {
      throw std::runtime_error("tensor pointer was null");
    }
    const auto gradient = tensor->inner.grad();
    if (!gradient.defined()) {
      throw std::runtime_error("tensor gradient is not defined");
    }
    auto result = std::make_unique<RtTensor>();
    result->inner = gradient.clone();
    return result.release();
  }, nullptr);
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
    const auto valid_optional_size = [](int64_t height, int64_t width) {
      return (height == -1 && width == -1) || (height > 0 && width > 0);
    };
    if (!valid_optional_size(resize_h, resize_w)) {
      throw std::runtime_error(
          "resize dimensions must both be positive when configured");
    }
    if (!valid_optional_size(crop_h, crop_w)) {
      throw std::runtime_error(
          "center_crop dimensions must both be positive when configured");
    }
    if ((mean_len == 0) != (std_values_len == 0)) {
      throw std::runtime_error(
          "normalize mean and std must either both be set or both be omitted");
    }
    if (mean_len != std_values_len) {
      throw std::runtime_error("normalize mean/std length mismatch");
    }
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
            [](float value) { return !std::isfinite(value) || value <= 0.0F; })) {
      throw std::runtime_error(
          "normalize std values must be finite and greater than zero");
    }
    if (std::any_of(
            pipeline->inner.mean.begin(),
            pipeline->inner.mean.end(),
            [](float value) { return !std::isfinite(value); })) {
      throw std::runtime_error("normalize mean values must be finite");
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

extern "C" RtTensorDataset* rt_tensor_dataset_hdf5_file_new(
    const char* path,
    const char* dataset_name) {
  return protect<RtTensorDataset*>([&]() {
    if (path == nullptr || dataset_name == nullptr) {
      throw std::runtime_error("HDF5 path or dataset name was null");
    }
#ifndef RTORCH_WITH_HDF5
    throw std::runtime_error(
        "RTorch was built without HDF5 support; enable the hdf5 Cargo feature and rebuild");
    return static_cast<RtTensorDataset*>(nullptr);
#else
    auto dataset = std::make_unique<RtTensorDataset>();
    const fs::path hdf5_path(path);
    dataset->inner.class_names.push_back(hdf5_path.stem().string());
    append_hdf5_samples(dataset->inner, hdf5_path, 0, dataset_name);
    return dataset.release();
#endif
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

extern "C" RtOwnedBatch* rt_tensor_dataset_get(
    const RtTensorDataset* dataset,
    size_t index,
    RtTransformPipeline* pipeline) {
  return protect<RtOwnedBatch*>([&]() {
    if (dataset == nullptr) {
      throw std::runtime_error("dataset pointer was null");
    }
    if (index >= dataset->inner.samples.size()) {
      throw std::runtime_error("dataset index out of range");
    }

    const auto& sample = dataset->inner.samples[index];
    auto tensor = load_tensor_from_sample(sample);
    if (pipeline != nullptr) {
      tensor = pipeline->inner.apply(tensor);
    }
    return make_owned_batch(tensor, std::vector<int64_t>{sample.label}).release();
  }, nullptr);
}

extern "C" RtDataLoader* rt_data_loader_new(
    RtTensorDataset* dataset,
    RtTransformPipeline* pipeline,
    size_t batch_size,
    bool shuffle) {
  return rt_data_loader_new_with_options(dataset, pipeline, batch_size, shuffle, false);
}

extern "C" RtDataLoader* rt_data_loader_new_with_options(
    RtTensorDataset* dataset,
    RtTransformPipeline* pipeline,
    size_t batch_size,
    bool shuffle,
    bool drop_last) {
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
    loader->inner.drop_last = drop_last;
    loader->inner.reset();
    return loader.release();
  }, nullptr);
}

extern "C" void rt_data_loader_free(RtDataLoader* loader) {
  delete loader;
}

extern "C" size_t rt_data_loader_len(const RtDataLoader* loader) {
  return protect<size_t>([&]() {
    if (loader == nullptr) {
      throw std::runtime_error("loader pointer was null");
    }
    return loader->inner.len();
  }, 0);
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
  const auto status = protect<int>([&]() {
    if (loader == nullptr || out == nullptr) {
      throw std::runtime_error("loader or batch output pointer was null");
    }
    if (loader->inner.cursor >= loader->inner.indices.size()) {
      *out = nullptr;
      return RT_STATUS_END;
    }

    const auto remaining = loader->inner.indices.size() - loader->inner.cursor;
    if (loader->inner.drop_last && remaining < loader->inner.batch_size) {
      loader->inner.cursor = loader->inner.indices.size();
      *out = nullptr;
      return RT_STATUS_END;
    }
    const auto take = std::min(loader->inner.batch_size, remaining);

    std::vector<torch::Tensor> batch_tensors;
    std::vector<int64_t> labels;
    batch_tensors.reserve(take);
    labels.reserve(take);

    for (size_t i = 0; i < take; ++i) {
      const auto sample_index = loader->inner.indices[loader->inner.cursor + i];
      const auto& sample = loader->inner.dataset->samples[sample_index];
      auto tensor = load_tensor_from_sample(sample);
      if (loader->inner.pipeline != nullptr) {
        tensor = loader->inner.pipeline->apply(tensor);
      }
      batch_tensors.push_back(tensor.contiguous());
      labels.push_back(sample.label);
    }

    auto batch = make_owned_batch(torch::stack(batch_tensors), std::move(labels));

    loader->inner.cursor += take;
    *out = batch.release();
    return RT_STATUS_OK;
  }, RT_STATUS_ERR);
  if (status == RT_STATUS_ERR && loader != nullptr) {
    loader->inner.cursor = loader->inner.indices.size();
    if (out != nullptr) {
      *out = nullptr;
    }
  }
  return status;
}

extern "C" bool rt_owned_batch_data_view(
    RtOwnedBatch* batch,
    const float** out_data,
    size_t* out_len) {
  return protect<bool>([&]() {
    if (batch == nullptr || out_data == nullptr || out_len == nullptr) {
      throw std::runtime_error("batch data view pointer was null");
    }
    materialize_host_f32(batch->inner);
    const auto count = static_cast<size_t>(batch->inner.host_f32.numel());
    *out_data = count == 0 ? nullptr : batch->inner.host_f32.data_ptr<float>();
    *out_len = count;
    return true;
  }, false);
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
    RtOwnedBatch* batch,
    RtStatistics* out) {
  return protect<bool>([&]() {
    if (batch == nullptr || out == nullptr) {
      throw std::runtime_error("batch or statistics output pointer was null");
    }
    if (!batch->inner.statistics.has_value()) {
      batch->inner.statistics = calculate_statistics(batch->inner.tensor);
    }
    *out = *batch->inner.statistics;
    return true;
  }, false);
}

extern "C" RtTensor* rt_owned_batch_tensor(const RtOwnedBatch* batch) {
  return protect<RtTensor*>([&]() {
    if (batch == nullptr) {
      throw std::runtime_error("batch pointer was null");
    }
    auto tensor = std::make_unique<RtTensor>();
    tensor->inner = batch->inner.tensor;
    return tensor.release();
  }, nullptr);
}

extern "C" void rt_owned_batch_free(RtOwnedBatch* batch) {
  if (batch == nullptr) {
    return;
  }
  delete batch;
}
