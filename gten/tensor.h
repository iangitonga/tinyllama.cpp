#pragma once

#include <memory>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "gten_types.h"
#include "log.h"
#include "quants.h"


namespace gten {

// TODO: make inside tensor.
static int64_t G_TensorMemAllocated = 0;


class Tensor {
public:
    Tensor() = default;
    Tensor(const std::vector<int>& shape, Dtype dtype);
    Tensor(const void* data_ptr, const std::vector<int>& shape, Dtype dtype);
    Tensor(const Tensor& rhs) = default;
    Tensor(Tensor&& rhs) = default;
    Tensor& operator=(const Tensor& rhs) = default;
    Tensor& operator=(Tensor&& rhs) = default;
    friend std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);
    Tensor permute(const std::vector<int>& new_shape);
    void print() const;
    void print_info() const;
    // Resize the tensor to have a new shape. This function does not perform
    // any reallocation and therefore, the tensor must have enough capacity
    // to accommodate the number of elements in the new shape.
    // NOTE: The purpose of this function is to allow us to allocate for
    // activations tensors to be able to hold all future predictions
    // activations but reshape them as we continously add activations.
    void resize(const std::vector<int>& new_shape);
    void set_strides(const std::vector<int>& strides);
    std::string shape_str() const;
    std::string strides_str() const;
    void save(const std::string& path) const;
    Tensor view(const std::vector<int>& new_shape) const;

    // Get the pointer to internal data buffer.
    template <typename T>
    T* data_ptr() { return reinterpret_cast<T*>(data_ptr_.get()); }

    template <typename T>
    const T* data_ptr() const { return reinterpret_cast<const T*>(data_ptr_.get()); }

    const void* data_ptr() const { return data_ptr_.get(); }
    void* data_ptr() { return data_ptr_.get(); }

    Dtype dtype() const { return dtype_; }

    // Get the number of bytes that an element in the tensor occupies.
    int itemsize() const {
        switch (dtype_) {
            case kQint8:
                return 1;
            case kInt32:
                return 4;
            case kFloat16:
                return 2;
            case kFloat32:
                return 4;
            default:
                GTEN_ASSERT(false);
                return 4;
        }
    }

    bool is_quantized() const  { return dtype_ == kQint8; }
    bool is_1d() const { return shape_.size() == 1; }
    bool is_2d() const { return shape_.size() == 2; }
    bool is_3d() const { return shape_.size() == 3; }
    int ndims() const { return shape_.size(); }

    // Get the number of elems in the tensor.
    int numel() const { return numel_; }

    /// Returns the size of the give dimension.
    int dimsize(int i) const {
        GTEN_ASSERT(i < int(shape_.size()));
        return shape_[i];
    }

    /// Returns the size of the give dimension.
    int stride(int i) const {
        GTEN_ASSERT(i < int(strides_.size()));
        return strides_[i];
    }

    /// Returns the size of the give dimension in bytes.
    int bstride(int i) const {
        GTEN_ASSERT(i < int(strides_.size()));

        switch (dtype_)
        {
            case kQint4: {
                if (strides_[i] == 1) {
                    return 1;
                }
                return (strides_[i]/globs::q4_block_size) * sizeof(Q4Block);
            }
            case kQint8: {
                if (strides_[i] == 1) {
                    return 1;
                }
                return (strides_[i]/globs::q8_block_size) * sizeof(Q8Block);
            }
            default:
                return strides_[i] * itemsize();
        }
    }

    size_t nbytes() const { return storage_size_; }

    const std::vector<int>& shape() const { return shape_; }

    bool shape_eq(const std::vector<int>& shape) const { return shape == shape_; }

private:
    Dtype dtype_ = kFloat32;
    std::shared_ptr<uint8_t> data_ptr_;
    int storage_size_ = 0;  // in_bytes
    int numel_ = 0;
    std::vector<int> shape_;
    std::vector<int> strides_;

    void validate_shape(const std::vector<int>& shape) const;
    void set_strides_from_shape(const std::vector<int>& shape);
    int numel_from_shape(const std::vector<int>& shape) const;
    void print_single(int item_idx, int row_idx, int col_idx, int n_cols) const;
};

} // Namespace xten
