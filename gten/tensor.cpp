#include <iostream>
#include <iomanip>
#include <fstream>

#include "tensor.h"
#include "quants.h"


namespace gten {

/*

Types of tensors:
- Weight tensor:
  ~ Allocate -> fill. Always static. 
- Activation tensor:
  ~ Allocate max size required for inference.
  ~ initial numel=0, shape=()
  ~ compute; update [numel, shape, strides]
  ~ compute with offsets.
*/

static void tensor_data_deleter(uint8_t* ptr) {
    std::free(ptr);
}

Tensor::Tensor(const std::vector<int>& shape, Dtype dtype)
    : dtype_{dtype}
{
    validate_shape(shape);
    shape_ = shape;
    set_strides_from_shape(shape);
    const int numel = numel_from_shape(shape);
    numel_ = numel;

    int alloc_bytes;
    if (dtype == kQint8 && shape.size() != 1) {
        const int last_dimsize = ndims() == 2 ? dimsize(1) : dimsize(2);
        const int block_size = globs::q8_block_size;
        const int blocks_per_row = (last_dimsize % block_size == 0)
                                   ? last_dimsize / block_size
                                   : last_dimsize / block_size + 1;
        
        const int n_blocks = ndims() == 2
                             ? dimsize(0) * blocks_per_row
                             : dimsize(0) * dimsize(1) * blocks_per_row;

        alloc_bytes = n_blocks * sizeof(Q8Block);
    } else if (dtype == kQint4) {
        GTEN_ASSERT(ndims() == 2);
        GTEN_ASSERT(dimsize(1) % globs::q8_block_size == 0);
        const int blocks_per_row = dimsize(1) / globs::q8_block_size;
        const int n_blocks = dimsize(0) * blocks_per_row;

        alloc_bytes = n_blocks * sizeof(Q4Block);
    }
    else {
        alloc_bytes = numel * itemsize();
    }

    void* raw_data_ptr = std::malloc(alloc_bytes);
    GTEN_ASSERTM(raw_data_ptr, "Failed to allocate %dMB of memory.", alloc_bytes / 1000000);

    data_ptr_ = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(raw_data_ptr), tensor_data_deleter);
    storage_size_ = alloc_bytes;
    G_TensorMemAllocated += alloc_bytes;
}


// An empty deleter allows us to use external data storage that we do not own.
static void empty_deleter(uint8_t* ptr) {  }

/// TODO: Add lock on the data ptr. [Unwritable tensor]. 
Tensor::Tensor(const void* data_ptr, const std::vector<int>& shape, Dtype dtype)
    : dtype_{dtype}
{
    GTEN_ASSERTM(data_ptr != nullptr, "Expected a non-null pointer but got a nullptr.");
    uint8_t* real_ptr = (uint8_t*)data_ptr;
    // An empty deleter ensures we do not delete the data since we do not own it.
    data_ptr_ = std::shared_ptr<uint8_t>(real_ptr, empty_deleter);
    validate_shape(shape);
    shape_ = shape;
    set_strides_from_shape(shape);
    numel_ = numel_from_shape(shape);
    storage_size_ = 0;
}

void Tensor::validate_shape(const std::vector<int> &shape) const
{
    GTEN_ASSERTM(shape.size() != 0, "The given shape is empty.");
    GTEN_ASSERTM(shape.size() <= 3, "Shape with dimensions > 3 not supported.");
    for (int i = 0; i < int(shape.size()); i++) {
        if (shape[i] <= 0) {
            std::cerr << "err\n";
            GTEN_ASSERTM(false, "The value of dimension %d: %d of the given shape is invalid!", i, shape[i]);
        }
    }
}

int Tensor::numel_from_shape(const std::vector<int>& shape) const {
    int numel = 1;
    for (int size : shape) {
        numel = numel * size;
    }
    return numel;
}

static std::string shape_to_str(const std::vector<int>& shape)
{
    std::stringstream s;
    s << "(";
    for (int i = 0; i < int(shape.size()); i++) {
        s << shape[i];
        if (i != int(shape.size()) - 1) {
            s << ", ";
        }
    }
    s << ")";
    
    return s.str();
}

// Contigous only???
void Tensor::resize(const std::vector<int>& new_shape) {
    validate_shape(new_shape);
    const int new_size = numel_from_shape(new_shape) * itemsize();
    GTEN_ASSERTM(
        new_size <= storage_size_,
        "The new shape provided %s with cap=%d exceeds shape %s with cap=%d.",
        shape_to_str(new_shape).c_str(), new_size, shape_str().c_str(), storage_size_);
    shape_ = new_shape;
    set_strides_from_shape(new_shape);
    numel_ = numel_from_shape(new_shape);
}


void Tensor::set_strides_from_shape(const std::vector<int>& shape) {
    // 1-dim: 1
    // 2-dim: d2, 1
    // 3-dim: d2*d3, d3, 1
    switch (shape.size()) {
        case 1: {
            strides_ = {1};
        } break;
        case 2: {
            const int d1 = shape[1];
            strides_ = {d1, 1};
        } break;
        case 3: {
            const int d1 = shape[1];
            const int d2 = shape[2];
            strides_ = {d1*d2, d2, 1};
        } break;
    }
}

// Should we create and return a new tensor with the new shape?
Tensor Tensor::view(const std::vector<int>& new_shape) const {
    validate_shape(new_shape);
    const int new_numel = numel_from_shape(new_shape);
    const int old_numel = numel_from_shape(shape_);
    GTEN_ASSERTM(new_numel == old_numel, "New shape numel `%d` must be equal with old shape numel `%d`.", new_numel, old_numel);

    Tensor out = *this;
    out.shape_ = new_shape;
    out.set_strides_from_shape(new_shape);

    return out;
}


// Should we create and return a new tensor with the new shape?
Tensor Tensor::permute(const std::vector<int> &indices)
{
    GTEN_ASSERTM(indices.size() == shape_.size(),
                "The dims of indices `%ld` given do not match the tensor dims `%ld`.",
                indices.size(), shape_.size());

    std::vector<int> new_shape = shape_;
    std::vector<int> new_strides = strides_;
    for (int i = 0; i < int(indices.size()); i++) {
        const int idx = indices[i];
        new_shape[i] = shape_[idx];
        new_strides[i] = strides_[idx]; 
    }
    shape_ = std::move(new_shape);
    strides_ = std::move(new_strides);
    
    return *this;
}

void Tensor::set_strides(const std::vector<int>& strides)
{
    GTEN_ASSERTM(strides.size() == shape_.size(), "The given strides ndims must match shape ndims.");
    for (int i = 0; i < int(strides.size()); i++) {
    //     if (strides[i] <= 0) {
    //         GTEN_ASSERTM(false, "The stride at index %d, `%d` is invalid.", i, strides[i]);
    //     }
    }
    strides_ = strides;
}

std::string Tensor::shape_str() const
{
    std::stringstream s;
    s << "(";
    for (int i = 0; i < int(shape_.size()); i++) {
        s << shape_[i];
        if (i != int(shape_.size()) - 1) {
            s << ", ";
        }
    }
    s << ")";
    
    return s.str();
}

std::string Tensor::strides_str() const {
    std::stringstream s;
    s << "(";
    for (int i = 0; i < int(strides_.size()); i++) {
        s << strides_[i];
        if (i != int(strides_.size()) - 1) {
            s << ", ";
        }
    }
    s << ")";
    
    return s.str();
}

void Tensor::save(const std::string& path) const
{
    std::ofstream fout{path, std::ios_base::binary};
    GTEN_ASSERTM(fout.is_open(), "Failed to save tensor at %s.", path.c_str());
    fout.write(data_ptr<char>(), nbytes());
}

void print_vector(const std::vector<int>& vec) {
    std::cout << "(";
    for (int i = 0; i < int(vec.size()); i++) {
        std::cout << vec[i];
        if (i != int(vec.size()) - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")\n";
}

void Tensor::print_info() const {
    auto data = data_ptr<void>();
    std::cout << "\nTensor(\n"
              << "  dtype    : " << dtype_str(dtype_) << "\n"
              << "  shape    : ";
    print_vector(shape_);
    std::cout << "  strides  : ";
    print_vector(strides_);
    std::cout << "  numel    : " << numel_ << "\n"
            //   << "  numel cap: " << storage_size_/itemsize() << "\n"
              << "  capacity : " << storage_size_ << " bytes\n"
              << "  pointer  : "   << data << "\n)\n";
    
}

void Tensor::print_single(int item_idx, int row_idx, int col_idx, int n_cols) const
{
    uint32_t max_cols = dtype_ == kInt32 ? 32 : 8;
    if (dtype_ == kFloat16) {
        std::cout << std::fixed
                  << std::setprecision(4)
                  << std::setw(7)
                  << fp16_to_fp32(data_ptr<Float16>()[item_idx]);
    }
    else if (dtype_ == kFloat32) {
        std::cout << std::fixed
                  << std::setprecision(4)
                  << std::setw(7)
                  << data_ptr<float>()[item_idx];
    } else if (dtype_ == kQint8) {
        std::cout << std::fixed
                  << std::setprecision(4)
                  << std::setw(7)
                  << int(data_ptr<Qint8>()[item_idx]); /// TODO: FIXME
                //   << deq(data_ptr<Qint8>()[item_idx], delta_[row_idx], qzerop_[row_idx]);
    }
    else {
        std::cout << std::setw(2) << data_ptr<int>()[item_idx];
    }
    if (col_idx != n_cols - 1) {
        std::cout << ", ";
    }
    if (col_idx > 0 && (col_idx % max_cols) == 0) {
        std::cout << "\n  ";
    }
}

void Tensor::print() const
{
    std::cout << "\n[";
    const int ndims = shape_.size();
    if (ndims == 1) {
        for (int col = 0; col < numel_; col += 1)
            print_single(col, 0, col, numel_);
    }
    else if (ndims == 2) {
        const int rows = shape_[0];
        const int cols = shape_[1];
        const int st0 = strides_[0];
        const int st1 = strides_[1];
        for (int row = 0; row < rows; row++) {
            if (row == 0) std::cout << "[";
            else std::cout << " [";
            for (int col = 0; col < cols; col++) {
                const int idx = row * st0 + col * st1;
                print_single(idx, row, col, cols);
            }
            if (row != rows - 1) std::cout << "]\n";
            else std::cout << "]";
        }
    }
    else // ndims=3
    {
        const int chs = shape_[0];
        const int rows = shape_[1];
        const int cols = shape_[2];
        const int st0 = strides_[0];
        const int st1 = strides_[1];
        const int st2 = strides_[2];

        for (int ch = 0; ch < chs; ch++)
        {
            if (ch == 0) std::cout << "[";
            else std::cout << " [";
            for (int row = 0; row < rows; row++) {
                if (row == 0) std::cout << "[";
                else std::cout << "  [";
                for (int col = 0; col < cols; col++) {
                    const int idx = ch * st0 + row * st1 + col * st2;
                    print_single(idx, row, col, cols);
                }
                std::cout << "]";
                if (row != rows - 1)
                    std::cout << "\n";
            }
            std::cout << "]";
            if (ch != chs - 1)
                std::cout << "\n\n";
        }
        
    }

    std::cout << "]\n\n";
}

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor) {
    tensor.print();
    return stream;
}

} // namespace gten.
