// gradbridge's TensorFlow op: calls a Python function on TensorFlow's own device buffers.
//
// tf.py_function always runs on the host, so TensorFlow copies GPU inputs to the CPU and the
// results back. GradbridgeCall has CPU and GPU kernels: TensorFlow allocates the outputs, and the
// kernel hands raw input/output pointers plus TensorFlow's CUDA stream to a callback registered
// from Python (a ctypes function, which takes the GIL itself). No Python headers are needed.
//
// Built on first use against the installed TensorFlow (see _tf_native.py).

#include <cstdint>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "xla/stream_executor/stream.h"

using namespace tensorflow;

extern "C" {
typedef int (*gradbridge_tf_callback)(int64_t callback_id, int n_in, void** in_ptrs, const int64_t* in_dims,
                                      const int* in_ranks, const int* in_dtypes, int n_out, void** out_ptrs,
                                      const int64_t* out_dims, const int* out_ranks, const int* out_dtypes,
                                      void* cuda_stream, int device_ordinal);

static gradbridge_tf_callback g_callback = nullptr;

void gradbridge_tf_set_callback(gradbridge_tf_callback cb) { g_callback = cb; }
}

REGISTER_OP("GradbridgeCall")
    .Input("ins: Tin")
    .Output("outs: Tout")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 1")
    .Attr("callback_id: int")
    .Attr("output_shapes: list(shape)")
    .SetIsStateful()  // opaque foreign code: never constant-fold or deduplicate it
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      std::vector<PartialTensorShape> shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &shapes));
      for (int i = 0; i < static_cast<int>(shapes.size()); ++i) {
        shape_inference::ShapeHandle h;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shapes[i], &h));
        c->set_output(i, h);
      }
      return absl::OkStatus();
    });

namespace {

void Describe(const Tensor& t, std::vector<void*>* ptrs, std::vector<int64_t>* dims, std::vector<int>* ranks,
              std::vector<int>* dtypes) {
  ptrs->push_back(const_cast<char*>(t.tensor_data().data()));
  ranks->push_back(t.dims());
  for (int d = 0; d < t.dims(); ++d) dims->push_back(t.dim_size(d));
  dtypes->push_back(static_cast<int>(t.dtype()));
}

template <bool kGpu>
class GradbridgeCallOp : public OpKernel {
 public:
  explicit GradbridgeCallOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("callback_id", &callback_id_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES(ctx, g_callback != nullptr, errors::Internal("gradbridge: TensorFlow callback not registered"));
    OpInputList ins;
    OP_REQUIRES_OK(ctx, ctx->input_list("ins", &ins));
    OpOutputList outs;
    OP_REQUIRES_OK(ctx, ctx->output_list("outs", &outs));

    std::vector<void*> in_ptrs, out_ptrs;
    std::vector<int64_t> in_dims, out_dims;
    std::vector<int> in_ranks, out_ranks, in_dtypes, out_dtypes;
    for (int i = 0; i < ins.size(); ++i) Describe(ins[i], &in_ptrs, &in_dims, &in_ranks, &in_dtypes);
    for (int i = 0; i < outs.size(); ++i) {
      TensorShape shape;
      OP_REQUIRES(ctx, output_shapes_[i].AsTensorShape(&shape),
                  errors::InvalidArgument("gradbridge: output shapes must be fully defined"));
      Tensor* t = nullptr;
      OP_REQUIRES_OK(ctx, outs.allocate(i, shape, &t));
      Describe(*t, &out_ptrs, &out_dims, &out_ranks, &out_dtypes);
    }

    void* stream = nullptr;
    int ordinal = -1;
    if (kGpu) {
      stream = ctx->op_device_context()->stream()->platform_specific_handle().stream;
      ordinal = ctx->device()->parsed_name().id;
    }
    const int rc = g_callback(callback_id_, static_cast<int>(in_ptrs.size()), in_ptrs.data(), in_dims.data(),
                              in_ranks.data(), in_dtypes.data(), static_cast<int>(out_ptrs.size()), out_ptrs.data(),
                              out_dims.data(), out_ranks.data(), out_dtypes.data(), stream, ordinal);
    OP_REQUIRES(ctx, rc == 0, errors::Internal("gradbridge: the bridged function raised (see the Python error)"));
  }

 private:
  int64_t callback_id_;
  std::vector<PartialTensorShape> output_shapes_;
};

}  // namespace

REGISTER_KERNEL_BUILDER(Name("GradbridgeCall").Device(DEVICE_CPU), GradbridgeCallOp<false>);
REGISTER_KERNEL_BUILDER(Name("GradbridgeCall").Device(DEVICE_GPU), GradbridgeCallOp<true>);
