// gradbridge's JAX consumer handler: an XLA FFI target that calls a Python function on XLA's own
// buffers.
//
// XLA invokes the handler with its input and output buffers (and, on CUDA, its stream). The
// handler reads them natively and calls a callback registered from Python (a ctypes function,
// which takes the GIL itself) with raw pointers, shapes and dtype codes. No Python call-frame
// parsing, no CUDA headers: the stream is passed through as an opaque pointer.
//
// Built on first use against the installed jaxlib (see _jax_native.py).

#include <cstdint>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

extern "C" {
typedef int (*gradbridge_jax_callback)(int64_t callback_id, int n_in, void** in_ptrs, const int64_t* in_dims,
                                       const int* in_ranks, const int* in_dtypes, int n_out, void** out_ptrs,
                                       const int64_t* out_dims, const int* out_ranks, const int* out_dtypes,
                                       void* cuda_stream, int device_ordinal);

static gradbridge_jax_callback g_callback = nullptr;

void gradbridge_jax_set_callback(gradbridge_jax_callback cb) { g_callback = cb; }
}

namespace {

void Describe(const ffi::AnyBuffer& b, std::vector<void*>* ptrs, std::vector<int64_t>* dims, std::vector<int>* ranks,
              std::vector<int>* dtypes) {
  ptrs->push_back(b.untyped_data());
  auto d = b.dimensions();
  ranks->push_back(static_cast<int>(d.size()));
  for (int64_t x : d) dims->push_back(x);
  dtypes->push_back(static_cast<int>(b.element_type()));
}

ffi::Error Call(void* stream, int32_t ordinal, int64_t callback_id, ffi::RemainingArgs args, ffi::RemainingRets rets) {
  if (g_callback == nullptr) return ffi::Error::Internal("gradbridge: JAX callback not registered");
  std::vector<void*> in_ptrs, out_ptrs;
  std::vector<int64_t> in_dims, out_dims;
  std::vector<int> in_ranks, out_ranks, in_dtypes, out_dtypes;
  for (size_t i = 0; i < args.size(); ++i) {
    auto arg = args.get<ffi::AnyBuffer>(i);
    if (arg.has_error()) return arg.error();
    Describe(*arg, &in_ptrs, &in_dims, &in_ranks, &in_dtypes);
  }
  for (size_t i = 0; i < rets.size(); ++i) {
    auto ret = rets.get<ffi::AnyBuffer>(i);
    if (ret.has_error()) return ret.error();
    Describe(**ret, &out_ptrs, &out_dims, &out_ranks, &out_dtypes);
  }
  const int rc = g_callback(callback_id, static_cast<int>(in_ptrs.size()), in_ptrs.data(), in_dims.data(),
                            in_ranks.data(), in_dtypes.data(), static_cast<int>(out_ptrs.size()), out_ptrs.data(),
                            out_dims.data(), out_ranks.data(), out_dtypes.data(), stream, ordinal);
  if (rc != 0) return ffi::Error::Internal("gradbridge: the bridged function raised (see the Python error)");
  return ffi::Error::Success();
}

ffi::Error CallCuda(void* stream, int32_t ordinal, int64_t callback_id, ffi::RemainingArgs args, ffi::RemainingRets rets) {
  return Call(stream, ordinal, callback_id, args, rets);
}

ffi::Error CallCpu(int64_t callback_id, ffi::RemainingArgs args, ffi::RemainingRets rets) {
  return Call(nullptr, -1, callback_id, args, rets);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(GradbridgeCallCuda, CallCuda,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<void*>>()
                                  .Ctx<ffi::DeviceOrdinal>()
                                  .Attr<int64_t>("callback_id")
                                  .RemainingArgs()
                                  .RemainingRets());

XLA_FFI_DEFINE_HANDLER_SYMBOL(GradbridgeCallCpu, CallCpu,
                              ffi::Ffi::Bind().Attr<int64_t>("callback_id").RemainingArgs().RemainingRets());
