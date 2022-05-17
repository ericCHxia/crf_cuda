#include "alg_cuda.cpp"
#include "score_cuda.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("alg_forward", &alg_forward, "alg forward (CUDA)");
    m.def("alg_backward", &alg_backward, "alg backward (CUDA)");
    m.def("score_forward", &score_forward, "score forward (CUDA)");
    m.def("score_backward", &score_backward, "score backward (CUDA)");
}