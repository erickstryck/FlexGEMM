#include <torch/extension.h>
#include "hash/api.h"
#include "grid_sample/api.h"
#include "spconv/api.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Hash functions
    m.def("hashmap_insert_cuda", &hashmap_insert_cuda);
    m.def("hashmap_lookup_cuda", &hashmap_lookup_cuda);
    m.def("hashmap_insert_3d_cuda", &hashmap_insert_3d_cuda);
    m.def("hashmap_lookup_3d_cuda", &hashmap_lookup_3d_cuda);
    m.def("hashmap_insert_3d_idx_as_val_cuda", &hashmap_insert_3d_idx_as_val_cuda);

    // Grid sample functions
    m.def("hashmap_build_grid_sample_3d_nearest_neighbor_map", &hashmap_build_grid_sample_3d_nearest_neighbor_map);
    m.def("hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight", &hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight);
   
    // Convolution functions
    m.def("hashmap_build_submanifold_conv_neighbour_map_cuda", &hashmap_build_submanifold_conv_neighbour_map_cuda);
    m.def("neighbor_map_post_process_for_masked_implicit_gemm_1", &neighbor_map_post_process_for_masked_implicit_gemm_1);
    m.def("neighbor_map_post_process_for_masked_implicit_gemm_2", &neighbor_map_post_process_for_masked_implicit_gemm_2);
}
