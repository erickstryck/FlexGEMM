/*
 * Neighbor map for grid sample
 *
 * Copyright (C) 2025, Jianfeng XIANG <belljig@outlook.com>
 * All rights reserved.
 *
 * Licensed under The MIT License [see LICENSE for details]
 *
 * Written by Jianfeng XIANG
 */

#pragma once
#include <torch/extension.h>


#define BLOCK_SIZE 256


/**
 * Build the neighbor map for grid sample with nearest interpolation
 * 
 * @param hashmap   [2N] uint32 tensor containing the hashmap (key-value pairs)
 * @param coords    [N, 4] int32 tensor containing the coordinates of input features
 * @param grid      [B, L, 3] float tensor containing the grid to be sampled
 * @param W         the number of width dimensions
 * @param H         the number of height dimensions
 * @param D         the number of depth dimensions
 * 
 * @return          [B, L] uint32 tensor containing the neighboring indices
 */
torch::Tensor hashmap_build_grid_sample_3d_nearest_neighbor_map(
    torch::Tensor& hashmap,
    const torch::Tensor& coords,
    const torch::Tensor& grid,
    const int W,
    const int H,
    const int D
);


/**
 * Build the neighbor map and weights for grid sample with trilinear interpolation
 * 
 * @param hashmap   [2N] uint32 tensor containing the hashmap (key-value pairs)
 * @param coords    [N, 4] int32 tensor containing the coordinates of input features
 * @param grid      [B, L, 3] float tensor containing the grid to be sampled
 * @param W         the number of width dimensions
 * @param H         the number of height dimensions
 * @param D         the number of depth dimensions
 * 
 * @return
 *      [B, L, 8] uint32 tensor containing the neighboring indices
 *      [B, L, 8] float tensor containing the weights
 */
std::tuple<torch::Tensor, torch::Tensor> hashmap_build_grid_sample_3d_trilinear_neighbor_map_weight(
    torch::Tensor& hashmap,
    const torch::Tensor& coords,
    const torch::Tensor& grid,
    const int W,
    const int H,
    const int D
);
