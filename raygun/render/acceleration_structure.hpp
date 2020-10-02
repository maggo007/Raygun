// The MIT License (MIT)
//
// Copyright (c) 2019,2020 The Raygun Authors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#pragma once

#include "example/sphere.hpp"
#include "raygun/gpu/gpu_buffer.hpp"
#include "raygun/render/mesh.hpp"

namespace raygun {
struct Scene;
} // namespace raygun

namespace raygun::render {

class TopLevelAS {
  public:
    TopLevelAS(const vk::CommandBuffer& cmd, const Scene& scene, vk::BuildAccelerationStructureFlagBitsKHR updatebit);

    operator vk::AccelerationStructureKHR() const { return *m_structure; }

    const gpu::Buffer& instanceOffsetTable() const { return *m_instanceOffsetTable; }

    const vk::WriteDescriptorSetAccelerationStructureKHR& descriptorInfo() const { return m_descriptorInfo; }

    void updateTLAS(const vk::CommandBuffer& cmd, const Scene& scene);

  private:
    vk::WriteDescriptorSetAccelerationStructureKHR m_descriptorInfo = {};

    vk::UniqueAccelerationStructureKHR m_structure;
    vk::UniqueDeviceMemory m_memory;
    gpu::UniqueBuffer m_instances;
    gpu::UniqueBuffer m_scratch;

    /// Shader relevant data is commonly combined into large buffers (e.g. one
    /// vertex buffer for all vertices). In order to find the data corresponding
    /// to a specific primitive, the offsets in these buffers must be known.
    /// This lookup table provides the needed offsets for each instance.
    gpu::UniqueBuffer m_instanceOffsetTable;
};

using UniqueTopLevelAS = std::unique_ptr<TopLevelAS>;

class BottomLevelAS {
  public:
    BottomLevelAS(const vk::CommandBuffer& cmd, const Mesh& mesh, vk::BuildAccelerationStructureFlagBitsKHR updatebit);

    BottomLevelAS(const vk::CommandBuffer& cmd, const Sphere& sphere, vk::BuildAccelerationStructureFlagBitsKHR updatebit);

    operator vk::AccelerationStructureKHR() const { return *m_structure; }

    void updateBLAS(const vk::CommandBuffer& cmd, const Mesh& mesh);

    void updateBLAS(const vk::CommandBuffer& cmd, const Sphere& sphere);

  private:
    vk::UniqueAccelerationStructureKHR m_structure;
    vk::UniqueDeviceMemory m_memory;
    gpu::UniqueBuffer m_scratch;
};

using UniqueBottomLevelAS = std::unique_ptr<BottomLevelAS>;

struct InstanceOffsetTableEntry {
    using uint = uint32_t;
#include "resources/shaders/instance_offset_table.def"
};

void accelerationStructureBarrier(const vk::CommandBuffer& cmd);

} // namespace raygun::render
