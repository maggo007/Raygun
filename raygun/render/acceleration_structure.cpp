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

#include "raygun/render/acceleration_structure.hpp"

//#include "example/sphere.hpp"
#include "raygun/gpu/gpu_utils.hpp"
#include "raygun/raygun.hpp"
#include "raygun/render/model.hpp"
#include "raygun/render/proc_model.hpp"
#include "raygun/scene.hpp"

namespace raygun::render {

namespace {

    vk::AccelerationStructureInstanceKHR instanceFromEntity(vk::Device device, const Entity& entity, uint32_t instanceId)
    {
        RAYGUN_ASSERT(entity.model->bottomLevelAS);

        vk::AccelerationStructureInstanceKHR instance = {};
        instance.setInstanceCustomIndex(instanceId);
        instance.setMask(0xff);
        instance.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleCullDisable);
        // instance.setInstanceShaderBindingTableRecordOffset(instanceId % 2);

        // 3x4 row-major affine transformation matrix.
        const auto transform = glm::transpose(entity.globalTransform().toMat4());
        memcpy(&instance.transform, &transform, sizeof(instance.transform));

        const auto blasAddress = device.getAccelerationStructureAddressKHR({vk::AccelerationStructureKHR(*entity.model->bottomLevelAS)});
        instance.setAccelerationStructureReference(blasAddress);

        return instance;
    }

    std::tuple<vk::UniqueAccelerationStructureKHR, vk::UniqueDeviceMemory, gpu::UniqueBuffer>
    createStructureMemoryScratch(const vk::AccelerationStructureCreateInfoKHR& createInfo)
    {
        VulkanContext& vc = RG().vc();

        auto structure = vc.device->createAccelerationStructureKHRUnique(createInfo);

        // allocate memory
        vk::UniqueDeviceMemory memory;
        {
            vk::AccelerationStructureMemoryRequirementsInfoKHR memInfo = {};
            memInfo.setAccelerationStructure(*structure);
            memInfo.setType(vk::AccelerationStructureMemoryRequirementsTypeKHR::eObject);

            auto memoryRequirements = vc.device->getAccelerationStructureMemoryRequirementsKHR(memInfo).memoryRequirements;

            vk::MemoryAllocateFlagsInfo allocInfoFlags = {};
            allocInfoFlags.setFlags(vk::MemoryAllocateFlagBits::eDeviceAddress);

            vk::MemoryAllocateInfo allocInfo = {};
            allocInfo.setAllocationSize(memoryRequirements.size);
            allocInfo.setMemoryTypeIndex(gpu::selectMemoryType(vc.physicalDevice, memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal));
            allocInfo.setPNext(&allocInfoFlags);

            memory = vc.device->allocateMemoryUnique(allocInfo);
            RAYGUN_DEBUG("Memory {} for Type:{} with Vertices:{}", allocInfo.allocationSize, ((int)createInfo.type == 0) ? "TOPLEVEL" : "BOTLEVEL",
                         createInfo.pGeometryInfos->maxVertexCount);
        }

        // bind memory
        {
            vk::BindAccelerationStructureMemoryInfoKHR bindInfo = {};
            bindInfo.setAccelerationStructure(*structure);
            bindInfo.setMemory(*memory);

            vc.device->bindAccelerationStructureMemoryKHR(bindInfo);
        }

        // setup scratch buffer
        gpu::UniqueBuffer scratch;
        {
            vk::AccelerationStructureMemoryRequirementsInfoKHR memInfo = {};
            memInfo.setAccelerationStructure(*structure);
            memInfo.setType(vk::AccelerationStructureMemoryRequirementsTypeKHR::eBuildScratch);

            auto memoryRequirements = vc.device->getAccelerationStructureMemoryRequirementsKHR(memInfo).memoryRequirements;

            scratch =
                std::make_unique<gpu::Buffer>(memoryRequirements.size, vk::BufferUsageFlagBits::eRayTracingKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                              vk::MemoryPropertyFlagBits::eDeviceLocal);
            // RAYGUN_DEBUG("Scratch memory{}", memoryRequirements.size);
        }

        return {std::move(structure), std::move(memory), std::move(scratch)};
    }

} // namespace

TopLevelAS::TopLevelAS(const vk::CommandBuffer& cmd, const Scene& scene, vk::BuildAccelerationStructureFlagBitsKHR updatebit)
{
    VulkanContext& vc = RG().vc();

    std::vector<vk::AccelerationStructureInstanceKHR> instances;
    std::vector<InstanceOffsetTableEntry> instanceOffsetTable;

    // Grab instances from scene.
    scene.root->forEachEntity([&](const Entity& entity) {
        // if set to invisible, do not descend to children
        if(!entity.isVisible()) return false;

        if(entity.transform().isZeroVolume()) return false;

        // if no model, then we skip this, but might still render children
        if(!entity.model) return true;

        const auto instance = instanceFromEntity(*vc.device, entity, (uint32_t)instances.size());
        instances.push_back(instance);

        const auto& vertexBufferRef = entity.model->mesh->vertexBufferRef;
        const auto& indexBufferRef = entity.model->mesh->indexBufferRef;
        const auto& materialBufferRef = entity.model->materialBufferRef;

        auto& entry = instanceOffsetTable.emplace_back();
        entry.vertexBufferOffset = vertexBufferRef.offsetInElements();
        entry.indexBufferOffset = indexBufferRef.offsetInElements();
        entry.materialBufferOffset = materialBufferRef.offsetInElements();

        return true;
    });

    RAYGUN_DEBUG("instances before spheres {}", instances.size());
    // adding sphere blas

    auto procModels = RG().resourceManager().procModels();
    // customindex because sphere buffer starts at index 0
    int customindex = 0;
    for(auto& model: procModels) {
        RAYGUN_ASSERT(model->bottomLevelAS);
        vk::AccelerationStructureInstanceKHR instance = {};
        instance.setInstanceCustomIndex(customindex++);
        instance.setMask(0xff);
        instance.setInstanceShaderBindingTableRecordOffset(1);
        instance.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleCullDisable);

        const auto blasAddress = vc.device->getAccelerationStructureAddressKHR({vk::AccelerationStructureKHR(*model->bottomLevelAS)});
        instance.setAccelerationStructureReference(blasAddress);

        constexpr auto identity = glm::identity<mat4>();
        const auto p = glm::transpose(glm::translate(identity, model->sphere->center));
        // memcpy(&instance.transform, &p, sizeof(instance.transform));

        // vk::TransformMatrixKHR transformation = std::array<std::array<float, 4>, 3>{
        //     // clang-format off
        // 1.0f, 0.0f, 0.0f, model->sphere->center.x,
        // 0.0f, 1.0f, 0.0f, model->sphere->center.y,
        // 0.0f, 0.0f, 1.0f, model->sphere->center.z
        //     // clang-format on
        // };

        vk::TransformMatrixKHR transformation = std::array<std::array<float, 4>, 3>{
            // clang-format off
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
            // clang-format on
        };

        instance.setTransform(transformation);
        instanceOffsetTable.emplace_back();

        instances.push_back(instance);
    }
    m_instances = gpu::copyToBuffer(instances, vk::BufferUsageFlagBits::eRayTracingKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress);
    m_instances->setName("Instances");

    m_instanceOffsetTable = gpu::copyToBuffer(instanceOffsetTable, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress);
    m_instanceOffsetTable->setName("Instance Offset Table");

    // setup
    {
        vk::AccelerationStructureCreateGeometryTypeInfoKHR geometryTypeInfo = {};
        geometryTypeInfo.setGeometryType(vk::GeometryTypeKHR::eInstances);
        geometryTypeInfo.setMaxPrimitiveCount((uint32_t)instances.size());

        vk::AccelerationStructureCreateInfoKHR createInfo = {};
        createInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel);
        createInfo.setFlags(updatebit);

        createInfo.setMaxGeometryCount((uint32_t)1);
        createInfo.setPGeometryInfos(&geometryTypeInfo);

        std::tie(m_structure, m_memory, m_scratch) = createStructureMemoryScratch(createInfo);

        vc.setObjectName(*m_structure, "TLAS Structure");
        vc.setObjectName(*m_memory, "TLAS Memory");
        m_scratch->setName("TLAS Scratch");
    }

    // build
    {
        vk::AccelerationStructureGeometryInstancesDataKHR instancesData = {};
        instancesData.setData(m_instances->address());

        vk::AccelerationStructureGeometryDataKHR geometryData = {};
        geometryData.setInstances(instancesData);

        std::array<vk::AccelerationStructureGeometryKHR, 1> geometries;
        geometries[0].setGeometry(geometryData);
        geometries[0].setGeometryType(vk::GeometryTypeKHR::eInstances);
        geometries[0].setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        const auto pGeometires = geometries.data();

        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo = {};
        buildInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel);
        buildInfo.setDstAccelerationStructure(*m_structure);
        buildInfo.setGeometryCount((uint32_t)geometries.size());
        buildInfo.setPpGeometries(&pGeometires);
        buildInfo.setFlags(updatebit);
        buildInfo.setScratchData(m_scratch->address());

        vk::AccelerationStructureBuildRangeInfoKHR offset = {};
        offset.setPrimitiveCount((uint32_t)instances.size());

        cmd.buildAccelerationStructuresKHR(buildInfo, &offset);
    }

    m_descriptorInfo.setAccelerationStructureCount(1);
    m_descriptorInfo.setPAccelerationStructures(&*m_structure);
}

void TopLevelAS::updateTLAS(const vk::CommandBuffer& cmd, const Scene& scene)
{
    VulkanContext& vc = RG().vc();

    std::vector<vk::AccelerationStructureInstanceKHR> instances;
    std::vector<InstanceOffsetTableEntry> instanceOffsetTable;

    // Grab instances from scene.
    scene.root->forEachEntity([&](const Entity& entity) {
        // if set to invisible, do not descend to children
        if(!entity.isVisible()) return false;

        if(entity.transform().isZeroVolume()) return false;

        // if no model, then we skip this, but might still render children
        if(!entity.model) return true;

        const auto instance = instanceFromEntity(*vc.device, entity, (uint32_t)instances.size());
        instances.push_back(instance);

        const auto& vertexBufferRef = entity.model->mesh->vertexBufferRef;
        const auto& indexBufferRef = entity.model->mesh->indexBufferRef;
        const auto& materialBufferRef = entity.model->materialBufferRef;

        auto& entry = instanceOffsetTable.emplace_back();
        entry.vertexBufferOffset = vertexBufferRef.offsetInElements();
        entry.indexBufferOffset = indexBufferRef.offsetInElements();
        entry.materialBufferOffset = materialBufferRef.offsetInElements();

        return true;
    });

    // adding spheres
    auto procModels = RG().resourceManager().procModels();
    // customindex because sphere buffer starts at index 0
    int customindex = 0;
    for(auto& model: procModels) {
        RAYGUN_ASSERT(model->bottomLevelAS);
        vk::AccelerationStructureInstanceKHR instance = {};
        instance.setInstanceCustomIndex(customindex++);
        instance.setMask(0xff);
        instance.setInstanceShaderBindingTableRecordOffset(1);
        instance.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleCullDisable);

        const auto blasAddress = vc.device->getAccelerationStructureAddressKHR({vk::AccelerationStructureKHR(*model->bottomLevelAS)});
        instance.setAccelerationStructureReference(blasAddress);

        constexpr auto identity = glm::identity<mat4>();
        const auto p = glm::transpose(glm::translate(identity, model->sphere->center));
        // memcpy(&instance.transform, &p, sizeof(instance.transform));

        // vk::TransformMatrixKHR transformation = std::array<std::array<float, 4>, 3>{
        //     // clang-format off
        // 1.0f, 0.0f, 0.0f, model->sphere->center.x,
        // 0.0f, 1.0f, 0.0f, model->sphere->center.y,
        // 0.0f, 0.0f, 1.0f, model->sphere->center.z
        //     // clang-format on
        // };

        vk::TransformMatrixKHR transformation = std::array<std::array<float, 4>, 3>{
            // clang-format off
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
            // clang-format on
        };

        instance.setTransform(transformation);
        instanceOffsetTable.emplace_back();

        instances.push_back(instance);
    }

    // RAYGUN_DEBUG("m_instances before size= {}", m_instances->size());

    auto instancesize = m_instances->size();
    bool rebuild = false;

    m_instances = gpu::copyToBuffer(instances, vk::BufferUsageFlagBits::eRayTracingKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress);
    m_instances->setName("Instances");

    // Rebuild if instances are not the same anymore
    if(instancesize != m_instances->size()) {
        rebuild = true;
    }

    // RAYGUN_DEBUG("m_instances after size= {}", m_instances->size());

    m_instanceOffsetTable = gpu::copyToBuffer(instanceOffsetTable, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress);
    m_instanceOffsetTable->setName("Instance Offset Table");

    // setup not needed for update except for instancenumber changes
    if(rebuild) {
        vk::AccelerationStructureCreateGeometryTypeInfoKHR geometryTypeInfo = {};
        geometryTypeInfo.setGeometryType(vk::GeometryTypeKHR::eInstances);
        geometryTypeInfo.setMaxPrimitiveCount((uint32_t)instances.size());

        vk::AccelerationStructureCreateInfoKHR createInfo = {};
        createInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel);
        createInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
        createInfo.setMaxGeometryCount((uint32_t)1);
        createInfo.setPGeometryInfos(&geometryTypeInfo);

        std::tie(m_structure, m_memory, m_scratch) = createStructureMemoryScratch(createInfo);

        vc.setObjectName(*m_structure, "TLAS Structure");
        vc.setObjectName(*m_memory, "TLAS Memory");
        m_scratch->setName("TLAS Scratch");
    }

    // build
    {
        vk::AccelerationStructureGeometryInstancesDataKHR instancesData = {};
        instancesData.setData(m_instances->address());

        vk::AccelerationStructureGeometryDataKHR geometryData = {};
        geometryData.setInstances(instancesData);

        std::array<vk::AccelerationStructureGeometryKHR, 1> geometries;
        geometries[0].setGeometry(geometryData);
        geometries[0].setGeometryType(vk::GeometryTypeKHR::eInstances);
        geometries[0].setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        const auto pGeometires = geometries.data();

        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo = {};
        buildInfo.setType(vk::AccelerationStructureTypeKHR::eTopLevel);
        buildInfo.setDstAccelerationStructure(*m_structure);
        buildInfo.setSrcAccelerationStructure(*m_structure);
        buildInfo.setGeometryCount((uint32_t)geometries.size());
        buildInfo.setPpGeometries(&pGeometires);
        buildInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate);
        if(!rebuild) buildInfo.setUpdate(true);
        buildInfo.setScratchData(m_scratch->address());

        vk::AccelerationStructureBuildRangeInfoKHR offset = {};
        offset.setPrimitiveCount((uint32_t)instances.size());

        cmd.buildAccelerationStructuresKHR(buildInfo, &offset);
    }

    m_descriptorInfo.setAccelerationStructureCount(1);
    m_descriptorInfo.setPAccelerationStructures(&*m_structure);
}

BottomLevelAS::BottomLevelAS(const vk::CommandBuffer& cmd, const Mesh& mesh, vk::BuildAccelerationStructureFlagBitsKHR updatebit)
{
    VulkanContext& vc = RG().vc();

    // setup
    {
        vk::AccelerationStructureGeometryKHR geometryTypeInfo = {};
        // vk::AccelerationStructureCreateGeometryTypeInfoKHR geometryTypeInfo = {};
        geometryTypeInfo.setGeometryType(vk::GeometryTypeKHR::eTriangles);
        geometryTypeInfo.setMaxPrimitiveCount((uint32_t)mesh.numFaces());
        geometryTypeInfo.setIndexType(vk::IndexType::eUint32);
        geometryTypeInfo.setMaxVertexCount((uint32_t)mesh.vertices.size());
        geometryTypeInfo.setVertexFormat(vk::Format::eR32G32B32Sfloat);

        vk::AccelerationStructureCreateInfoKHR createInfo = {};
        createInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
        createInfo.setFlags(updatebit);
        createInfo.setMaxGeometryCount(1);
        createInfo.setPGeometryInfos(&geometryTypeInfo);

        std::tie(m_structure, m_memory, m_scratch) = createStructureMemoryScratch(createInfo);

        vc.setObjectName(*m_structure, "BLAS Structure");
        vc.setObjectName(*m_memory, "BLAS Memory");
        m_scratch->setName("BLAS Scratch");
    }

    // build
    {
        vk::AccelerationStructureGeometryTrianglesDataKHR triangles = {};
        triangles.setVertexData(mesh.vertexBufferRef.bufferAddress);
        triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);
        triangles.setVertexStride(mesh.vertexBufferRef.elementSize);
        triangles.setIndexData(mesh.indexBufferRef.bufferAddress);
        triangles.setIndexType(vk::IndexType::eUint32);

        vk::AccelerationStructureBuildRangeInfoKHR offsetInfo = {};
        offsetInfo.setPrimitiveCount((uint32_t)mesh.numFaces());
        offsetInfo.setPrimitiveOffset(mesh.indexBufferRef.offsetInBytes);
        offsetInfo.setFirstVertex(mesh.vertexBufferRef.offsetInElements());

        vk::AccelerationStructureGeometryDataKHR geometryData = {};
        geometryData.setTriangles(triangles);

        std::array<vk::AccelerationStructureGeometryKHR, 1> geometries;
        geometries[0].setGeometry(geometryData);
        geometries[0].setGeometryType(vk::GeometryTypeKHR::eTriangles);
        geometries[0].setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        const auto pGeometires = geometries.data();

        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo = {};
        buildInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
        buildInfo.setFlags(updatebit);
        buildInfo.setDstAccelerationStructure(*m_structure);
        buildInfo.setGeometryCount((uint32_t)geometries.size());
        buildInfo.setPpGeometries(&pGeometires);
        buildInfo.setScratchData(m_scratch->address());

        cmd.buildAccelerationStructuresKHR(buildInfo, &offsetInfo);
    }
}

void BottomLevelAS::updateBLAS(const vk::CommandBuffer& cmd, const Mesh& mesh)
{
    VulkanContext& vc = RG().vc();

    // setup not needed for update
    // {
    //     vk::AccelerationStructureCreateGeometryTypeInfoKHR geometryTypeInfo = {};
    //     geometryTypeInfo.setGeometryType(vk::GeometryTypeKHR::eTriangles);
    //     geometryTypeInfo.setMaxPrimitiveCount((uint32_t)mesh.numFaces());
    //     geometryTypeInfo.setIndexType(vk::IndexType::eUint32);
    //     geometryTypeInfo.setMaxVertexCount((uint32_t)mesh.vertices.size());
    //     geometryTypeInfo.setVertexFormat(vk::Format::eR32G32B32Sfloat);

    //     vk::AccelerationStructureCreateInfoKHR createInfo = {};
    //     createInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
    //     createInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate);
    //     createInfo.setMaxGeometryCount(1);
    //     createInfo.setPGeometryInfos(&geometryTypeInfo);

    //     std::tie(m_structure, m_memory, m_scratch) = createStructureMemoryScratch(createInfo);

    //     vc.setObjectName(*m_structure, "BLAS Structure");
    //     vc.setObjectName(*m_memory, "BLAS Memory");
    //     m_scratch->setName("BLAS Scratch");
    // }

    // build
    {
        vk::AccelerationStructureGeometryTrianglesDataKHR triangles = {};
        triangles.setVertexData(mesh.vertexBufferRef.bufferAddress);
        triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);
        triangles.setVertexStride(mesh.vertexBufferRef.elementSize);
        triangles.setIndexData(mesh.indexBufferRef.bufferAddress);
        triangles.setIndexType(vk::IndexType::eUint32);

        vk::AccelerationStructureBuildRangeInfoKHR offsetInfo = {};
        offsetInfo.setPrimitiveCount((uint32_t)mesh.numFaces());
        offsetInfo.setPrimitiveOffset(mesh.indexBufferRef.offsetInBytes);
        offsetInfo.setFirstVertex(mesh.vertexBufferRef.offsetInElements());

        vk::AccelerationStructureGeometryDataKHR geometryData = {};
        geometryData.setTriangles(triangles);

        std::array<vk::AccelerationStructureGeometryKHR, 1> geometries;
        geometries[0].setGeometry(geometryData);
        geometries[0].setGeometryType(vk::GeometryTypeKHR::eTriangles);
        geometries[0].setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        const auto pGeometires = geometries.data();

        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo = {};
        buildInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
        buildInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate);
        buildInfo.setDstAccelerationStructure(*m_structure);
        buildInfo.setGeometryCount((uint32_t)geometries.size());
        buildInfo.setPpGeometries(&pGeometires);
        buildInfo.setScratchData(m_scratch->address());
        buildInfo.setUpdate(true);
        buildInfo.setSrcAccelerationStructure(*m_structure);
        cmd.buildAccelerationStructuresKHR(buildInfo, &offsetInfo);
    }
}

void accelerationStructureBarrier(const vk::CommandBuffer& cmd)
{
    vk::MemoryBarrier memoryBarrier = {};
    memoryBarrier.setSrcAccessMask(vk::AccessFlagBits::eAccelerationStructureReadKHR | vk::AccessFlagBits::eAccelerationStructureReadKHR);
    memoryBarrier.setDstAccessMask(vk::AccessFlagBits::eAccelerationStructureReadKHR | vk::AccessFlagBits::eAccelerationStructureReadKHR);

    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {},
                        {memoryBarrier}, {}, {});
}

BottomLevelAS::BottomLevelAS(const vk::CommandBuffer& cmd, const ProcModel& procmodel, vk::BuildAccelerationStructureFlagBitsKHR updatebit)
{
    VulkanContext& vc = RG().vc();

    // setup
    {
        vk::AccelerationStructureCreateGeometryTypeInfoKHR geometryTypeInfo = {};
        geometryTypeInfo.setGeometryType(vk::GeometryTypeKHR::eAabbs);
        geometryTypeInfo.setMaxPrimitiveCount((uint32_t)1);
        // geometryTypeInfo.setMaxPrimitiveCount((uint32_t)RG().resourceManager().spheres().size());

        geometryTypeInfo.setIndexType(vk::IndexType::eNoneKHR);
        geometryTypeInfo.setVertexFormat(vk::Format::eUndefined);
        geometryTypeInfo.setAllowsTransforms(VK_FALSE);
        geometryTypeInfo.setMaxVertexCount((uint32_t)0);

        vk::AccelerationStructureCreateInfoKHR createInfo = {};
        createInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
        createInfo.setFlags(updatebit);
        createInfo.setMaxGeometryCount((uint32_t)1);
        createInfo.setPGeometryInfos(&geometryTypeInfo);

        std::tie(m_structure, m_memory, m_scratch) = createStructureMemoryScratch(createInfo);

        vc.setObjectName(*m_structure, "BLAS Structure");
        vc.setObjectName(*m_memory, "BLAS Memory");
        m_scratch->setName("BLAS Scratch");

        // build

        vk::AccelerationStructureGeometryAabbsDataKHR aabbs = {};
        // aabbs.setData(RG().renderSystem().m_spheresAabbBuffer->address());
        aabbs.setData(procmodel.aabbBufferRef.bufferAddress);
        aabbs.setStride(sizeof(Aabb));

        std::array<vk::AccelerationStructureBuildRangeInfoKHR, 1> offsetInfo = {};
        offsetInfo[0].setFirstVertex(0);
        offsetInfo[0].setPrimitiveCount((uint32_t)1);
        offsetInfo[0].setPrimitiveOffset(procmodel.aabbBufferRef.offsetInBytes);
        // offsetInfo[0].setPrimitiveOffset(0);
        offsetInfo[0].setTransformOffset(0);

        const auto pOffsetInfo = offsetInfo.data();

        vk::AccelerationStructureGeometryDataKHR geometryData = {};
        geometryData.setAabbs(aabbs);

        std::array<vk::AccelerationStructureGeometryKHR, 1> geometries;
        geometries[0].setGeometry(geometryData);
        geometries[0].setGeometryType(vk::GeometryTypeKHR::eAabbs);
        geometries[0].setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        const auto pGeometires = geometries.data();

        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo = {};
        buildInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
        buildInfo.setFlags(updatebit);
        buildInfo.setDstAccelerationStructure(*m_structure);
        buildInfo.setGeometryCount((uint32_t)geometries.size());
        buildInfo.setPpGeometries(&pGeometires);
        buildInfo.setScratchData(m_scratch->address());

        cmd.buildAccelerationStructuresKHR(buildInfo, pOffsetInfo);
    }
}

void BottomLevelAS::updateBLAS(const vk::CommandBuffer& cmd, const ProcModel& procmodel)
{
    VulkanContext& vc = RG().vc();

    // setup
    // {
    //     vk::AccelerationStructureCreateGeometryTypeInfoKHR geometryTypeInfo = {};
    //     geometryTypeInfo.setGeometryType(vk::GeometryTypeKHR::eAabbs);
    //     geometryTypeInfo.setMaxPrimitiveCount((uint32_t)sizeof(Sphere));
    //     geometryTypeInfo.setIndexType(vk::IndexType::eNoneKHR);
    //     geometryTypeInfo.setVertexFormat(vk::Format::eUndefined);
    //     geometryTypeInfo.setAllowsTransforms(VK_FALSE);

    //     vk::AccelerationStructureCreateInfoKHR createInfo = {};
    //     createInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
    //     createInfo.setFlags(updatebit);
    //     createInfo.setMaxGeometryCount(1);
    //     createInfo.setPGeometryInfos(&geometryTypeInfo);

    //     std::tie(m_structure, m_memory, m_scratch) = createStructureMemoryScratch(createInfo);

    //     vc.setObjectName(*m_structure, "BLAS Structure");
    //     vc.setObjectName(*m_memory, "BLAS Memory");
    //     m_scratch->setName("BLAS Scratch");
    // }

    // build
    {
        vk::AccelerationStructureGeometryAabbsDataKHR aabbs = {};
        // aabbs.setData(RG().renderSystem().m_spheresAabbBuffer->address());
        aabbs.setData(procmodel.aabbBufferRef.bufferAddress);
        aabbs.setStride(sizeof(Aabb));

        std::array<vk::AccelerationStructureBuildRangeInfoKHR, 1> offsetInfo = {};
        offsetInfo[0].setFirstVertex(0);
        offsetInfo[0].setPrimitiveCount((uint32_t)1);
        offsetInfo[0].setPrimitiveOffset(procmodel.aabbBufferRef.offsetInBytes);
        // offsetInfo[0].setPrimitiveOffset(0);
        offsetInfo[0].setTransformOffset(0);

        const auto pOffsetInfo = offsetInfo.data();

        vk::AccelerationStructureGeometryDataKHR geometryData = {};
        geometryData.setAabbs(aabbs);

        std::array<vk::AccelerationStructureGeometryKHR, 1> geometries;
        geometries[0].setGeometry(geometryData);
        geometries[0].setGeometryType(vk::GeometryTypeKHR::eAabbs);
        geometries[0].setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        const auto pGeometires = geometries.data();

        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo = {};
        buildInfo.setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
        buildInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate);
        buildInfo.setDstAccelerationStructure(*m_structure);
        buildInfo.setGeometryCount((uint32_t)geometries.size());
        buildInfo.setPpGeometries(&pGeometires);
        buildInfo.setScratchData(m_scratch->address());
        buildInfo.setUpdate(true);
        buildInfo.setSrcAccelerationStructure(*m_structure);

        cmd.buildAccelerationStructuresKHR(buildInfo, pOffsetInfo);
    }
}

} // namespace raygun::render
