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

#include "raygun/render/raytracer.hpp"

#include "raygun/entity.hpp"
#include "raygun/gpu/gpu_utils.hpp"
#include "raygun/logging.hpp"
#include "raygun/profiler.hpp"
#include "raygun/raygun.hpp"
#include "raygun/utils/array_utils.hpp"

#include "resources/shaders/compute_shader_shared.def"
#include "resources/shaders/raytracer_bindings.h"

namespace raygun::render {

Raytracer::Raytracer() : vc(RG().vc())
{
    auto properties = vc.physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPropertiesNV>();
    raytracingProperties = properties.get<vk::PhysicalDeviceRayTracingPropertiesNV>();

    setupRaytracingDescriptorSet();

    setupPostprocessing();

    setupRaytracingImages();

    setupRaytracingPipeline();

    setupShaderBindingTable();

    RAYGUN_INFO("Raytracer initialized");
}

void Raytracer::setupBottomLevelAS()
{
    auto cmd = vc.computeQueue->createCommandBuffer();
    auto fence = vc.device->createFenceUnique({});

    cmd->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    auto& models = RG().resourceManager().models();
    for(auto& model: models) {
        if(!model->bottomLevelAS) {
            model->bottomLevelAS = std::make_unique<BottomLevelAS>(*cmd, *model->mesh);
        }
    }

    cmd->end();
    vc.computeQueue->submit(*cmd, *fence);
    vc.waitForFence(*fence);
}

void Raytracer::setupTopLevelAS(vk::CommandBuffer& cmd, const Scene& scene)
{
    RG().profiler().writeTimestamp(cmd, TimestampQueryID::ASBuildStart);

    m_topLevelAS = std::make_unique<TopLevelAS>(cmd, scene);

    accelerationStructureBarrier(cmd);

    RG().profiler().writeTimestamp(cmd, TimestampQueryID::ASBuildEnd);
}

const gpu::Image& Raytracer::doRaytracing(vk::CommandBuffer& cmd)
{
    cmd.bindPipeline(vk::PipelineBindPoint::eRayTracingNV, *m_pipeline);

    cmd.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingNV, *m_pipelineLayout, 0, m_descriptorSet.set(), {});

    RG().profiler().writeTimestamp(cmd, TimestampQueryID::RTTotalStart);

    RG().profiler().writeTimestamp(cmd, TimestampQueryID::RTOnlyStart);

    const auto stride = raytracingProperties.shaderGroupHandleSize;

    cmd.traceRaysNV(*m_sbtBuffer, m_raygenGroupIndex * stride,       //
                    *m_sbtBuffer, m_missGroupIndex * stride, stride, //
                    *m_sbtBuffer, m_hitGroupIndex * stride, stride,  //
                    *m_sbtBuffer, 0, 0,                              //
                    vc.windowSize.width, vc.windowSize.height, 1);

    RG().profiler().writeTimestamp(cmd, TimestampQueryID::RTOnlyEnd);

    RG().profiler().writeTimestamp(cmd, TimestampQueryID::PostprocStart);

    int dispatchWidth = vc.windowSize.width / COMPUTE_WG_X_SIZE + ((vc.windowSize.width % COMPUTE_WG_X_SIZE) > 0 ? 1 : 0);
    int dispatchHeight = vc.windowSize.height / COMPUTE_WG_Y_SIZE + ((vc.windowSize.height % COMPUTE_WG_Y_SIZE) > 0 ? 1 : 0);

    RG().profiler().writeTimestamp(cmd, TimestampQueryID::RoughStart);
    m_roughPrepare->dispatch(cmd, dispatchWidth, dispatchHeight);
    for(int i = 0; i < 10; ++i) {
        m_roughBlurH->dispatch(cmd, dispatchWidth, dispatchHeight);
        m_roughBlurV->dispatch(cmd, dispatchWidth, dispatchHeight);
    }
    RG().profiler().writeTimestamp(cmd, TimestampQueryID::RoughEnd);

    m_postprocess->dispatch(cmd, dispatchWidth, dispatchHeight);

    ImGui::Checkbox("Use FXAA", &m_useFXAA);
    if(m_useFXAA) {
        m_fxaa->dispatch(cmd, dispatchWidth, dispatchHeight);
        std::swap(m_baseImage, m_finalImage);
    }

    RG().profiler().writeTimestamp(cmd, TimestampQueryID::PostprocEnd);

    RG().profiler().writeTimestamp(cmd, TimestampQueryID::RTTotalEnd);

    return selectResultImage();
}

void Raytracer::updateRenderTarget(const gpu::Buffer& uniformBuffer, const gpu::Buffer& vertexBuffer, const gpu::Buffer& indexBuffer,
                                   const gpu::Buffer& materialBuffer)
{
    // Bind acceleration structure
    m_descriptorSet.bind(RAYGUN_RAYTRACER_BINDING_ACCELERATION_STRUCTURE, *m_topLevelAS);

    // Bind images
    m_descriptorSet.bind(RAYGUN_RAYTRACER_BINDING_OUTPUT_IMAGE, *m_baseImage);
    m_descriptorSet.bind(RAYGUN_RAYTRACER_BINDING_ROUGH_IMAGE, *m_roughImage);
    m_descriptorSet.bind(RAYGUN_RAYTRACER_BINDING_NORMAL_IMAGE, *m_normalImage);

    // Bind buffers
    m_descriptorSet.bind(RAYGUN_RAYTRACER_BINDING_UNIFORM_BUFFER, uniformBuffer);
    m_descriptorSet.bind(RAYGUN_RAYTRACER_BINDING_VERTEX_BUFFER, vertexBuffer);
    m_descriptorSet.bind(RAYGUN_RAYTRACER_BINDING_INDEX_BUFFER, indexBuffer);
    m_descriptorSet.bind(RAYGUN_RAYTRACER_BINDING_MATERIAL_BUFFER, materialBuffer);
    m_descriptorSet.bind(RAYGUN_RAYTRACER_BINDING_INSTANCE_OFFSET_TABLE, m_topLevelAS->instanceOffsetTable());

    m_descriptorSet.update();

    RG().computeSystem().updateDescriptors(
        uniformBuffer, {&*m_finalImage, &*m_baseImage, &*m_normalImage, &*m_roughImage, &*m_roughTransitions, &*m_roughColorsA, &*m_roughColorsB});
}

void Raytracer::imageShaderWriteBarrier(vk::CommandBuffer& cmd, vk::Image& image)
{
    vk::ImageMemoryBarrier barrier = {};
    barrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite);
    barrier.setDstAccessMask(vk::AccessFlagBits::eShaderWrite);
    barrier.setOldLayout(vk::ImageLayout::eUndefined);
    barrier.setNewLayout(vk::ImageLayout::eGeneral);
    barrier.setImage(image);
    barrier.setSubresourceRange(gpu::defaultImageSubresourceRange());

    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands, {}, {}, {}, {barrier});
}

void Raytracer::setupRaytracingImages()
{
    m_baseImage = std::make_unique<gpu::Image>(vc.windowSize);
    m_normalImage = std::make_unique<gpu::Image>(vc.windowSize);
    m_roughImage = std::make_unique<gpu::Image>(vc.windowSize);
    m_finalImage = std::make_unique<gpu::Image>(vc.windowSize);

    m_roughTransitions = std::make_unique<gpu::Image>(vc.windowSize, vk::Format::eR8Snorm);
    m_roughColorsA = std::make_unique<gpu::Image>(vc.windowSize);
    m_roughColorsB = std::make_unique<gpu::Image>(vc.windowSize);
}

void Raytracer::setupRaytracingDescriptorSet()
{
    m_descriptorSet.addBinding(RAYGUN_RAYTRACER_BINDING_ACCELERATION_STRUCTURE, 1, vk::DescriptorType::eAccelerationStructureNV,
                               vk::ShaderStageFlagBits::eRaygenNV | vk::ShaderStageFlagBits::eClosestHitNV);

    m_descriptorSet.addBinding(RAYGUN_RAYTRACER_BINDING_OUTPUT_IMAGE, 1, vk::DescriptorType::eStorageImage, vk::ShaderStageFlagBits::eRaygenNV);
    m_descriptorSet.addBinding(RAYGUN_RAYTRACER_BINDING_ROUGH_IMAGE, 1, vk::DescriptorType::eStorageImage, vk::ShaderStageFlagBits::eRaygenNV);
    m_descriptorSet.addBinding(RAYGUN_RAYTRACER_BINDING_NORMAL_IMAGE, 1, vk::DescriptorType::eStorageImage, vk::ShaderStageFlagBits::eRaygenNV);

    m_descriptorSet.addBinding(RAYGUN_RAYTRACER_BINDING_UNIFORM_BUFFER, 1, vk::DescriptorType::eUniformBuffer,
                               vk::ShaderStageFlagBits::eRaygenNV | vk::ShaderStageFlagBits::eClosestHitNV | vk::ShaderStageFlagBits::eMissNV);
    m_descriptorSet.addBinding(RAYGUN_RAYTRACER_BINDING_VERTEX_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eClosestHitNV);
    m_descriptorSet.addBinding(RAYGUN_RAYTRACER_BINDING_INDEX_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eClosestHitNV);
    m_descriptorSet.addBinding(RAYGUN_RAYTRACER_BINDING_MATERIAL_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eClosestHitNV);

    m_descriptorSet.addBinding(RAYGUN_RAYTRACER_BINDING_INSTANCE_OFFSET_TABLE, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eClosestHitNV);

    m_descriptorSet.generate();
}

namespace {

    vk::RayTracingShaderGroupCreateInfoNV generalShaderGroup(uint32_t generalShaderIndex)
    {
        vk::RayTracingShaderGroupCreateInfoNV info;
        info.setGeneralShader(generalShaderIndex);
        info.setClosestHitShader(VK_SHADER_UNUSED_NV);
        info.setAnyHitShader(VK_SHADER_UNUSED_NV);
        info.setIntersectionShader(VK_SHADER_UNUSED_NV);

        return info;
    }

    vk::RayTracingShaderGroupCreateInfoNV HitShaderGroup(uint32_t closestHitShaderIndex)
    {
        vk::RayTracingShaderGroupCreateInfoNV info{vk::RayTracingShaderGroupTypeNV::eTrianglesHitGroup};
        info.setGeneralShader(VK_SHADER_UNUSED_NV);
        info.setClosestHitShader(closestHitShaderIndex);
        info.setAnyHitShader(VK_SHADER_UNUSED_NV);
        info.setIntersectionShader(VK_SHADER_UNUSED_NV);

        return info;
    }

} // namespace

void Raytracer::setupRaytracingPipeline()
{
    m_shaderGroups.clear();

    std::vector<vk::PipelineShaderStageCreateInfo> stages;

    m_raygenGroupIndex = (uint32_t)m_shaderGroups.size();
    const auto raygenShader = RG().resourceManager().loadShader("raygen.rgen");
    {
        m_shaderGroups.push_back(generalShaderGroup((uint32_t)stages.size()));
        stages.push_back(raygenShader->shaderStageInfo(vk::ShaderStageFlagBits::eRaygenNV));
    }

    m_missGroupIndex = (uint32_t)m_shaderGroups.size();
    const auto missShader = RG().resourceManager().loadShader("miss.rmiss");
    {
        m_shaderGroups.push_back(generalShaderGroup((uint32_t)stages.size()));
        stages.push_back(missShader->shaderStageInfo(vk::ShaderStageFlagBits::eMissNV));
    }

    const auto shadowMissShader = RG().resourceManager().loadShader("shadowMiss.rmiss");
    {
        m_shaderGroups.push_back(generalShaderGroup((uint32_t)stages.size()));
        stages.push_back(shadowMissShader->shaderStageInfo(vk::ShaderStageFlagBits::eMissNV));
    }

    m_hitGroupIndex = (uint32_t)m_shaderGroups.size();
    const auto closestHitShader = RG().resourceManager().loadShader("closesthit.rchit");
    {
        m_shaderGroups.push_back(HitShaderGroup((uint32_t)stages.size()));
        stages.push_back(closestHitShader->shaderStageInfo(vk::ShaderStageFlagBits::eClosestHitNV));
    }

    {
        vk::PipelineLayoutCreateInfo info = {};
        info.setSetLayoutCount(1);
        info.setPSetLayouts(&m_descriptorSet.layout());

        m_pipelineLayout = vc.device->createPipelineLayoutUnique(info);
    }

    {
        vk::RayTracingPipelineCreateInfoNV info = {};
        info.setStageCount((uint32_t)stages.size());
        info.setPStages(stages.data());
        info.setGroupCount((uint32_t)m_shaderGroups.size());
        info.setPGroups(m_shaderGroups.data());
        info.setMaxRecursionDepth(7);
        info.setLayout(*m_pipelineLayout);

        m_pipeline = vc.device->createRayTracingPipelineNVUnique(nullptr, info);
    }
}

void Raytracer::setupShaderBindingTable()
{
    const auto sbtSize = m_shaderGroups.size() * raytracingProperties.shaderGroupHandleSize;

    m_sbtBuffer = std::make_unique<gpu::Buffer>(sbtSize, vk::BufferUsageFlagBits::eRayTracingNV, vk::MemoryPropertyFlagBits::eHostVisible);

    vc.device->getRayTracingShaderGroupHandlesNV(*m_pipeline, 0, (uint32_t)m_shaderGroups.size(), sbtSize, m_sbtBuffer->map());

    m_sbtBuffer->unmap();
}

void Raytracer::setupPostprocessing()
{
    auto& cs = RG().computeSystem();

    m_roughPrepare = cs.createComputePass("rough_prepare.comp");
    m_roughBlurH = cs.createComputePass("rough_blur_h.comp");
    m_roughBlurV = cs.createComputePass("rough_blur_v.comp");

    m_postprocess = cs.createComputePass("postprocess.comp");
    m_fxaa = cs.createComputePass("fxaa.comp");
}

const gpu::Image& Raytracer::selectResultImage()
{
    // For debugging purposes the result image can be selected via ImGui.

    const char* imageNames[] = {"Final", "Base/Temp", "Normal", "Rough", "RTransition", "RCA", "RCB"};
    gpu::Image* images[] = {m_finalImage.get(),       m_baseImage.get(),    m_normalImage.get(), m_roughImage.get(),
                            m_roughTransitions.get(), m_roughColorsA.get(), m_roughColorsB.get()};
    static_assert(RAYGUN_ARRAY_COUNT(imageNames) == RAYGUN_ARRAY_COUNT(images));

    static int selectedResult = 0;
    ImGui::Combo("Image", &selectedResult, imageNames, RAYGUN_ARRAY_COUNT(imageNames));

    return *images[selectedResult];
}

} // namespace raygun::render
