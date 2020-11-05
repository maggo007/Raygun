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

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#include "payload.h"
#include "raytracer_bindings.h"

struct Sphere {
    vec3 center;
    float radius;
};

struct Aabb {
    vec3 minimum;
    vec3 maximum;
};

hitAttributeEXT vec3 attribs;
layout(binding = RAYGUN_RAYTRACER_BINDING_ACCELERATION_STRUCTURE, set = 0) uniform accelerationStructureEXT topLevelAS;

layout(binding = RAYGUN_RAYTRACER_BINDING_UNIFORM_BUFFER, set = 0) uniform UniformBufferObject{
#include "uniform_buffer_object.def"
} ubo;

struct Material {
#include "gpu_material.def"
};

layout(binding = RAYGUN_RAYTRACER_BINDING_MATERIAL_BUFFER, set = 0) buffer Materials
{
    Material m[];
}
materials;

struct InstanceOffsetTableEntry {
#include "instance_offset_table.def"
};

layout(binding = RAYGUN_RAYTRACER_BINDING_INSTANCE_OFFSET_TABLE, set = 0) buffer InstanceOffsetTable
{
    InstanceOffsetTableEntry e[];
}
instanceOffsetTable;

layout(binding = RAYGUN_RAYTRACER_BINDING_SPHERE_BUFFER, set = 0, scalar) buffer allSpheres_
{
    Sphere i[];
}
allSpheres;

void main()
{
    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    Sphere instance = allSpheres.i[gl_InstanceCustomIndexEXT];

    vec3 normal = normalize(worldPos - instance.center);

    vec3 diffuse = vec3(abs(instance.center.x) / 10.0f, abs(instance.center.y) / 10.0f, abs(instance.center.z) / 10.0f);

    // Computing the normal for a cube
    if(gl_HitKindEXT == KIND_CUBE) // Aabb
    {
        vec3 absN = abs(normal);
        float maxC = max(max(absN.x, absN.y), absN.z);
        normal = (maxC == absN.x) ? vec3(sign(normal.x), 0, 0) : (maxC == absN.y) ? vec3(0, sign(normal.y), 0) : vec3(0, 0, sign(normal.z));
    }

    float dot_product = max(dot(-ubo.lightDir, normal), 0.2);
    vec3 basecolor = diffuse * dot_product;

    payload.hitValue = basecolor;
    payload.normal = normal;
    payload.roughValue = vec4(basecolor, 0.3);
    payload.depth = gl_HitTEXT;
}
