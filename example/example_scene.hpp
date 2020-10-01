#pragma once

#include "raygun/scene.hpp"
#include "raygun/ui/ui.hpp"

#include "aabb.hpp"
#include "ball.hpp"
#include "sphere.hpp"

// for added sphere https://nvpro-samples.github.io/vk_raytracing_tutorial/vkrt_tuto_intersection.md.html

class ExampleScene : public raygun::Scene {
  public:
    ExampleScene();

    void processInput(raygun::input::Input input, double timeDelta) override;
    void update(double timeDelta) override;
    void addMesh() override;
    std::vector<std::shared_ptr<Sphere>> m_spheres;
    std::vector<std::shared_ptr<Aabb>> m_aabb;

  private:
    static constexpr raygun::vec3 CAMERA_OFFSET = {5.0f, 10.0f, 10.0f};

    std::shared_ptr<Ball> m_ball;

    std::vector<std::shared_ptr<Ball>> m_ballVector;

    std::unique_ptr<raygun::ui::Factory> m_uiFactory;
    std::shared_ptr<raygun::ui::Window> m_menu;

    void showMenu();
};
