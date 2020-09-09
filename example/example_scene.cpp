#include "example_scene.hpp"

#include "raygun/assert.hpp"
#include "raygun/raygun.hpp"

using namespace raygun;
using namespace raygun::physics;

ExampleScene::ExampleScene()
{
    // setup level
    auto level = RG().resourceManager().loadEntity("room");
    // level->forEachEntity([](Entity& entity) {
    //    if(entity.model) {
    //        RG().physicsSystem().attachRigidStatic(entity, GeometryType::TriangleMesh);
    //    }
    //});
    root->addChild(level);

    // setup ball
    m_ball = std::make_shared<Ball>();
    m_ball->moveTo({3.0f, 0.0f, -3.0f});
    root->addChild(m_ball);

    // add random balls to scene https://stackoverflow.com/questions/686353/random-float-number-generation
    std::srand(1);
    float LO = -5.0;
    float HI = 5.0;
    for(size_t i = 0; i < 10; i++) {
        auto tempmesh = std::make_shared<Ball>();
        float r1 = LO + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (HI - LO)));
        float r2 = LO + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (HI - LO)));
        float r3 = LO + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (HI - LO)));
        tempmesh->moveTo(vec3(r1, r2, r3));
        m_ballVector.emplace_back(tempmesh);
        root->addChild(tempmesh);
    }

    // setup music
    // auto musicTrack = RG().resourceManager().loadSound("lone_rider");
    // RG().audioSystem().music().play(musicTrack);

    // setup ui stuff
    const auto font = RG().resourceManager().loadFont("NotoSans");
    m_uiFactory = std::make_unique<ui::Factory>(font);
}

void ExampleScene::processInput(raygun::input::Input input, double timeDelta)
{
    if(input.reload) {
        RG().loadScene(std::make_unique<ExampleScene>());
    }

    if(input.cancel) {
        showMenu();
    }

    auto inputDir = vec2{-input.dir.y, -input.dir.x};

    // Take camera direction into account.
    const auto cam2D = glm::normalize(vec2(CAMERA_OFFSET.x, CAMERA_OFFSET.z));
    const auto angle = glm::orientedAngle(vec2(0, 1), cam2D);
    inputDir = glm::rotate(inputDir, angle);

    const auto strength = 2000.0 * timeDelta;

    // auto rigidDynamic = dynamic_cast<physx::PxRigidDynamic*>(m_ball->physicsActor.get());
    // RAYGUN_ASSERT(rigidDynamic);
    // rigidDynamic->addTorque((float)strength * physx::PxVec3(inputDir.x, 0.f, inputDir.y), physx::PxForceMode::eIMPULSE);

    m_ball->move(raygun::vec3(inputDir.x, inputDir.y, 0.0));
}

void ExampleScene::update(double)
{
    camera->moveTo(m_ball->transform().position + CAMERA_OFFSET);
    camera->lookAt(m_ball->transform().position);

    // m_ball->update();
}

void ExampleScene::showMenu()
{
    m_menu = m_uiFactory->window("menu", "Menu");
    m_uiFactory->addWithLayout(*m_menu, ui::Layout(vec2(0.5, 0.2), vec2(0, 0.3)), [&](ui::Factory& f) {
        f.button("Continue", [&] { camera->removeChild(m_menu); });
        f.button("Quit", [] { RG().renderSystem().makeFade<render::FadeTransition>(0.4, []() { RG().quit(); }); });
    });
    m_menu->doLayout();
    m_menu->move(vec3{0.0f, 0.0f, -4.0f});
    m_menu->setAnimation(ScaleAnimation(0.25, vec3(1, 0, 1), vec3(1)));

    camera->addChild(m_menu);

    // Alternatively, you can spawn the test window to see all available
    // controls and layouts. Note that this window cannot be closed as no button
    // has an action associated with it.

    // camera->addChild(ui::uiTestWindow(*m_uiFactory));
}

void ExampleScene::addMesh() {}
