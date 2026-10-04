#include "Trajectory.hh"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <gz/common/Console.hh>
#include <gz/math/Pose3.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <sdf/Element.hh>

namespace swarm
{
class SwarmPlayback final : public gz::sim::System,
                            public gz::sim::ISystemConfigure,
                            public gz::sim::ISystemPreUpdate,
                            public gz::sim::ISystemPostUpdate
{
  struct Drone
  {
    std::string name;
    Trajectory trajectory;
    gz::sim::Entity entity{gz::sim::kNullEntity};
  };

 public:
  void Configure(const gz::sim::Entity &entity,
                 const std::shared_ptr<const sdf::Element> &sdf,
                 gz::sim::EntityComponentManager &,
                 gz::sim::EventManager &) override
  {
    world = entity;
    try
    {
      if (!sdf->HasElement("drone"))
        throw std::runtime_error("No drone trajectories configured");
      auto config = sdf->Clone();
      auto entry = config->GetElement("drone");
      while (entry)
      {
        drones.push_back({entry->Get<std::string>("name"),
            Trajectory(entry->Get<std::string>("trajectory"))});
        entry = entry->GetNextElement("drone");
      }
      if (sdf->HasElement("telemetry"))
      {
        telemetry.open(sdf->Get<std::string>("telemetry"));
        if (!telemetry) throw std::runtime_error("Cannot open telemetry file");
        telemetry << "time,drone,x,y,z,yaw\n" << std::setprecision(10);
      }
      ready = true;
      gzmsg << "SwarmPlayback: " << drones.size()
            << " trajectories loaded (kinematic beta).\n";
    }
    catch (const std::exception &error)
    {
      gzerr << "SwarmPlayback: " << error.what() << "\n";
    }
  }

  void PreUpdate(const gz::sim::UpdateInfo &info,
                 gz::sim::EntityComponentManager &ecm) override
  {
    if (!ready || info.paused) return;
    const double time = std::chrono::duration<double>(info.simTime).count();
    for (auto &drone : drones)
    {
      if (drone.entity == gz::sim::kNullEntity)
      {
        drone.entity = ecm.EntityByComponents(gz::sim::components::Model(),
            gz::sim::components::Name(drone.name),
            gz::sim::components::ParentEntity(world));
        if (drone.entity == gz::sim::kNullEntity) continue;
      }
      const auto pose = drone.trajectory.At(time);
      gz::sim::Model(drone.entity).SetWorldPoseCmd(ecm,
          gz::math::Pose3d(pose[0], pose[1], pose[2], 0, 0, pose[3]));
    }
  }

  void PostUpdate(const gz::sim::UpdateInfo &info,
                  const gz::sim::EntityComponentManager &ecm) override
  {
    if (!ready || info.paused || !telemetry) return;
    const double time = std::chrono::duration<double>(info.simTime).count();
    // Simulation resets rewind the trajectory and restart logging immediately.
    if (time < lastLog) lastLog = -1.0;
    if (time - lastLog < 0.1 - 1e-8) return;
    lastLog = time;
    for (const auto &drone : drones)
    {
      if (drone.entity == gz::sim::kNullEntity) continue;
      const auto pose = gz::sim::worldPose(drone.entity, ecm);
      telemetry << time << ',' << drone.name << ',' << pose.Pos().X() << ','
                << pose.Pos().Y() << ',' << pose.Pos().Z() << ','
                << pose.Rot().Yaw() << '\n';
    }
    telemetry.flush();
  }

 private:
  gz::sim::Entity world{gz::sim::kNullEntity};
  bool ready{false};
  double lastLog{-1.0};
  std::vector<Drone> drones;
  std::ofstream telemetry;
};
}  // namespace swarm

GZ_ADD_PLUGIN(swarm::SwarmPlayback, gz::sim::System,
              swarm::SwarmPlayback::ISystemConfigure,
              swarm::SwarmPlayback::ISystemPreUpdate,
              swarm::SwarmPlayback::ISystemPostUpdate)
GZ_ADD_PLUGIN_ALIAS(swarm::SwarmPlayback, "swarm::SwarmPlayback")
