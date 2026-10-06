#include "Trajectory.hh"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>

int main()
{
  const auto path = std::filesystem::temp_directory_path() /
      ("swarm-trajectory-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()) + ".csv");
  {
    std::ofstream file(path);
    file << "time,x,y,z,yaw\n0,0,0,0,3\n1,2,4,6,4\n2,4,8,0,5\n";
  }
  swarm::Trajectory trajectory(path.string());
  assert(trajectory.At(-1)[0] == 0);
  assert(trajectory.At(0.5)[2] == 3);
  assert(trajectory.At(0.5)[3] == 3.5);  // Unwrapped yaw, no 2pi jump.
  assert(trajectory.At(0.999, false)[2] == 0);
  assert(trajectory.At(1.0, false)[2] == 6);
  assert(trajectory.At(1.5, false)[2] == 6);
  assert(trajectory.At(2.0, false)[2] == 0);
  assert(trajectory.At(10)[0] == 4);
  assert(trajectory.At(1.5)[2] == 3);
  assert(trajectory.At(0.5)[2] == 3);  // Gazebo reset/rewind.
  {
    std::ofstream file(path);
    file << "time,x,y,z,yaw\n0,0,0,0,0\n0,1,1,1,1\n";
  }
  bool rejected = false;
  try { swarm::Trajectory invalid(path.string()); }
  catch (const std::runtime_error &) { rejected = true; }
  assert(rejected);
  std::filesystem::remove(path);
  std::cout << "Trajectory interpolation, rewind and validation: OK\n";
}
